"""
The paper trading engine.

Two jobs, run in this order every session:

  1. MANAGE what is already open -- profit targets, stops, time stops. First,
     because an exit frees capital and because a time stop that slips by a day
     is exactly how a small winner becomes a large loser.
  2. ENTER new ideas from the most recent scan, if room remains.

Two design decisions carry most of the value:

REVALIDATION. The scan runs after the close; trading happens the next morning.
That is a 17-hour gap, and the scanner's prices are stale by then. So before
any order, the live chain is re-quoted and the package re-priced. If the market
has moved beyond a tolerance, the trade is SKIPPED and the reason recorded.
This is the honest test of whether an end-of-day signal survives to a tradeable
next-morning fill -- and it is the question a backtest can never answer.

WORKING THE LIMIT. The order starts at the mid and is improved in steps toward
a worst-acceptable price, then cancelled if still unfilled. It never crosses
the spread on arrival. This matters more than it sounds: on a spread collecting
$17, giving up two cents of slippage is 12% of the trade. An auto-trader that
market-orders its way in will destroy an edge the scanner correctly identified.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, datetime

from .broker.base import Broker, OrderLeg, occ_symbol, parse_occ
from .journal import MULT, Journal, JournalEntry


@dataclass
class TraderConfig:
    max_new_per_session: int = 2
    max_open_positions: int = 3
    one_position_per_underlying: bool = True

    # How far the market may have moved since the scan before we refuse.
    # Expressed as a fraction of the scan's own net price.
    max_price_drift: float = 0.20

    # Order working
    work_steps: int = 4            # price improvements before giving up
    poll_seconds: float = 20.0     # wait between improvements (0 in tests)
    concession: float = 0.80       # how far from mid toward natural we will go

    # Exit rules
    credit_profit: float = 0.50    # close credit trades at 50% of max credit
    credit_stop_mult: float = 2.0  # stop at 2x the credit received
    debit_profit: float = 0.60     # close debit spreads at 60% of max profit
    debit_stop: float = 0.50       # stop at -50% of the debit
    long_profit: float = 0.80      # single options: +80%
    time_stop_credit: int = 21
    time_stop_debit: int = 14

    dry_run: bool = False


@dataclass
class SessionReport:
    when: datetime = field(default_factory=datetime.now)
    entered: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    exits: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    account_equity: float = 0.0
    open_after: int = 0


def _legs_from_idea(idea: dict) -> list:
    """Scanner idea -> OCC-symbol order legs."""
    out = []
    for l in idea.get("legs", []):
        exp = l["expiry"]
        if isinstance(exp, str):
            exp = date.fromisoformat(exp[:10])
        out.append(OrderLeg(
            symbol=occ_symbol(idea["ticker"], exp, l["kind"], float(l["strike"])),
            side=l["action"], ratio=int(l.get("qty", 1))))
    return out


def _reverse(legs: list) -> list:
    return [OrderLeg(l.symbol, "sell" if l.side == "buy" else "buy", l.ratio)
            for l in legs]


def net_price(legs: list, quotes: dict, which: str = "mid") -> float:
    """
    Net package price as a DEBIT: positive = you pay, negative = you receive.
    which='mid' prices at the midpoint, 'natural' at the crossing price.
    """
    total = 0.0
    for l in legs:
        q = quotes.get(l.symbol)
        if q is None or (q.bid <= 0 and q.ask <= 0):
            raise KeyError(f"no quote for {l.symbol}")
        px = q.mid if which == "mid" else (q.ask if l.side == "buy" else q.bid)
        total += (px if l.side == "buy" else -px) * l.ratio
    return round(total, 4)


def stop_is_toothless(entry_net: float, stop_net: float,
                      max_loss: float, tol: float = 0.85) -> bool:
    """
    Is the stop so close to maximum loss that it does not protect anything?

    On narrow spreads with a small credit this happens constantly and it is
    easy to miss. A $1-wide spread collecting $0.29 has a max loss of $0.71,
    and the standard "exit at 2x the credit received" rule puts the stop at
    $0.87 to close -- just 13 cents short of losing the maximum. The rule was
    designed for spreads collecting about a third of their width; applied to a
    thin credit it is decoration.

    Knowing this changes how you size: a trade whose stop cannot fire is a
    max-loss-or-nothing bet, so it must be sized against the full width.
    """
    if max_loss <= 0:
        return False
    close_at_max = -entry_net + max_loss / MULT
    return close_at_max > 0 and stop_net >= tol * close_at_max


def compute_targets(entry_net: float, max_profit: float | None,
                    cfg: TraderConfig) -> tuple:
    """
    Closing-order net prices that take profit and stop out.
    Comparison is uniform for every structure: take profit when the cost to
    close falls to `target`, stop when it rises to `stop`.
    """
    if entry_net < 0:                       # opened for a credit
        credit = -entry_net
        return (round(credit * (1.0 - cfg.credit_profit), 2),
                round(credit * (1.0 + cfg.credit_stop_mult), 2))
    debit = entry_net                       # opened for a debit
    if max_profit and max_profit > 0:
        gain = max_profit / MULT            # per share
        target = -(debit + cfg.debit_profit * gain)
    else:                                   # unbounded upside (long option)
        target = -(debit * (1.0 + cfg.long_profit))
    return round(target, 2), round(-(debit * (1.0 - cfg.debit_stop)), 2)


class Trader:
    def __init__(self, broker: Broker, journal: Journal,
                 cfg: TraderConfig | None = None, log=print):
        self.broker = broker
        self.journal = journal
        self.cfg = cfg or TraderConfig()
        self.log = log

    # ------------------------------------------------------------------
    def _work_order(self, legs: list, qty: int, mid: float, natural: float,
                    tag: str, ceiling: float | None = None) -> tuple:
        """
        Place at the mid and improve toward `natural` in steps. Returns
        (order, filled_net or None). Never starts by crossing the spread.

        `ceiling` is a hard cap on the net price we will ever pay. It exists
        because the sum of each leg's bid/ask can exceed what the package can
        possibly be worth: crossing both sides of a $1-wide spread can cost
        $1.14, which is more than the maximum it can ever pay out. Without
        this cap a closing order can realise a loss LARGER than the defined
        max loss of a defined-risk position -- the one thing such a position
        is supposed to make impossible.
        """
        cfg = self.cfg
        worst = mid + cfg.concession * (natural - mid)
        if ceiling is not None:
            worst = min(worst, ceiling)
            if worst < mid:
                worst = mid
        start = round(mid, 2)
        order = self.broker.submit_multileg(legs, qty, start, tag=tag)
        if order.status == "rejected":
            return order, None
        if order.status == "filled":
            return order, order.filled_avg_price if order.filled_avg_price is not None else start

        for i in range(1, cfg.work_steps + 1):
            if cfg.poll_seconds:
                time.sleep(cfg.poll_seconds)
            cur = self.broker.get_order(order.id)
            if cur is None:
                break
            if cur.status == "filled":
                return cur, cur.filled_avg_price if cur.filled_avg_price is not None else cur.limit_price
            if not cur.is_open:
                return cur, None
            nxt = round(start + (worst - start) * i / cfg.work_steps, 2)
            try:
                order = self.broker.replace_order(order.id, nxt)
            except RuntimeError:
                break
            if order.status == "filled":
                return order, order.filled_avg_price if order.filled_avg_price is not None else nxt

        cur = self.broker.get_order(order.id)
        if cur and cur.status == "filled":
            return cur, cur.filled_avg_price if cur.filled_avg_price is not None else cur.limit_price
        if cur and cur.is_open:
            self.broker.cancel_order(cur.id)
        return (cur or order), None

    # ------------------------------------------------------------------ exits
    def manage_exits(self, rep: SessionReport) -> None:
        today = getattr(self.broker, "today", date.today())
        for e in list(self.journal.open_entries()):
            if e.status != "open" or e.entry_net is None:
                continue
            legs = [OrderLeg(l["symbol"], l["side"], int(l.get("ratio", 1)))
                    for l in e.legs]
            close_legs = _reverse(legs)
            try:
                quotes = self.broker.get_quotes([l.symbol for l in close_legs])
                close_mid = net_price(close_legs, quotes, "mid")
                close_nat = net_price(close_legs, quotes, "natural")
            except KeyError as exc:
                rep.errors.append(f"{e.id}: cannot quote to exit ({exc})")
                continue

            expiry = date.fromisoformat(e.expiry)
            dte = (expiry - today).days
            reason = ""
            if close_mid <= e.target_net:
                reason = "profit target"
            elif close_mid >= e.stop_net:
                reason = "stop loss"
            elif dte <= e.time_stop_dte:
                reason = f"time stop ({dte} DTE)"
            elif dte <= 0:
                reason = "expiry"
            if not reason:
                continue

            # Closing can never rationally cost more than the price at which
            # the position is already at its maximum loss.
            ceiling = None
            if e.max_loss > 0 and e.entry_net is not None:
                ceiling = round(-e.entry_net + e.max_loss / MULT, 2)
            order, fill = self._work_order(close_legs, e.qty, close_mid, close_nat,
                                           tag=f"x-{e.id}"[:48], ceiling=ceiling)
            if fill is None:
                # An exit that cannot be filled at or below the max-loss price is
                # HELD, not chased. For a defined-risk spread the loss is already
                # capped by the structure, so paying above that price to escape
                # would realise a bigger loss than the position can suffer. The
                # attempt is recorded and retried next session rather than buried.
                e.exit_attempts += 1
                e.pending_exit_reason = reason
                e.notes.append(f"{today}: exit triggered ({reason}) but no fill at or "
                               f"below the max-loss price; holding, will retry")
                rep.exits.append({"id": e.id, "ticker": e.ticker, "reason": reason,
                                  "status": "unfilled", "attempts": e.exit_attempts})
                self.log(f"  HOLD  {e.ticker:<6} {reason} triggered but unfillable "
                         f"at/below max loss (attempt {e.exit_attempts}) -- holding")
                continue
            e.exit_order_id = order.id
            e.exit_net = fill
            e.closed_on = str(today)
            e.exit_reason = reason
            e.status = "closed"
            e.pending_exit_reason = ""
            e.pnl = e.compute_pnl()
            rep.exits.append({"id": e.id, "ticker": e.ticker, "reason": reason,
                              "pnl": e.pnl, "exit_net": fill})
            self.log(f"  EXIT  {e.ticker:<6} {e.strategy:<24} {reason:<18} "
                     f"P&L ${e.pnl:+,.2f}")

    # ----------------------------------------------------------------- entry
    def enter_new(self, ideas: list, rep: SessionReport) -> None:
        cfg = self.cfg
        today = getattr(self.broker, "today", date.today())
        n_open = len(self.journal.open_entries())
        added = 0

        for idea in sorted(ideas, key=lambda i: -float(i.get("score", 0))):
            if added >= cfg.max_new_per_session:
                break
            if n_open + added >= cfg.max_open_positions:
                rep.skipped.append({"ticker": idea.get("ticker"),
                                    "reason": f"already at {cfg.max_open_positions} open positions"})
                continue
            ticker = idea["ticker"]
            tid = f"{ticker}-{idea.get('expiry')}-{idea.get('strategy')}"
            if self.journal.by_id(tid):
                continue
            if cfg.one_position_per_underlying and self.journal.has_open_in(ticker):
                rep.skipped.append({"ticker": ticker,
                                    "reason": "already holding a position in this underlying"})
                continue
            if int(idea.get("contracts", 0)) < 1:
                rep.skipped.append({"ticker": ticker, "reason": "scanner sized it at 0 contracts"})
                continue

            legs = _legs_from_idea(idea)
            try:
                quotes = self.broker.get_quotes([l.symbol for l in legs])
                live_mid = net_price(legs, quotes, "mid")
                live_nat = net_price(legs, quotes, "natural")
            except KeyError as exc:
                rep.skipped.append({"ticker": ticker, "reason": f"no live quote ({exc})"})
                continue

            # --- the revalidation gate -------------------------------------
            scan_net = -float(idea.get("net_cash", 0)) / MULT   # net_cash is +credit
            drift = abs(live_mid - scan_net)
            tol = max(cfg.max_price_drift * abs(scan_net), 0.03)
            if drift > tol:
                rep.skipped.append({
                    "ticker": ticker,
                    "reason": (f"price moved since the scan: ${scan_net:+.2f} -> "
                               f"${live_mid:+.2f} (drift ${drift:.2f} > ${tol:.2f} "
                               f"tolerance). The edge the scanner measured is not "
                               f"the trade available now.")})
                self.log(f"  SKIP  {ticker:<6} price drifted {scan_net:+.2f} -> {live_mid:+.2f}")
                continue

            qty = int(idea.get("contracts", 1))
            order, fill = self._work_order(legs, qty, live_mid, live_nat, tag=tid[:48])

            entry = JournalEntry(
                id=tid, ticker=ticker, strategy=idea.get("strategy", ""),
                opened_on=str(today), expiry=str(idea.get("expiry"))[:10],
                legs=[{"symbol": l.symbol, "side": l.side, "ratio": l.ratio} for l in legs],
                qty=qty,
                score=float(idea.get("score", 0)), pop=float(idea.get("pop", 0)),
                pop_at_rv=float(idea.get("pop_at_rv", 0) or 0),
                breakeven_win_rate=float(idea.get("required_pop", 0) or 0),
                ev_at_rv=float(idea.get("ev_at_rv", 0) or 0),
                scan_net=round(scan_net, 2),
                iv_rank=idea.get("iv_rank"), iv_rv_ratio=idea.get("iv_rv_ratio"),
                entry_order_id=order.id,
                max_loss=float(idea.get("max_loss", 0) or 0),
                time_stop_dte=(self.cfg.time_stop_credit if scan_net < 0
                               else self.cfg.time_stop_debit))

            if fill is None:
                entry.status = "abandoned"
                entry.notes.append("no fill while working the limit; order cancelled")
                self.journal.add(entry)
                rep.skipped.append({"ticker": ticker,
                                    "reason": "worked the limit to the worst acceptable "
                                              "price without a fill"})
                self.log(f"  NOFILL {ticker:<6} worked to {live_mid:+.2f}.."
                         f"{live_mid + cfg.concession*(live_nat-live_mid):+.2f}, no fill")
                continue

            entry.entry_net = fill
            entry.entry_slippage = round((fill - scan_net) * MULT * qty, 2)
            entry.status = "open"
            entry.target_net, entry.stop_net = compute_targets(
                fill, float(idea.get("max_profit") or 0) or None, cfg)
            if stop_is_toothless(fill, entry.stop_net, entry.max_loss):
                msg = (f"stop at ${entry.stop_net:.2f} is within 10% of max loss "
                       f"(${entry.max_loss:,.0f}) -- it will rarely fire before the "
                       f"trade is already fully lost. Treat this as a "
                       f"max-loss-or-nothing position.")
                entry.notes.append(msg)
                rep.errors.append(f"{tid}: {msg}")
                self.log(f"  NOTE  {ticker:<6} {msg[:76]}")
            self.journal.add(entry)
            rep.entered.append({"id": tid, "ticker": ticker, "qty": qty,
                                "net": fill, "slippage": entry.entry_slippage,
                                "score": entry.score})
            added += 1
            self.log(f"  ENTER {ticker:<6} {entry.strategy:<24} {qty}x @ "
                     f"${fill:+.2f}  (scan ${scan_net:+.2f}, slip "
                     f"${entry.entry_slippage:+.2f})  score {entry.score:.0f}")

    # ------------------------------------------------------------------- run
    def run(self, ideas: list | None = None) -> SessionReport:
        rep = SessionReport()
        try:
            acct = self.broker.get_account()
            rep.account_equity = acct.equity
        except RuntimeError as exc:
            rep.errors.append(f"account: {exc}")

        self.log("Managing open positions...")
        self.manage_exits(rep)
        if ideas:
            self.log("Considering new ideas...")
            self.enter_new(ideas, rep)
        self.journal.save()
        rep.open_after = len(self.journal.open_entries())
        return rep
