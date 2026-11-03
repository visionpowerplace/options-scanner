"""
Offline paper broker: a fill simulator implementing the same Broker interface.

Why this is not optional. The trading loop is the riskiest code in the package
-- it places orders. Testing it only against a live broker would mean the first
time the exit logic runs is on a real (paper) account, with no way to
reproduce a bug or fast-forward three weeks to a time stop. This simulator
makes the whole loop deterministic and instant.

THE FILL MODEL is the part that matters, and it is deliberately pessimistic.

Express every package as a NET DEBIT (positive = you pay, negative = you
receive). Then for any structure:

    natural = what you get by crossing the spread (buy at ask, sell at bid)
    mid     = the theoretical mid of the package
    natural >= mid, always -- crossing is always worse for you

An order fills only when its limit reaches a threshold part-way from mid
towards natural:

    fill if  limit >= mid + aggression x (natural - mid)

`aggression` is how far you must concede to get filled: 0 means mid fills
always (fantasy), 1 means only crossing fills (worst case). The default 0.35
says you give up about a third of the half-spread, which is realistic for
liquid options worked patiently.

This matters because it is exactly the thing an optimistic backtest gets
wrong. A system that assumes mid fills will look profitable on spreads whose
entire edge is smaller than the bid/ask.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from ..core.bs import bs_price
from .base import Account, Broker, Order, OrderLeg, Position, Quote, parse_occ


@dataclass
class SimConfig:
    aggression: float = 0.35      # fraction of mid->natural you must concede
    reject_rate: float = 0.0      # fraction of orders the "exchange" rejects
    partial_rate: float = 0.0     # fraction that fill partially first
    seed: int = 7


class SimulatedBroker(Broker):
    name = "simulator"
    is_paper = True

    def __init__(self, provider, starting_equity: float = 500.0,
                 as_of: date | None = None, cfg: SimConfig | None = None):
        self.provider = provider
        self.cfg = cfg or SimConfig()
        self.rng = random.Random(self.cfg.seed)
        self.today = as_of or date.today()
        self.equity = starting_equity
        self.cash = starting_equity
        self.realized = 0.0
        self._positions: dict[str, Position] = {}
        self._orders: dict[str, Order] = {}
        self._seq = 0
        self.price_shift: dict[str, float] = {}   # ticker -> cumulative % move
        self.fills: list = []

    # ------------------------------------------------------------- pricing
    def _spot(self, ticker: str) -> float:
        return self.provider.get_spot(ticker) * (1.0 + self.price_shift.get(ticker, 0.0))

    def _theo(self, symbol: str) -> float:
        p = parse_occ(symbol)
        S = self._spot(p["underlying"])
        dte = max((p["expiry"] - self.today).days, 0)
        T = dte / 365.0
        iv = self.provider._iv(p["underlying"], S, p["strike"], max(T, 1 / 365.0))
        if T <= 0:
            return max(0.0, S - p["strike"]) if p["kind"] == "call" else max(0.0, p["strike"] - S)
        return bs_price(S, p["strike"], T, 0.0425,
                        self.provider.get_dividend_yield(p["underlying"]), iv, p["kind"])

    def get_quotes(self, symbols: list) -> dict:
        out = {}
        for s in symbols:
            theo = self._theo(s)
            p = parse_occ(s)
            dte = max((p["expiry"] - self.today).days, 1)
            moneyness = abs(math.log(max(p["strike"], 0.01) / max(self._spot(p["underlying"]), 0.01)))
            rel = 0.012 + 0.10 * moneyness + 0.02 * math.exp(-dte / 12.0)
            half = max(0.01, min(theo * rel, max(0.02, theo * 0.45)))
            tick = 0.05 if theo >= 3.0 else 0.01
            bid = max(0.0, round((theo - half) / tick) * tick)
            ask = round((theo + half) / tick) * tick
            if ask <= bid:
                ask = bid + tick
            out[s] = Quote(symbol=s, bid=round(bid, 2), ask=round(ask, 2), last=round(theo, 2))
        return out

    # --------------------------------------------------------- net pricing
    def _net(self, legs: list, quotes: dict, which: str) -> float:
        """
        Net package price as a DEBIT (positive = pay, negative = receive).
        which='mid' uses mids; which='natural' uses the crossing price.
        """
        total = 0.0
        for l in legs:
            q = quotes[l.symbol]
            if which == "mid":
                px = q.mid
            else:
                px = q.ask if l.side == "buy" else q.bid
            total += (px if l.side == "buy" else -px) * l.ratio
        return total

    # ------------------------------------------------------------- account
    def get_account(self) -> Account:
        unreal = sum(self._mark(p) for p in self._positions.values())
        return Account(equity=self.cash + unreal, cash=self.cash,
                       buying_power=self.cash, options_level=3, is_paper=True)

    def _mark(self, pos: Position) -> float:
        q = self.get_quotes([pos.symbol])[pos.symbol]
        return (q.mid - pos.avg_price) * pos.qty * 100.0

    def get_positions(self) -> list:
        out = []
        for p in self._positions.values():
            q = self.get_quotes([p.symbol])[p.symbol]
            p.market_price = q.mid
            p.unrealized_pl = self._mark(p)
            out.append(p)
        return out

    def get_orders(self, status: str = "open") -> list:
        if status == "open":
            return [o for o in self._orders.values() if o.is_open]
        if status in ("all", "closed"):
            return list(self._orders.values())
        return [o for o in self._orders.values() if o.status == status]

    # -------------------------------------------------------------- orders
    def submit_multileg(self, legs: list, qty: int, limit_price: float,
                        tag: str = "") -> Order:
        self._seq += 1
        oid = f"sim-{self._seq:05d}"
        o = Order(id=oid, status="new", legs=list(legs), qty=int(qty),
                  limit_price=float(limit_price), tag=tag,
                  submitted_at=datetime.now())
        if self.rng.random() < self.cfg.reject_rate:
            o.status = "rejected"
        self._orders[oid] = o
        self._try_fill(o)
        return o

    def replace_order(self, order_id: str, limit_price: float) -> Order:
        o = self._orders.get(order_id)
        if o is None or not o.is_open:
            raise RuntimeError(f"order {order_id} is not open")
        o.limit_price = float(limit_price)
        self._try_fill(o)
        return o

    def get_order(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)

    def cancel_order(self, order_id: str) -> bool:
        o = self._orders.get(order_id)
        if o is None or not o.is_open:
            return False
        o.status = "canceled"
        return True

    def _try_fill(self, o: Order) -> None:
        """Apply the fill rule described in this module's docstring."""
        if not o.is_open:
            return
        quotes = self.get_quotes([l.symbol for l in o.legs])
        mid = self._net(o.legs, quotes, "mid")
        natural = self._net(o.legs, quotes, "natural")
        threshold = mid + self.cfg.aggression * (natural - mid)
        if o.limit_price + 1e-9 < threshold:
            return                      # price not good enough; keep working

        remaining = o.qty - o.filled_qty
        n = remaining
        if self.cfg.partial_rate and remaining > 1 and self.rng.random() < self.cfg.partial_rate:
            n = max(1, remaining // 2)

        for l in o.legs:
            signed = n * l.ratio * (1 if l.side == "buy" else -1)
            # Attribute the net package price to the legs by their mid weight.
            leg_px = quotes[l.symbol].mid
            self._apply_fill(l.symbol, signed, leg_px)
        # Cash moves by the NET package price actually paid/received.
        self.cash -= o.limit_price * n * 100.0

        o.filled_qty += n
        o.filled_avg_price = o.limit_price
        o.status = "filled" if o.filled_qty >= o.qty else "partially_filled"
        self.fills.append({"order": o.id, "tag": o.tag, "qty": n,
                           "net": o.limit_price, "when": self.today.isoformat()})

    def _apply_fill(self, symbol: str, signed_qty: int, price: float) -> None:
        pos = self._positions.get(symbol)
        if pos is None:
            self._positions[symbol] = Position(symbol=symbol, qty=signed_qty,
                                               avg_price=price)
            return
        new_qty = pos.qty + signed_qty
        if new_qty == 0:
            del self._positions[symbol]
            return
        if (pos.qty > 0) == (signed_qty > 0):     # adding to the same side
            pos.avg_price = ((pos.avg_price * abs(pos.qty) + price * abs(signed_qty))
                             / abs(new_qty))
        pos.qty = new_qty

    # ------------------------------------------------------- time travel
    def advance(self, days: int = 1, moves: dict | None = None) -> None:
        """
        Move the clock and optionally move underlyings, then re-test every
        working order. This is what makes exits testable: a 21-DTE time stop
        can be reached in a millisecond instead of three weeks.
        `moves` is {ticker: fractional move}, applied cumulatively.
        """
        self.today += timedelta(days=days)
        for tk, mv in (moves or {}).items():
            self.price_shift[tk] = self.price_shift.get(tk, 0.0) + mv
        for o in list(self._orders.values()):
            if o.is_open:
                self._try_fill(o)

    def market_is_open(self) -> bool:
        return self.today.weekday() < 5
