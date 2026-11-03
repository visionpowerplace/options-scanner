"""
Strategy construction and position analytics.

Design decision worth stating. Most option tools hard-code a P&L formula per
strategy ("max loss = width - credit", and so on). That duplicates logic,
quietly breaks when a strike snaps to a different ladder than intended, and
gives no way to add a structure without new formulas to get wrong.

Instead, every position here is just a list of legs, and ALL analytics are
derived from the position's payoff function:

    P&L(S) = sum over legs of  +/- intrinsic(S)  -  net entry price

That function is piecewise-linear with corners only at the strikes, so:
  * max profit / max loss are found exactly by evaluating at the corners and
    checking the slope beyond the outermost one (which is how unlimited-upside
    positions are detected rather than assumed),
  * breakevens are found exactly by linear interpolation inside each segment,
  * probability of profit is the lognormal measure of the profitable segments,
    not an approximation like "1 - short delta",
  * expected value integrates the same function against the terminal density.

One code path, exact for any structure, and verified against the textbook
closed-form results in tests/test_strategies.py.

Costs are modelled, not ignored: fills are assumed to be worse than the mid by
a configurable fraction of the spread, and round-trip commissions are charged.
A scanner that reports mid-to-mid P&L is lying to you about small credits.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date

import numpy as np

from .bs import expected_payoff, greeks, prob_above, prob_touch
from .chain import ChainView

MULT = 100          # shares per US equity option contract

# Strategy identifiers
LONG_CALL = "long_call"
LONG_PUT = "long_put"
CALL_DEBIT = "call_debit_spread"
PUT_DEBIT = "put_debit_spread"
PUT_CREDIT = "put_credit_spread"
CALL_CREDIT = "call_credit_spread"
IRON_CONDOR = "iron_condor"
CASH_SECURED_PUT = "cash_secured_put"
COVERED_CALL = "covered_call"

PRETTY = {
    LONG_CALL: "Long Call",
    LONG_PUT: "Long Put",
    CALL_DEBIT: "Bull Call Debit Spread",
    PUT_DEBIT: "Bear Put Debit Spread",
    PUT_CREDIT: "Bull Put Credit Spread",
    CALL_CREDIT: "Bear Call Credit Spread",
    IRON_CONDOR: "Iron Condor",
    CASH_SECURED_PUT: "Cash-Secured Put",
    COVERED_CALL: "Covered Call",
}

# Which strategies are net sellers of premium (different management rules).
CREDIT_STRATEGIES = {PUT_CREDIT, CALL_CREDIT, IRON_CONDOR, CASH_SECURED_PUT, COVERED_CALL}
DEFINED_RISK = {CALL_DEBIT, PUT_DEBIT, PUT_CREDIT, CALL_CREDIT, IRON_CONDOR,
                LONG_CALL, LONG_PUT}


# --------------------------------------------------------------------------
@dataclass
class Leg:
    action: str          # 'buy' | 'sell'
    kind: str            # 'call' | 'put'
    strike: float
    expiry: date
    dte: int
    bid: float
    ask: float
    mid: float
    iv: float
    delta: float
    gamma: float
    theta: float
    vega: float
    open_interest: int
    volume: int
    contract: str = ""
    qty: int = 1

    @property
    def sign(self) -> int:
        """+1 if the leg is bought (cash out), -1 if sold (cash in)."""
        return 1 if self.action == "buy" else -1

    @property
    def spread_abs(self) -> float:
        return max(0.0, self.ask - self.bid)

    def fill_price(self, slip_frac: float) -> float:
        """
        Assumed fill. Buying, you pay above the mid; selling, you receive below
        it. slip_frac=0 is a perfect mid fill (optimistic), 1.0 means paying the
        full natural bid/ask (pessimistic). Default config uses 0.4.
        """
        return self.mid + self.sign * slip_frac * self.spread_abs / 2.0

    def intrinsic(self, S: float) -> float:
        return max(0.0, S - self.strike) if self.kind == "call" else max(0.0, self.strike - S)

    def describe(self) -> str:
        return (f"{'BUY ' if self.action == 'buy' else 'SELL'} {self.qty} "
                f"{self.expiry:%d-%b-%y} {self.strike:g} {self.kind.upper()}")

    def as_dict(self):
        return asdict(self)


@dataclass
class Trade:
    ticker: str
    strategy: str
    legs: list
    spot: float
    expiry: date
    dte: int

    # Pricing (per contract, in dollars, slippage- and commission-adjusted)
    net_price: float = 0.0        # >0 debit paid, <0 credit received (per share)
    net_cash: float = 0.0         # dollars per contract: -debit or +credit
    commission: float = 0.0
    max_profit: float | None = None   # None = theoretically unlimited
    max_loss: float = 0.0
    breakevens: list = field(default_factory=list)
    width: float = 0.0
    collateral: float = 0.0       # buying power actually tied up

    # Probability / value
    pop: float = 0.0              # probability of profit, measured at IMPLIED vol
    pop_at_rv: float = 0.0        # ... and at REALISED vol (see analyze())
    p_touch_short: float | None = None
    pop_source: str = "lognormal, zero drift"
    ev_at_iv: float = 0.0
    ev_at_rv: float = 0.0
    exp_win: float = 0.0          # E[profit | the trade wins], at realised vol
    exp_loss: float = 0.0         # E[loss   | the trade loses], at realised vol
    breakeven_win_rate: float = 0.0   # win rate needed to break even (see analyze)
    risk_reward: float = 0.0
    return_on_risk: float = 0.0
    credit_to_width: float | None = None

    # Net greeks per contract
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_theta: float = 0.0
    net_vega: float = 0.0

    # Quality
    liq_score: float = 0.0
    max_spread_pct: float = 0.0
    min_oi: int = 0
    notes: list = field(default_factory=list)

    @property
    def pretty(self) -> str:
        return PRETTY.get(self.strategy, self.strategy)

    @property
    def is_credit(self) -> bool:
        return self.strategy in CREDIT_STRATEGIES

    @property
    def short_strikes(self) -> list:
        return [l.strike for l in self.legs if l.action == "sell"]

    def payoff(self, S: float) -> float:
        """Position value at expiry per share, before entry cost."""
        return sum(l.sign * l.qty * l.intrinsic(S) for l in self.legs)

    def pnl(self, S: float) -> float:
        """P&L per CONTRACT in dollars at underlying price S at expiry."""
        return (self.payoff(S) - self.net_price) * MULT - self.commission

    def order_ticket(self, contracts: int = 1) -> str:
        """The order as you would enter it, at the suggested contract count."""
        return " / ".join(
            f"{'BUY ' if l.action == 'buy' else 'SELL'} {contracts * l.qty} "
            f"{l.expiry:%d-%b-%y} {l.strike:g} {l.kind.upper()}"
            for l in self.legs)

    def as_dict(self):
        d = asdict(self)
        d["legs"] = [l.as_dict() for l in self.legs]
        d["pretty"] = self.pretty
        return d


# --------------------------------------------------------------------------
# Generic analytics -- the single code path all strategies share
# --------------------------------------------------------------------------
def _profit_segments(t: Trade, s_max_mult: float = 4.0):
    """
    Partition the price axis at the strikes and return
    [(lo, hi, pnl_lo, pnl_hi), ...] covering 0 -> s_max.
    Within each segment P&L is exactly linear.
    """
    pts = sorted({0.0} | {l.strike for l in t.legs} | {t.spot * s_max_mult})
    return [(a, b, t.pnl(a), t.pnl(b)) for a, b in zip(pts[:-1], pts[1:])]


def analyze(t: Trade, atm_iv: float, rv: float, drift: float = 0.0) -> Trade:
    """
    Fill in every derived field on a Trade from its legs alone.

    max_profit / max_loss: exact. A piecewise-linear function attains its
    extremes at the corners or at the boundary, so evaluating the corners and
    inspecting the slope past the outermost one is sufficient -- and is how an
    unlimited-upside position is identified instead of assumed.
    """
    # --- entry price with slippage already applied by the caller via legs ---
    kinks = sorted({l.strike for l in t.legs})
    probe = [0.0] + kinks + [t.spot * 6.0]
    vals = [t.pnl(s) for s in probe]

    t.max_loss = max(0.0, -min(vals))
    # Slope beyond the highest strike tells us whether profit is capped above.
    far_slope = (t.pnl(t.spot * 6.0) - t.pnl(t.spot * 5.0)) / (t.spot * 1.0)
    unlimited_up = far_slope > 1e-6
    t.max_profit = None if unlimited_up else max(vals)
    if t.max_loss <= 0.0:
        # Can only happen if the quoted prices are internally inconsistent.
        t.max_loss = 0.01
        t.notes.append("Quoted prices imply no loss scenario -- almost certainly a "
                       "stale or crossed quote. Verify in your broker before trusting.")

    # --- breakevens: exact roots inside each linear segment ---
    bes = []
    for a, b, pa, pb in _profit_segments(t):
        if pa == 0.0 and a > 0:
            bes.append(a)
        if (pa < 0 < pb) or (pb < 0 < pa):
            bes.append(a + (b - a) * (-pa) / (pb - pa))
    t.breakevens = sorted(round(x, 2) for x in set(bes))

    # --- probability of profit, under TWO measures -----------------------
    #
    # This distinction is the single most important idea in the scanner.
    #
    # Measured at IMPLIED vol, a fairly-priced spread ALWAYS has
    # POP < required_pop. That is not a quirk, it is no-arbitrage: if the
    # market's own vol made a structure favourable, it would be free money.
    # So scoring premium selling at implied vol would reject every credit
    # spread ever written, which is obviously wrong -- people make money
    # selling premium.
    #
    # The reason they do is that implied vol usually EXCEEDS what the stock
    # goes on to realise (the variance risk premium). The real bet behind a
    # credit spread is "this will move less than options are pricing". So POP
    # is computed twice:
    #
    #   pop        at implied vol  -- the conservative, market-implied number.
    #                                 Reported as THE probability of profit.
    #   pop_at_rv  at realised vol -- the probability if the stock keeps moving
    #                                 the way it has been. This is what the
    #                                 edge is scored on.
    #
    # Both are shown in the report so the gap between them is visible rather
    # than buried, because that gap IS the trade's thesis.
    T = max(t.dte, 0) / 365.0
    sig = atm_iv if (atm_iv and atm_iv > 0) else 0.3

    def _pop_at(vol_used: float) -> float:
        acc = 0.0
        for a, b, pa, pb in _profit_segments(t):
            if pa <= 0 and pb <= 0:
                continue
            lo, hi = a, b
            if pa < 0 < pb:                  # profitable only above the root
                lo = a + (b - a) * (-pa) / (pb - pa)
            elif pb < 0 < pa:                # profitable only below the root
                hi = a + (b - a) * (-pa) / (pb - pa)
            p_hi = prob_above(t.spot, hi, T, vol_used, drift) if hi > 0 else 1.0
            p_lo = prob_above(t.spot, lo, T, vol_used, drift) if lo > 0 else 1.0
            acc += max(0.0, p_lo - p_hi)
        return round(min(max(acc, 0.0), 1.0), 4)

    t.pop = _pop_at(sig)
    t.pop_at_rv = _pop_at(rv) if (rv and rv > 0) else t.pop

    # --- probability of touching the nearest short strike before expiry ---
    if t.short_strikes:
        nearest = min(t.short_strikes, key=lambda k: abs(k - t.spot))
        t.p_touch_short = round(prob_touch(t.spot, nearest, T, sig, drift), 4)

    # --- expected value, two measures ---
    pay = lambda s: t.pnl(s)
    t.ev_at_iv = round(expected_payoff(pay, t.spot, T, sig, drift, nodes=kinks), 2)
    rv_used = rv if (rv and rv > 0) else sig
    t.ev_at_rv = round(expected_payoff(pay, t.spot, T, rv_used, drift, nodes=kinks), 2)

    # --- the win rate this trade actually needs to break even ---------------
    #
    # The obvious formula, max_loss / (max_loss + max_profit), is wrong for any
    # structure whose theoretical max profit is not realistically reachable.
    # A long put's "max profit" assumes the stock goes to zero, which would
    # make its break-even win rate ~5% and rank it above everything else on a
    # probability edge it does not have.
    #
    # So the real quantity is used instead: the CONDITIONAL expected win and
    # loss under the realised-vol measure.
    #
    #     break-even win rate = E[loss | lose] / (E[loss | lose] + E[win | win])
    #
    # This is well defined for every structure, bounded or not, and it is the
    # number a win rate must actually beat. The clipped payoffs gain corners at
    # the breakevens as well as the strikes, so both are passed as integration
    # nodes to keep the quadrature exact.
    nodes_all = sorted(set(kinks) | {b for b in t.breakevens if b > 0})
    e_win = expected_payoff(lambda s: max(0.0, t.pnl(s)), t.spot, T, rv_used,
                            drift, nodes=nodes_all)
    e_loss = expected_payoff(lambda s: max(0.0, -t.pnl(s)), t.spot, T, rv_used,
                             drift, nodes=nodes_all)
    p_win = max(t.pop_at_rv, 1e-6)
    p_lose = max(1.0 - t.pop_at_rv, 1e-6)
    t.exp_win = round(e_win / p_win, 2)
    t.exp_loss = round(e_loss / p_lose, 2)
    denom = t.exp_win + t.exp_loss
    t.breakeven_win_rate = round(t.exp_loss / denom, 4) if denom > 0 else 1.0

    # --- ratios ---
    if t.max_profit is None:
        t.risk_reward = float("inf")
        t.return_on_risk = float("inf")
    elif t.max_loss > 0:
        t.risk_reward = round(t.max_profit / t.max_loss, 3)
        t.return_on_risk = round(t.max_profit / t.max_loss * 100.0, 1)
    if t.width > 0 and t.net_cash > 0:
        t.credit_to_width = round(t.net_cash / (t.width * MULT), 3)

    # --- net greeks ---
    t.net_delta = round(sum(l.sign * l.qty * l.delta for l in t.legs), 4)
    t.net_gamma = round(sum(l.sign * l.qty * l.gamma for l in t.legs), 5)
    t.net_theta = round(sum(l.sign * l.qty * l.theta for l in t.legs) * MULT, 2)
    t.net_vega = round(sum(l.sign * l.qty * l.vega for l in t.legs) * MULT, 2)

    # --- liquidity roll-up: judged by the WORST leg, since one illiquid leg
    #     is enough to make the whole spread untradeable ---
    t.max_spread_pct = round(max((l.spread_abs / l.mid if l.mid > 0 else 1.0)
                                 for l in t.legs), 4)
    t.min_oi = int(min(l.open_interest for l in t.legs))
    return t


def _leg_from_row(row, action: str, expiry: date, dte: int) -> Leg:
    return Leg(
        action=action, kind=str(row["kind"]), strike=float(row["strike"]),
        expiry=expiry, dte=dte,
        bid=float(row["bid"]), ask=float(row["ask"]), mid=float(row["mid"]),
        iv=float(row["iv"]), delta=float(row["delta"]), gamma=float(row["gamma"]),
        theta=float(row["theta"]), vega=float(row["vega"]),
        open_interest=int(row["open_interest"]), volume=int(row["volume"]),
        contract=str(row.get("contract", "")),
    )


def _finalize(t: Trade, legs: list, slip: float, comm_per_leg: float,
              cv: ChainView, rv: float, collateral: float | None = None) -> Trade:
    t.legs = legs
    t.net_price = sum(l.sign * l.qty * l.fill_price(slip) for l in legs)
    t.net_cash = round(-t.net_price * MULT, 2)      # +credit received / -debit paid
    t.commission = round(len(legs) * comm_per_leg * 2.0, 2)   # round turn
    want = {(l.kind, round(l.strike, 4)) for l in legs}
    scores = [float(r.liq_score) for r in cv.df.itertuples(index=False)
              if (r.kind, round(float(r.strike), 4)) in want]
    t.liq_score = round(float(np.mean(scores)) if scores else 0.0, 1)
    t = analyze(t, cv.atm_iv(), rv)
    t.collateral = round(collateral if collateral is not None else t.max_loss, 2)
    return t


# --------------------------------------------------------------------------
# Strategy builders. Each returns a Trade or None if the chain cannot
# support a sane structure (which is a valid, common outcome).
# --------------------------------------------------------------------------
def build_long_option(cv: ChainView, kind: str, target_delta: float,
                      slip: float, comm: float, rv: float) -> Trade | None:
    row = cv.nearest_delta(kind, target_delta)
    if row is None:
        return None
    leg = _leg_from_row(row, "buy", cv.expiry, cv.dte)
    t = Trade(ticker=cv.ticker, strategy=LONG_CALL if kind == "call" else LONG_PUT,
              legs=[], spot=cv.spot, expiry=cv.expiry, dte=cv.dte)
    t = _finalize(t, [leg], slip, comm, cv, rv)
    t.notes.append("Unlimited theoretical upside; max loss is the premium paid. "
                   "Long premium means time decay works against you every day.")
    return t


def build_vertical(cv: ChainView, kind: str, long_delta: float, short_delta: float,
                   slip: float, comm: float, rv: float,
                   min_width_strikes: int = 1) -> Trade | None:
    """
    A two-leg vertical. Buying the higher-delta leg and selling the lower-delta
    one gives a DEBIT spread (directional, capped); reversing gives a CREDIT
    spread (premium selling). Which one it is falls out of the net price, so
    the strategy label is assigned from the realised structure rather than
    assumed up front.
    """
    long_row = cv.nearest_delta(kind, long_delta)
    short_row = cv.nearest_delta(kind, short_delta, exclude_strikes={float(long_row["strike"])} if long_row is not None else None)
    if long_row is None or short_row is None:
        return None
    if float(long_row["strike"]) == float(short_row["strike"]):
        return None

    step = cv.strike_step()
    if abs(float(long_row["strike"]) - float(short_row["strike"])) < step * min_width_strikes - 1e-9:
        return None

    legs = [_leg_from_row(long_row, "buy", cv.expiry, cv.dte),
            _leg_from_row(short_row, "sell", cv.expiry, cv.dte)]
    width = abs(legs[0].strike - legs[1].strike)

    net = sum(l.sign * l.fill_price(slip) for l in legs)
    if net > 0:                       # paid -> debit spread
        strat = CALL_DEBIT if kind == "call" else PUT_DEBIT
    else:                             # received -> credit spread
        strat = CALL_CREDIT if kind == "call" else PUT_CREDIT

    t = Trade(ticker=cv.ticker, strategy=strat, legs=[], spot=cv.spot,
              expiry=cv.expiry, dte=cv.dte, width=width)
    t = _finalize(t, legs, slip, comm, cv, rv)
    return t


def build_credit_spread(cv: ChainView, kind: str, short_delta: float, width: float,
                        slip: float, comm: float, rv: float) -> Trade | None:
    """
    Sell the `short_delta` option, buy protection `width` dollars further OTM.
    The protective leg is snapped to a strike that actually exists AND passes
    the liquidity filter, so the modelled width is the width you can really get.
    """
    short_row = cv.nearest_delta(kind, short_delta)
    if short_row is None:
        return None
    long_row = cv.at_or_beyond(kind, float(short_row["strike"]), width)
    if long_row is None:
        return None
    legs = [_leg_from_row(short_row, "sell", cv.expiry, cv.dte),
            _leg_from_row(long_row, "buy", cv.expiry, cv.dte)]
    actual_width = abs(legs[0].strike - legs[1].strike)
    if actual_width <= 0:
        return None
    strat = PUT_CREDIT if kind == "put" else CALL_CREDIT
    t = Trade(ticker=cv.ticker, strategy=strat, legs=[], spot=cv.spot,
              expiry=cv.expiry, dte=cv.dte, width=actual_width)
    t = _finalize(t, legs, slip, comm, cv, rv)
    if t.net_cash <= 0:
        return None                      # must be a net credit to be this trade
    t.notes.append(f"Defined risk: max loss ${t.max_loss:,.0f} per contract "
                   f"(${actual_width:g} width less credit).")
    return t


def build_iron_condor(cv: ChainView, short_delta: float, width: float,
                      slip: float, comm: float, rv: float) -> Trade | None:
    """Short put spread + short call spread. Profits from the stock going nowhere."""
    sp = cv.nearest_delta("put", short_delta)
    sc = cv.nearest_delta("call", short_delta)
    if sp is None or sc is None or float(sp["strike"]) >= float(sc["strike"]):
        return None
    lp = cv.at_or_beyond("put", float(sp["strike"]), width)
    lc = cv.at_or_beyond("call", float(sc["strike"]), width)
    if lp is None or lc is None:
        return None
    legs = [_leg_from_row(sp, "sell", cv.expiry, cv.dte),
            _leg_from_row(lp, "buy", cv.expiry, cv.dte),
            _leg_from_row(sc, "sell", cv.expiry, cv.dte),
            _leg_from_row(lc, "buy", cv.expiry, cv.dte)]
    put_w = abs(legs[0].strike - legs[1].strike)
    call_w = abs(legs[3].strike - legs[2].strike)
    t = Trade(ticker=cv.ticker, strategy=IRON_CONDOR, legs=[], spot=cv.spot,
              expiry=cv.expiry, dte=cv.dte, width=max(put_w, call_w))
    t = _finalize(t, legs, slip, comm, cv, rv)
    if t.net_cash <= 0:
        return None
    t.notes.append(f"Profit zone {t.breakevens[0]:g} to {t.breakevens[-1]:g} at expiry "
                   f"(spot {cv.spot:g}).")
    return t


def build_cash_secured_put(cv: ChainView, short_delta: float,
                           slip: float, comm: float, rv: float) -> Trade | None:
    """
    Naked short put, fully cash-secured. Max loss is large (strike to zero) so
    the collateral, not the max loss, is what should drive sizing -- which is
    why collateral is set explicitly here.
    """
    row = cv.nearest_delta("put", short_delta)
    if row is None:
        return None
    leg = _leg_from_row(row, "sell", cv.expiry, cv.dte)
    t = Trade(ticker=cv.ticker, strategy=CASH_SECURED_PUT, legs=[], spot=cv.spot,
              expiry=cv.expiry, dte=cv.dte)
    collateral = leg.strike * MULT - (leg.fill_price(slip) * MULT)
    t = _finalize(t, [leg], slip, comm, cv, rv, collateral=collateral)
    t.notes.append(f"Requires ${collateral:,.0f} collateral. If assigned you buy "
                   f"100 shares at {leg.strike:g} (effective cost "
                   f"{leg.strike - leg.fill_price(slip):.2f}).")
    return t


def build_covered_call(cv: ChainView, short_delta: float,
                       slip: float, comm: float, rv: float) -> Trade | None:
    """
    Short call against 100 long shares. The stock leg is included in the payoff
    so breakeven, max profit and POP are the real ones for the combined
    position, not the option's in isolation.
    """
    row = cv.nearest_delta("call", short_delta)
    if row is None:
        return None
    leg = _leg_from_row(row, "sell", cv.expiry, cv.dte)
    t = CoveredCallTrade(ticker=cv.ticker, strategy=COVERED_CALL, legs=[],
                         spot=cv.spot, expiry=cv.expiry, dte=cv.dte)
    collateral = cv.spot * MULT - leg.fill_price(slip) * MULT
    t = _finalize(t, [leg], slip, comm, cv, rv, collateral=collateral)
    t.notes.append(f"Requires 100 shares (~${cv.spot * MULT:,.0f}). Caps upside at "
                   f"{leg.strike:g}; keeps all downside below breakeven.")
    return t


class CoveredCallTrade(Trade):
    """Covered call: payoff includes the 100 long shares."""

    def payoff(self, S: float) -> float:
        return S + super().payoff(S)

    def pnl(self, S: float) -> float:
        # Entry: pay spot for shares, receive the call premium.
        basis = self.spot + self.net_price
        return (self.payoff(S) - basis) * MULT - self.commission
