"""
Trade scoring.

Every idea gets a 0-100 score built from six transparent components, and the
report shows all six plus the reasons and warnings behind them. Opaque scores
are worthless: if you cannot see why an idea ranked where it did, you cannot
tell a real edge from a bug.

The component worth the most attention is `probability`. Scoring raw
probability-of-profit is a classic trap -- it always favours selling far
out-of-the-money options, where POP is 95% and the one loss wipes out thirty
wins. What is scored here instead is PROBABILITY EDGE:

    required_pop = max_loss / (max_loss + max_profit)      <- break-even POP
    edge         = actual_pop - required_pop

`required_pop` is the win rate a structure needs just to break even given its
payoff asymmetry. A 95%-POP credit spread needing 94% to break even has almost
no edge. A 65%-POP spread needing 50% has a lot. That asymmetry is the whole
game, and this is the metric that captures it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from .regime import (TREND_DOWN, TREND_DOWN_STRONG, TREND_NEUTRAL, TREND_UP,
                     TREND_UP_STRONG, VOL_CHEAP, VOL_EXTREME, VOL_NORMAL, VOL_RICH)
from .strategies import (CALL_CREDIT, CALL_DEBIT, CASH_SECURED_PUT, COVERED_CALL,
                         CREDIT_STRATEGIES, IRON_CONDOR, LONG_CALL, LONG_PUT,
                         PUT_CREDIT, PUT_DEBIT, Trade)


@dataclass
class ScoreCard:
    total: float
    components: dict = field(default_factory=dict)
    penalties: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    required_pop: float = 0.0
    pop_edge: float = 0.0
    ev_per_risk: float = 0.0

    def as_dict(self):
        return asdict(self)


def _lin(x: float, lo: float, hi: float) -> float:
    """Map x from [lo, hi] onto 0-100, clamped. Handles inverted ranges."""
    if hi == lo:
        return 50.0
    v = (x - lo) / (hi - lo) * 100.0
    return float(min(max(v, 0.0), 100.0))


def required_pop(trade: Trade) -> float:
    """
    The win rate this structure needs to break even.

    Uses the conditional expected win and loss computed in strategies.analyze()
    rather than max profit / max loss, because theoretical maxima are not
    reachable for every structure (a long put's max profit assumes the stock
    goes to zero) and would badly distort the comparison between structures.
    """
    return float(trade.breakeven_win_rate) if trade.breakeven_win_rate else 1.0


# --------------------------------------------------------------------------
def _regime_fit(trade: Trade, regime, cfg) -> tuple[float, list, list]:
    """How well does this structure match what the market is actually doing?"""
    reasons, warns = [], []
    tr, vol = regime.trend, regime.vol
    bias = regime.directional_bias
    s = abs(tr.score)
    strength = tr.strength

    bullish = trade.strategy in (LONG_CALL, CALL_DEBIT, PUT_CREDIT, CASH_SECURED_PUT)
    bearish = trade.strategy in (LONG_PUT, PUT_DEBIT, CALL_CREDIT)
    neutral = trade.strategy in (IRON_CONDOR,)
    income = trade.strategy in (COVERED_CALL,)

    if neutral:
        # Condors want chop: low trend score, low ADX, price mid-range.
        fit = _lin(-abs(tr.score), -60.0, -5.0) * 0.55 + _lin(-tr.adx, -35.0, -12.0) * 0.45
        if abs(tr.score) < 15 and tr.adx < 20:
            reasons.append(f"Range-bound: trend score {tr.score:+.0f}, ADX {tr.adx:.0f} "
                           f"-- no directional pressure for a neutral position to fight")
        if abs(tr.score) > 35:
            warns.append(f"Trend score {tr.score:+.0f} is directional; a condor fights it")
    elif income:
        fit = _lin(tr.score, -40.0, 60.0)
        if tr.score > 10:
            reasons.append("Mild upward drift suits a covered call")
    else:
        aligned = (bias > 0 and bullish) or (bias < 0 and bearish)
        opposed = (bias > 0 and bearish) or (bias < 0 and bullish)
        if aligned:
            fit = _lin(s, cfg.regime.min_trend_score, 75.0) * 0.6 + _lin(strength, 30.0, 85.0) * 0.4
            reasons.append(
                f"{'Bullish' if bullish else 'Bearish'} structure aligned with a "
                f"{tr.label.replace('_', ' ')} (score {tr.score:+.0f}, strength "
                f"{strength:.0f}, ADX {tr.adx:.0f}, R² {tr.slope_r2:.2f})")
        elif opposed:
            fit = _lin(-s, -70.0, -10.0) * 0.5
            warns.append(f"Structure fights the prevailing {tr.label.replace('_', ' ')}")
        else:
            # Neutral tape: credit spreads still work, long premium does not.
            fit = 45.0 if trade.strategy in CREDIT_STRATEGIES else 25.0
            if trade.strategy in (LONG_CALL, LONG_PUT):
                warns.append("No clear trend -- a directional long needs movement to pay")

        if trade.strategy in (LONG_CALL, LONG_PUT, CALL_DEBIT, PUT_DEBIT):
            if tr.adx < cfg.regime.min_adx_directional:
                fit *= 0.75
                warns.append(f"ADX {tr.adx:.0f} below {cfg.regime.min_adx_directional:.0f} "
                             f"-- weak trend for a directional debit")
            if abs(tr.ext_atr) > cfg.regime.max_extension_atr:
                fit *= 0.8
                warns.append(f"Price {abs(tr.ext_atr):.1f} ATR from its 21-EMA -- "
                             f"chasing an extended move")
    return float(min(max(fit, 0.0), 100.0)), reasons, warns


def _premium_edge(trade: Trade, regime, cfg) -> tuple[float, list, list]:
    """
    Are you on the right side of volatility pricing?
    Sellers want rich IV; buyers want cheap IV. Getting this backwards is the
    most common and most expensive mistake in retail options.
    """
    reasons, warns = [], []
    vol = regime.vol
    rank = vol.iv_rank if vol.iv_rank is not None else 50.0
    ratio = vol.iv_rv_ratio if vol.iv_rv_ratio == vol.iv_rv_ratio else 1.15
    selling = trade.strategy in CREDIT_STRATEGIES

    if selling:
        by_rank = _lin(rank, 30.0, 85.0)
        by_ratio = _lin(ratio, 1.00, 1.55)
        score = by_rank * 0.5 + by_ratio * 0.5
        if rank >= cfg.regime.iv_rich:
            reasons.append(f"IV rank {rank:.0f} ({vol.iv_rank_source}) -- premium is rich, "
                           f"favourable for a seller")
        if ratio >= 1.30:
            reasons.append(f"IV {ratio:.2f}x realised vol -- you are being paid more than "
                           f"the stock has actually been moving")
        if rank < 30 or ratio < 1.0:
            warns.append(f"IV rank {rank:.0f} / IV-RV {ratio:.2f} -- selling cheap premium; "
                         f"little cushion if the stock moves")
    else:
        by_rank = _lin(-rank, -75.0, -15.0)
        by_ratio = _lin(-ratio, -1.45, -0.95)
        score = by_rank * 0.5 + by_ratio * 0.5
        if rank <= cfg.regime.iv_cheap:
            reasons.append(f"IV rank {rank:.0f} ({vol.iv_rank_source}) -- premium is cheap, "
                           f"favourable for a buyer")
        if ratio <= 1.05:
            reasons.append(f"IV only {ratio:.2f}x realised vol -- not overpaying for movement")
        if rank > 65 or ratio > 1.4:
            warns.append(f"IV rank {rank:.0f} / IV-RV {ratio:.2f} -- buying expensive premium; "
                         f"you need a big move just to overcome the vol you paid for")
    return float(score), reasons, warns


def capital_at_risk(trade: Trade) -> float:
    """
    The denominator to judge returns against.

    For defined-risk structures that is the max loss. For a cash-secured put or
    covered call, max loss is the stock going to zero -- a number that would
    make every such trade look like a rounding error. Those are judged against
    the capital actually committed, which is what the position really ties up.
    """
    if trade.strategy in (CASH_SECURED_PUT, COVERED_CALL):
        return max(trade.collateral, 1.0)
    return max(min(trade.max_loss, trade.collateral or trade.max_loss), 1.0)


def _ev_component(trade: Trade) -> tuple[float, float, list, list]:
    """
    EV under realised vol, per dollar of capital at risk, ANNUALISED.

    Annualising matters: a 2% return over 30 days and a 2% return over 60 days
    are not the same trade, and without this the scanner would systematically
    prefer longer-dated positions for no good reason.
    """
    reasons, warns = [], []
    risk = capital_at_risk(trade)
    per_risk = trade.ev_at_rv / risk
    ann = per_risk * (365.0 / max(trade.dte, 1))
    score = _lin(ann, -0.20, 1.20)
    if ann > 0.25:
        reasons.append(f"Expected value +${trade.ev_at_rv:,.0f}/contract on "
                       f"${risk:,.0f} at risk ({ann:+.0%}/yr annualised) if the stock "
                       f"keeps moving the way it has been")
    if trade.ev_at_rv < 0:
        warns.append(f"Negative expected value (${trade.ev_at_rv:,.0f}/contract) even at "
                     f"realised vol -- you are paying more for movement than the stock "
                     f"has been delivering")
    return float(score), float(ann), reasons, warns


def _probability_component(trade: Trade) -> tuple[float, float, float, list, list]:
    """
    Probability EDGE: the win rate the structure gets, against the win rate its
    payoff asymmetry requires just to break even.

    Scored on the REALISED-vol probability, not the implied-vol one. At implied
    vol a fairly-priced spread necessarily has POP below its required POP (no
    free lunch), so scoring there would zero out every premium-selling trade.
    The edge being measured is explicitly "implied vol overstates what this
    stock actually does", which is only visible under the realised-vol measure.
    Both numbers are reported.
    """
    reasons, warns = [], []
    req = required_pop(trade)
    edge = trade.pop_at_rv - req
    score = _lin(edge, -0.08, 0.18)
    if edge > 0.05:
        reasons.append(f"Needs a {req:.0%} win rate to break even (risking ~"
                       f"${trade.exp_loss:,.0f} to make ~${trade.exp_win:,.0f} on average); "
                       f"gets {trade.pop_at_rv:.0%} at the stock's realised vol -- a "
                       f"{edge:+.0%} edge. At implied vol it is {trade.pop:.0%}, and that "
                       f"gap is the volatility premium you are being paid")
    elif edge < 0:
        warns.append(f"Even at realised vol the {trade.pop_at_rv:.0%} win rate is below the "
                     f"{req:.0%} this payoff needs -- the odds do not cover the "
                     f"risk-reward")
    if trade.p_touch_short is not None and trade.p_touch_short > 0.55:
        warns.append(f"{trade.p_touch_short:.0%} chance of touching the short strike before "
                     f"expiry -- expect it to look ugly even if it finishes a winner")
    return float(score), float(req), float(edge), reasons, warns


def _structure_component(trade: Trade) -> tuple[float, list, list]:
    """Is the structure itself well built -- enough credit, sane geometry?"""
    reasons, warns = [], []
    if trade.strategy in (PUT_CREDIT, CALL_CREDIT, IRON_CONDOR):
        ctw = trade.credit_to_width or 0.0
        score = _lin(ctw, 0.12, 0.38)
        if ctw >= 0.28:
            reasons.append(f"Collecting {ctw:.0%} of the spread width -- strong premium "
                           f"for the risk taken")
        if ctw < 0.15:
            warns.append(f"Only {ctw:.0%} of width collected -- risking "
                         f"${trade.max_loss:,.0f} to make ${trade.max_profit:,.0f}")
    elif trade.strategy in (CASH_SECURED_PUT, COVERED_CALL):
        # Judged on annualised yield on committed capital -- the only sensible
        # measure for a position whose nominal max loss is the stock to zero.
        cap = max(trade.collateral, 1.0)
        yld = (trade.max_profit or 0.0) / cap * (365.0 / max(trade.dte, 1))
        score = _lin(yld, 0.04, 0.42)
        if yld >= 0.18:
            reasons.append(f"{yld:.0%}/yr annualised on ${cap:,.0f} of committed capital "
                           f"if it expires worthless")
        if yld < 0.08:
            warns.append(f"Only {yld:.0%}/yr on ${cap:,.0f} tied up -- thin "
                         f"compensation for holding assignment risk")
    elif trade.max_profit is None:
        score = 55.0                     # long options: EV and premium edge carry it
    else:
        rr = trade.max_profit / max(trade.max_loss, 1.0)
        score = _lin(rr, 0.5, 2.2)
        # A big payoff ratio means nothing on its own. A 13:1 spread that wins
        # 5% of the time is a raffle ticket, and scoring the ratio alone would
        # rank it above a sound 1.5:1 that wins 60% -- so the ratio is damped by
        # how often it actually pays.
        if trade.pop_at_rv < 0.35:
            score *= _lin(trade.pop_at_rv, 0.05, 0.35) / 100.0
            warns.append(f"Risk-reward looks attractive at {rr:.1f}:1, but it only wins "
                         f"{trade.pop_at_rv:.0%} of the time -- the payoff ratio is high "
                         f"because the odds are poor, not because it is good value")
        elif rr >= 1.2:
            reasons.append(f"Risk-reward {rr:.2f}:1 -- making ${trade.max_profit:,.0f} "
                           f"against ${trade.max_loss:,.0f} risked")
    return float(score), reasons, warns


# --------------------------------------------------------------------------
def score_trade(trade: Trade, regime, cfg) -> ScoreCard:
    reasons: list[str] = []
    warns: list[str] = []

    fit, r1, w1 = _regime_fit(trade, regime, cfg)
    prem, r2, w2 = _premium_edge(trade, regime, cfg)
    ev, ev_per_risk, r3, w3 = _ev_component(trade)
    prob, req, edge, r4, w4 = _probability_component(trade)
    struct, r5, w5 = _structure_component(trade)
    liq = float(trade.liq_score)

    for lst, src in ((reasons, r1 + r2 + r3 + r4 + r5), (warns, w1 + w2 + w3 + w4 + w5)):
        lst.extend(src)

    comps = {
        "regime_fit": round(fit, 1),
        "premium_edge": round(prem, 1),
        "expected_value": round(ev, 1),
        "probability": round(prob, 1),
        "liquidity": round(liq, 1),
        "structure": round(struct, 1),
    }
    W = cfg.scoring
    wmap = {"regime_fit": W.regime_fit, "premium_edge": W.premium_edge,
            "expected_value": W.expected_value, "probability": W.probability,
            "liquidity": W.liquidity, "structure": W.structure}
    wsum = sum(wmap.values()) or 1.0
    base = sum(comps[k] * wmap[k] for k in comps) / wsum

    # ---------------------------- penalties ----------------------------
    pen: dict[str, float] = {}
    ed = regime.earnings_in_days
    if ed is not None and 0 <= ed <= trade.dte:
        pen["earnings_inside_expiry"] = cfg.earnings.score_penalty
        warns.append(f"Earnings in {ed} days, before this {trade.dte}-DTE expiry -- "
                     f"a gap through your strikes is the main risk here")
    elif ed is not None and 0 <= ed <= trade.dte + cfg.earnings.warn_window_days:
        pen["earnings_near"] = 4.0

    if trade.max_spread_pct > 0.07:
        pen["wide_spread"] = min(14.0, (trade.max_spread_pct - 0.07) * 220.0)
        warns.append(f"Worst leg has a {trade.max_spread_pct:.1%} bid/ask spread -- "
                     f"slippage will take a real bite")
    if trade.min_oi < cfg.liquidity.min_open_interest * 1.5:
        pen["thin_oi"] = 4.0
    if regime.vol.iv_rank_source == "rv_proxy":
        pen["iv_rank_estimated"] = 3.0
    if regime.avg_dollar_volume and regime.avg_dollar_volume < cfg.filters.min_avg_dollar_volume:
        pen["thin_underlying"] = 6.0
    if trade.pop_at_rv < 0.30 and trade.max_profit is not None:
        pen["low_pop"] = 18.0
    if trade.pop_at_rv - trade.pop > 0.18:
        # The thesis leans heavily on realised vol staying below implied. True
        # on average, but it is the assumption doing the work -- flag it.
        pen["vol_assumption_heavy"] = 4.0
        warns.append(f"Edge depends substantially on realised vol ({trade.pop_at_rv:.0%} POP) "
                     f"staying below implied ({trade.pop:.0%} POP). If volatility expands, "
                     f"this turns against you")

    total = float(min(max(base - sum(pen.values()), 0.0), 100.0))
    return ScoreCard(total=round(total, 1), components=comps,
                     penalties={k: round(v, 1) for k, v in pen.items()},
                     reasons=reasons, warnings=warns,
                     required_pop=round(req, 4), pop_edge=round(edge, 4),
                     ev_per_risk=round(ev_per_risk, 4))
