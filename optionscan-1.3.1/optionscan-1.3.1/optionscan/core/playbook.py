"""
The playbook: which structures to even consider, given the regime.

This is the piece that makes the scanner coherent rather than a list of
everything that happens to be cheap. The logic is a 2x2 that professional
desks run on, and it is worth stating plainly because it is the opposite of
what most retail scanners do:

                    |  IV CHEAP / NORMAL        |  IV RICH / EXTREME
    ----------------+---------------------------+--------------------------
    UPTREND         |  BUY premium:             |  SELL premium:
                    |  long call, call debit    |  put credit spread, CSP
    ----------------+---------------------------+--------------------------
    DOWNTREND       |  BUY premium:             |  SELL premium:
                    |  long put, put debit      |  call credit spread
    ----------------+---------------------------+--------------------------
    NEUTRAL         |  NOTHING -- no edge.      |  SELL premium both sides:
                    |  Sit out.                 |  iron condor
    ----------------+---------------------------+--------------------------

Two things follow from it that matter more than any individual setup:

1. Direction alone never decides the structure. Being right about direction and
   wrong about volatility still loses money -- a 60%-IV call on a stock
   realising 30% needs a huge move just to break even on the vol you overpaid.

2. "No edge" is a real, frequent, correct answer. A neutral tape with cheap
   premium offers nothing worth trading, and the scanner says so instead of
   manufacturing an idea. Some days the honest output is an empty list.
"""
from __future__ import annotations

from .regime import (TREND_DOWN, TREND_DOWN_STRONG, TREND_NEUTRAL, TREND_UP,
                     TREND_UP_STRONG, VOL_CHEAP, VOL_EXTREME, VOL_NORMAL, VOL_RICH)
from .strategies import (CALL_CREDIT, CALL_DEBIT, CASH_SECURED_PUT, COVERED_CALL,
                         IRON_CONDOR, LONG_CALL, LONG_PUT, PUT_CREDIT, PUT_DEBIT)


def candidates(regime, cfg) -> list[str]:
    """Ordered list of strategy ids worth building for this underlying today."""
    tr, vol = regime.trend, regime.vol
    rank = vol.iv_rank if vol.iv_rank is not None else 50.0
    rich = rank >= cfg.regime.iv_rich or vol.label in (VOL_RICH, VOL_EXTREME)
    cheap = rank <= cfg.regime.iv_cheap or vol.label == VOL_CHEAP

    up = tr.label in (TREND_UP, TREND_UP_STRONG)
    down = tr.label in (TREND_DOWN, TREND_DOWN_STRONG)
    strong = tr.label in (TREND_UP_STRONG, TREND_DOWN_STRONG)
    tradeable_trend = (tr.strength >= cfg.regime.min_trend_strength_directional
                       and tr.adx >= cfg.regime.min_adx_directional
                       and abs(tr.ext_atr) <= cfg.regime.max_extension_atr)

    out: list[str] = []

    if up:
        if rich:
            out += [PUT_CREDIT, CASH_SECURED_PUT]
        if cheap or not rich:
            if tradeable_trend:
                out += [CALL_DEBIT]
                if strong and cheap:
                    # A single long call only earns its theta cost when the
                    # trend is strong AND the premium is genuinely cheap.
                    out += [LONG_CALL]
            else:
                out += [CALL_DEBIT]
    elif down:
        if rich:
            out += [CALL_CREDIT]
        if cheap or not rich:
            if tradeable_trend:
                out += [PUT_DEBIT]
                if strong and cheap:
                    out += [LONG_PUT]
            else:
                out += [PUT_DEBIT]
    else:                                   # neutral
        if rich:
            out += [IRON_CONDOR]
            if tr.score > 5:
                out += [PUT_CREDIT]
            elif tr.score < -5:
                out += [CALL_CREDIT]
        # Cheap premium + no trend = no edge. Deliberately nothing.

    # Covered calls are an income overlay on shares you already own, so they
    # are only offered where the tape supports them, never as a directional bet.
    if rich and (up or tr.label == TREND_NEUTRAL) and COVERED_CALL in cfg.strategies_enabled:
        out += [COVERED_CALL]

    seen, ordered = set(), []
    for s in out:
        if s in cfg.strategies_enabled and s not in seen:
            seen.add(s)
            ordered.append(s)
    return ordered


def no_edge_reason(regime, cfg) -> str:
    """Explain an empty candidate list -- 'nothing today' should be legible."""
    tr, vol = regime.trend, regime.vol
    rank = vol.iv_rank if vol.iv_rank is not None else 50.0
    if tr.label == TREND_NEUTRAL and rank < cfg.regime.iv_rich:
        return (f"No trend (score {tr.score:+.0f}, ADX {tr.adx:.0f}) and premium is not "
                f"rich (IV rank {rank:.0f}) -- nothing to buy, nothing worth selling")
    if abs(tr.ext_atr) > cfg.regime.max_extension_atr:
        return (f"Price is {abs(tr.ext_atr):.1f} ATR from its 21-EMA -- too extended to "
                f"enter directionally at a sane price")
    if tr.adx < cfg.regime.min_adx_directional and rank < cfg.regime.iv_rich:
        return (f"ADX {tr.adx:.0f} means chop, and IV rank {rank:.0f} means premium is "
                f"not rich enough to get paid for selling into it")
    return "No structure met the configured thresholds"
