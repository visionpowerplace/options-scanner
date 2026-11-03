"""
Regime classification -- the decision layer that chooses WHAT KIND of trade to
look for before any contract is considered.

Two independent axes:

  TREND  -- direction and, separately, strength. Direction alone is not enough:
            a +40% move that happened in three gaps is not a trend you can buy
            a 45-day call on. So strength blends ADX, EMA alignment, the
            R-squared of the log-price regression, and momentum confirmation.

  VOL    -- is option premium rich or cheap RIGHT NOW relative to how much the
            stock is actually moving. This is the axis most retail scanners
            ignore, and it decides whether you should be a buyer or a seller of
            premium. Buying 60% IV on a stock realising 30% is a losing game
            even when you get the direction right.

The product of the two axes picks the strategy family (see strategies.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from . import indicators as ind

# ------------------------------------------------------------------ labels
TREND_UP_STRONG = "strong_uptrend"
TREND_UP = "uptrend"
TREND_NEUTRAL = "neutral"
TREND_DOWN = "downtrend"
TREND_DOWN_STRONG = "strong_downtrend"

VOL_CHEAP = "cheap"
VOL_NORMAL = "normal"
VOL_RICH = "rich"
VOL_EXTREME = "extreme"


@dataclass
class TrendSignal:
    label: str
    score: float                 # -100 (max bearish) .. +100 (max bullish)
    strength: float              # 0..100, direction-agnostic conviction
    adx: float = float("nan")
    rsi: float = float("nan")
    macd_hist: float = float("nan")
    ema_stack: str = ""
    slope_ann: float = float("nan")
    slope_r2: float = float("nan")
    atr_pct: float = float("nan")
    ext_atr: float = float("nan")    # distance from EMA20 in ATR units
    range_pos: float = float("nan")
    rel_strength: float = 0.0
    components: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


@dataclass
class VolSignal:
    atm_iv: float
    rv20: float
    rv60: float
    parkinson20: float
    iv_rv_ratio: float
    iv_rank: float | None
    iv_rank_source: str          # 'history' (real) or 'rv_proxy' (estimated)
    iv_rank_obs: int
    iv_percentile: float | None
    label: str
    term_slope: float            # front IV / back IV; >1 = front-loaded event vol
    notes: list = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


@dataclass
class Regime:
    ticker: str
    spot: float
    trend: TrendSignal
    vol: VolSignal
    earnings_in_days: int | None = None
    avg_dollar_volume: float = 0.0
    gap_risk: float = 0.0
    atr: float = 0.0
    notes: list = field(default_factory=list)

    @property
    def directional_bias(self) -> int:
        if self.trend.label in (TREND_UP, TREND_UP_STRONG):
            return 1
        if self.trend.label in (TREND_DOWN, TREND_DOWN_STRONG):
            return -1
        return 0


# --------------------------------------------------------------------------
def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _last(s: pd.Series, default=float("nan")):
    if s is None or len(s) == 0:
        return default
    v = s.iloc[-1]
    return default if pd.isna(v) else float(v)


def classify_trend(df: pd.DataFrame, bench: pd.Series | None = None,
                   cfg=None) -> TrendSignal:
    """
    Blend eight independent reads of trend into one signed score.

    Each component is normalised to -1..+1 and weighted. Using several weak but
    uncorrelated signals beats trusting one strong one: any single indicator
    whipsaws, but it is unusual for ADX, the EMA stack, the regression slope and
    momentum to all agree at a turning point.
    """
    notes: list[str] = []
    close = df["close"]
    if len(close) < 60:
        return TrendSignal(TREND_NEUTRAL, 0.0, 0.0, notes=["insufficient history"])

    e8, e21 = ind.ema(close, 8), ind.ema(close, 21)
    e50 = ind.ema(close, 50)
    e200 = ind.ema(close, 200) if len(close) >= 200 else ind.ema(close, max(50, len(close) // 2))
    px = float(close.iloc[-1])
    v8, v21, v50, v200 = _last(e8), _last(e21), _last(e50), _last(e200)

    adx_v = _last(ind.adx(df, 14), 0.0)
    rsi_v = _last(ind.rsi(close, 14), 50.0)
    _, _, hist = ind.macd(close)
    hist_v = _last(hist, 0.0)
    hist_prev = float(hist.iloc[-4]) if len(hist) > 4 and not pd.isna(hist.iloc[-4]) else hist_v
    slope_s, r2_s = ind.log_slope(close, 40)
    slope_v, r2_v = _last(slope_s, 0.0), _last(r2_s, 0.0)
    atr_v = _last(ind.atr(df, 14), 0.0)
    atr_pct = atr_v / px if px else 0.0
    ext = (px - v21) / atr_v if atr_v > 0 and not np.isnan(v21) else 0.0
    rpos = _last(ind.range_position(df, 60), 0.5)
    rel = ind.relative_strength(close, bench, 63) if bench is not None else 0.0

    # --- components, each in -1..+1 -------------------------------------
    c = {}

    # 1. EMA stack: how many of the four alignment conditions hold.
    conds = [px > v21, v8 > v21, v21 > v50, v50 > v200]
    valid = [x for x in conds if not (isinstance(x, float) and np.isnan(x))]
    up_n = sum(1 for x in valid if x)
    c["ema_stack"] = (up_n / len(valid)) * 2.0 - 1.0 if valid else 0.0
    stack_txt = "".join("+" if x else "-" for x in conds)

    # 2. Regression slope, scaled by how orderly the move is (R-squared).
    #    A 50%/yr slope with R2 0.2 is noise; the same slope at R2 0.8 is a trend.
    c["slope"] = _clip(slope_v / 0.60, -1.0, 1.0) * _clip(r2_v / 0.55, 0.0, 1.0)

    # 3. MACD histogram level and whether it is expanding.
    scale = max(atr_v * 0.55, 1e-9)
    c["macd"] = _clip(hist_v / scale, -1.0, 1.0) * 0.7 + \
                _clip((hist_v - hist_prev) / scale, -1.0, 1.0) * 0.3

    # 4. RSI distance from 50, flattened at the extremes.
    c["rsi"] = _clip((rsi_v - 50.0) / 22.0, -1.0, 1.0)

    # 5. Position within the 60-day range.
    c["range_pos"] = _clip((rpos - 0.5) * 2.4, -1.0, 1.0) if not np.isnan(rpos) else 0.0

    # 6. Relative strength vs the benchmark.
    c["rel_strength"] = _clip(rel / 0.12, -1.0, 1.0)

    # 7. Price vs the long-term average -- the regime filter.
    c["long_term"] = _clip((px / v200 - 1.0) / 0.12, -1.0, 1.0) if v200 and not np.isnan(v200) else 0.0

    # 8. Short-term thrust: 10-day return in ATR units.
    if len(close) > 11 and atr_v > 0:
        c["thrust"] = _clip((px - float(close.iloc[-11])) / (atr_v * 3.2), -1.0, 1.0)
    else:
        c["thrust"] = 0.0

    weights = {"ema_stack": 0.20, "slope": 0.20, "macd": 0.12, "rsi": 0.10,
               "range_pos": 0.10, "rel_strength": 0.08, "long_term": 0.12,
               "thrust": 0.08}
    raw = sum(c[k] * w for k, w in weights.items())
    score = _clip(raw * 100.0, -100.0, 100.0)

    # Strength: conviction, independent of direction. ADX and R-squared carry
    # most of it because they measure orderliness rather than displacement.
    strength = _clip(
        abs(score) * 0.45
        + _clip((adx_v - 12.0) / 0.28, 0.0, 100.0) * 0.35
        + _clip(r2_v * 100.0, 0.0, 100.0) * 0.20,
        0.0, 100.0)

    if adx_v < 18:
        notes.append(f"ADX {adx_v:.0f} -- choppy, trend-following edge is weak")
    if abs(ext) > 2.5:
        notes.append(f"{abs(ext):.1f} ATR from 21-EMA -- extended, poor entry location")
    if rsi_v > 75:
        notes.append(f"RSI {rsi_v:.0f} -- overbought")
    if rsi_v < 25:
        notes.append(f"RSI {rsi_v:.0f} -- oversold")
    if r2_v < 0.25 and abs(score) > 35:
        notes.append("directional move is disorderly (low R-squared) -- treat as range")

    # Label. Requires BOTH displacement and strength to call a trend, which is
    # what stops the scanner recommending debit spreads in a chop.
    if score >= 45 and strength >= 55:
        label = TREND_UP_STRONG
    elif score >= 18:
        label = TREND_UP
    elif score <= -45 and strength >= 55:
        label = TREND_DOWN_STRONG
    elif score <= -18:
        label = TREND_DOWN
    else:
        label = TREND_NEUTRAL

    return TrendSignal(
        label=label, score=round(score, 1), strength=round(strength, 1),
        adx=round(adx_v, 1), rsi=round(rsi_v, 1), macd_hist=round(hist_v, 4),
        ema_stack=stack_txt, slope_ann=round(slope_v, 4), slope_r2=round(r2_v, 3),
        atr_pct=round(atr_pct, 4), ext_atr=round(ext, 2),
        range_pos=round(rpos, 3) if not np.isnan(rpos) else float("nan"),
        rel_strength=round(rel, 4),
        components={k: round(v, 3) for k, v in c.items()}, notes=notes)


def classify_vol(df: pd.DataFrame, atm_iv: float, iv_store=None,
                 ticker: str = "", front_iv: float | None = None,
                 back_iv: float | None = None, cfg=None) -> VolSignal:
    """
    Decide whether option premium is rich or cheap.

    IV rank is the right measure (where today's IV sits in its own 1-year
    range) but it needs IV HISTORY, which free data feeds do not publish. So:

      * If the local IV history store holds enough observations, use the real
        IV rank and label the source 'history'.
      * Otherwise fall back to an IV/RV-based estimate labelled 'rv_proxy', and
        say so in the report. An estimate you can see is worth far more than a
        confident number you cannot audit.

    The IV/RV ratio works from day one and is arguably the more direct measure
    anyway: it compares what options cost against how much the stock is
    actually moving. Equity options typically carry a 10-25% premium to
    realised vol (the variance risk premium), so ~1.15 is 'fair', not cheap.
    """
    notes: list[str] = []
    close = df["close"]
    rv20 = _last(ind.realized_vol(close, 20), float("nan"))
    rv60 = _last(ind.realized_vol(close, 60), float("nan"))
    pk20 = _last(ind.parkinson_vol(df, 20), float("nan"))

    # Blended realised-vol baseline: recent close-to-close, the high-low
    # estimator (more efficient, faster to react) and a longer anchor.
    parts = [(rv20, 0.45), (pk20, 0.30), (rv60, 0.25)]
    usable = [(v, w) for v, w in parts if v and not np.isnan(v) and v > 0]
    rv_blend = sum(v * w for v, w in usable) / sum(w for _, w in usable) if usable else float("nan")

    ratio = (atm_iv / rv_blend) if (atm_iv and rv_blend and rv_blend > 0) else float("nan")

    iv_rank, source, obs, iv_pct = None, "unavailable", 0, None
    if iv_store is not None and ticker and atm_iv:
        iv_rank, obs = iv_store.iv_rank(ticker, atm_iv)
        iv_pct, _ = iv_store.iv_percentile(ticker, atm_iv)
        if iv_rank is not None:
            source = "history"
    if iv_rank is None:
        # Proxy. Map the IV/RV ratio onto a 0-100 scale calibrated so that the
        # typical 1.15 variance risk premium lands at ~50, then nudge by where
        # realised vol itself sits in its own one-year range (high RV with high
        # IV is less of an opportunity than high IV with calm RV).
        if ratio and not np.isnan(ratio):
            base = _clip((ratio - 0.85) / (1.60 - 0.85) * 100.0, 0.0, 100.0)
            rv_pct = _last(ind.percent_rank(ind.realized_vol(close, 20), 252), 50.0)
            iv_rank = _clip(base * 0.75 + (100.0 - rv_pct) * 0.25, 0.0, 100.0)
            source = "rv_proxy"
            obs = int(len(close))
            notes.append("IV rank is ESTIMATED from IV/RV (no IV history yet); "
                         "it becomes a true IV rank after ~40 daily runs")
        else:
            iv_rank = 50.0
            notes.append("IV rank unavailable -- defaulting to neutral 50")

    term = (front_iv / back_iv) if (front_iv and back_iv and back_iv > 0) else float("nan")
    if term and not np.isnan(term) and term > 1.12:
        notes.append(f"front-month IV {term:.2f}x back month -- event premium in the front")

    if iv_rank >= 70 or (ratio and ratio >= 1.45):
        label = VOL_EXTREME if (iv_rank >= 85 or (ratio and ratio >= 1.8)) else VOL_RICH
    elif iv_rank <= 25 or (ratio and ratio <= 0.95):
        label = VOL_CHEAP
    else:
        label = VOL_NORMAL

    if ratio and not np.isnan(ratio):
        if ratio < 0.95:
            notes.append(f"IV/RV {ratio:.2f} -- options cheap vs actual movement; favour buying premium")
        elif ratio > 1.35:
            notes.append(f"IV/RV {ratio:.2f} -- options expensive vs actual movement; favour selling premium")

    return VolSignal(
        atm_iv=round(atm_iv, 4) if atm_iv else float("nan"),
        rv20=round(rv20, 4) if not np.isnan(rv20) else float("nan"),
        rv60=round(rv60, 4) if not np.isnan(rv60) else float("nan"),
        parkinson20=round(pk20, 4) if not np.isnan(pk20) else float("nan"),
        iv_rv_ratio=round(ratio, 3) if ratio and not np.isnan(ratio) else float("nan"),
        iv_rank=round(iv_rank, 1) if iv_rank is not None else None,
        iv_rank_source=source, iv_rank_obs=obs,
        iv_percentile=round(iv_pct, 1) if iv_pct is not None else None,
        label=label,
        term_slope=round(term, 3) if term and not np.isnan(term) else float("nan"),
        notes=notes)


# --------------------------------------------------------------------------
# Vectorised trend score -- one value per bar, for backtesting
# --------------------------------------------------------------------------
def trend_score_series(df: pd.DataFrame, bench: pd.Series | None = None) -> pd.DataFrame:
    """
    The same composite trend score as classify_trend(), computed for EVERY bar
    at once.

    Why this exists rather than calling classify_trend in a loop: the loop is
    O(n^2) in the history length and makes a real backtest impractically slow.
    More importantly, every input here is a causal rolling statistic, so the
    value at bar t uses only data up to and including t -- no look-ahead. That
    property is what makes the backtest trustworthy, and it is verified against
    classify_trend's own output in tests/test_backtest.py, which asserts the
    two agree at the final bar to within rounding.

    Returns a frame indexed like `df` with columns: score, strength, label.
    """
    close = df["close"]
    n = len(close)
    out = pd.DataFrame(index=df.index, columns=["score", "strength", "label"])
    if n < 60:
        return out

    e8, e21, e50 = ind.ema(close, 8), ind.ema(close, 21), ind.ema(close, 50)
    e200 = ind.ema(close, 200) if n >= 200 else ind.ema(close, max(50, n // 2))
    adx_s = ind.adx(df, 14)
    rsi_s = ind.rsi(close, 14)
    _, _, hist = ind.macd(close)
    slope_s, r2_s = ind.log_slope(close, 40)
    atr_s = ind.atr(df, 14)
    rpos_s = ind.range_position(df, 60)

    # 1. EMA stack -- fraction of the four alignment conditions that hold.
    conds = pd.concat([(close > e21), (e8 > e21), (e21 > e50), (e50 > e200)], axis=1)
    c_stack = conds.sum(axis=1) / 4.0 * 2.0 - 1.0

    # 2. Regression slope, scaled by orderliness.
    c_slope = (slope_s / 0.60).clip(-1, 1) * (r2_s / 0.55).clip(0, 1)

    # 3. MACD level plus expansion.
    scale = (atr_s * 0.55).replace(0, np.nan)
    c_macd = ((hist / scale).clip(-1, 1) * 0.7
              + ((hist - hist.shift(3)) / scale).clip(-1, 1) * 0.3)

    # 4-8.
    c_rsi = ((rsi_s - 50.0) / 22.0).clip(-1, 1)
    c_rpos = ((rpos_s - 0.5) * 2.4).clip(-1, 1)
    if bench is not None and len(bench):
        b = bench.reindex(close.index).ffill()
        rel = (close / close.shift(63) - 1.0) - (b / b.shift(63) - 1.0)
        c_rel = (rel / 0.12).clip(-1, 1)
    else:
        c_rel = pd.Series(0.0, index=close.index)
    c_long = ((close / e200 - 1.0) / 0.12).clip(-1, 1)
    c_thrust = ((close - close.shift(10)) / (atr_s * 3.2)).clip(-1, 1)

    W = {"ema_stack": 0.20, "slope": 0.20, "macd": 0.12, "rsi": 0.10,
         "range_pos": 0.10, "rel_strength": 0.08, "long_term": 0.12, "thrust": 0.08}
    comp = {"ema_stack": c_stack, "slope": c_slope, "macd": c_macd, "rsi": c_rsi,
            "range_pos": c_rpos, "rel_strength": c_rel, "long_term": c_long,
            "thrust": c_thrust}
    raw = sum(comp[k].fillna(0.0) * w for k, w in W.items())
    score = (raw * 100.0).clip(-100, 100)

    strength = (score.abs() * 0.45
                + ((adx_s - 12.0) / 0.28).clip(0, 100) * 0.35
                + (r2_s * 100.0).clip(0, 100) * 0.20).clip(0, 100)

    label = pd.Series(TREND_NEUTRAL, index=close.index, dtype=object)
    label[(score >= 18)] = TREND_UP
    label[(score <= -18)] = TREND_DOWN
    label[(score >= 45) & (strength >= 55)] = TREND_UP_STRONG
    label[(score <= -45) & (strength >= 55)] = TREND_DOWN_STRONG

    out["score"] = score
    out["strength"] = strength
    out["label"] = label
    # Bars before the indicators are warm carry no opinion.
    warm = adx_s.notna() & r2_s.notna() & e50.notna()
    out.loc[~warm, ["score", "strength"]] = np.nan
    out.loc[~warm, "label"] = None
    return out
