"""
Technical and volatility indicators.

All functions take/return pandas objects aligned to the input index and never
look ahead: every value at bar t uses only data up to and including t. That
property is what makes the backtester honest, so it is tested explicitly in
tests/test_indicators.py.

Wilder's original smoothing (alpha = 1/n) is used for RSI, ATR and ADX so the
numbers match what your charting platform shows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------- trend
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).mean()


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Returns (macd_line, signal_line, histogram)."""
    line = s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def rsi(s: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI."""
    d = s.diff()
    gain = d.clip(lower=0.0)
    loss = (-d).clip(lower=0.0)
    ag = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    al = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = ag / al.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(100.0).where(al.notna(), np.nan)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range, Wilder-smoothed. The scanner's unit of distance."""
    return true_range(df).ewm(alpha=1.0 / window, adjust=False,
                              min_periods=window).mean()


def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Wilder's ADX -- trend STRENGTH regardless of direction.
    Above ~25 means a real trend; below ~20 means chop, which is the single
    most useful filter for not buying directional premium in a range.
    """
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = 1.0 / window
    tr_s = true_range(df).ewm(alpha=a, adjust=False, min_periods=window).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=a, adjust=False, min_periods=window).mean() / tr_s
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=a, adjust=False, min_periods=window).mean() / tr_s
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=a, adjust=False, min_periods=window).mean()


def directional_index(df: pd.DataFrame, window: int = 14):
    """(+DI, -DI) for reading which side the trend is on."""
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = 1.0 / window
    tr_s = true_range(df).ewm(alpha=a, adjust=False, min_periods=window).mean()
    p = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=a, adjust=False,
                                                       min_periods=window).mean() / tr_s
    m = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=a, adjust=False,
                                                        min_periods=window).mean() / tr_s
    return p, m


def log_slope(s: pd.Series, window: int = 40):
    """
    OLS slope of log price over `window` bars, annualised, with R-squared.
    Slope says how fast; R-squared says how orderly. A high slope with low
    R-squared is a spike, not a trend -- a distinction that matters a lot when
    choosing between buying and selling premium.
    """
    n = len(s)
    slope = pd.Series(np.nan, index=s.index)
    r2 = pd.Series(np.nan, index=s.index)
    if n < window:
        return slope, r2
    y_all = np.log(s.to_numpy(dtype=float))
    x = np.arange(window, dtype=float)
    x_c = x - x.mean()
    denom = (x_c ** 2).sum()
    for i in range(window - 1, n):
        y = y_all[i - window + 1: i + 1]
        if not np.all(np.isfinite(y)):
            continue
        y_c = y - y.mean()
        b = float((x_c * y_c).sum() / denom)
        resid = y_c - b * x_c
        ss_tot = float((y_c ** 2).sum())
        slope.iloc[i] = b * TRADING_DAYS
        r2.iloc[i] = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 1e-18 else 0.0
    return slope, r2


# ---------------------------------------------------------------- volatility
def log_returns(s: pd.Series) -> pd.Series:
    return np.log(s / s.shift(1))


def realized_vol(s: pd.Series, window: int = 20) -> pd.Series:
    """Annualised close-to-close realised volatility."""
    return log_returns(s).rolling(window, min_periods=window).std(ddof=1) * np.sqrt(TRADING_DAYS)


def parkinson_vol(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Parkinson high-low estimator -- about 5x more efficient than close-to-close
    for the same window, so it reacts faster without being noisier. Used as a
    cross-check on the realised-vol number that drives the IV/RV edge.
    """
    hl = np.log(df["high"] / df["low"]) ** 2
    return np.sqrt(hl.rolling(window, min_periods=window).mean()
                   / (4.0 * np.log(2.0)) * TRADING_DAYS)


def percent_rank(s: pd.Series, window: int = 252) -> pd.Series:
    """Percentile (0-100) of each value within its own trailing window."""
    def _pr(a: np.ndarray) -> float:
        return float((a[:-1] < a[-1]).mean() * 100.0) if len(a) > 1 else 50.0
    return s.rolling(window, min_periods=max(20, window // 5)).apply(_pr, raw=True)


def range_position(df: pd.DataFrame, window: int = 60) -> pd.Series:
    """Where price sits in its N-day range: 0 = at the low, 1 = at the high."""
    hi = df["high"].rolling(window, min_periods=window).max()
    lo = df["low"].rolling(window, min_periods=window).min()
    return ((df["close"] - lo) / (hi - lo).replace(0.0, np.nan)).clip(0.0, 1.0)


def bollinger(s: pd.Series, window: int = 20, n_std: float = 2.0):
    m = sma(s, window)
    sd = s.rolling(window, min_periods=window).std(ddof=1)
    return m - n_std * sd, m, m + n_std * sd


def bb_width(s: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    lo, mid, hi = bollinger(s, window, n_std)
    return (hi - lo) / mid


def volume_trend(df: pd.DataFrame, fast: int = 10, slow: int = 50) -> pd.Series:
    """Ratio of recent to baseline volume. >1.2 means participation is rising."""
    return (df["volume"].rolling(fast, min_periods=fast).mean()
            / df["volume"].rolling(slow, min_periods=slow).mean())


def max_drawdown(s: pd.Series) -> float:
    """Worst peak-to-trough decline over the whole series (as a negative decimal)."""
    if s.empty:
        return 0.0
    curve = s.to_numpy(dtype=float)
    peak = np.maximum.accumulate(curve)
    return float(np.min(curve / peak - 1.0))


def relative_strength(s: pd.Series, bench: pd.Series, window: int = 63) -> float:
    """
    Excess return vs a benchmark over `window` bars. Positive means the name is
    leading the market -- a meaningful tiebreaker between similar-looking setups.
    """
    a, b = s.dropna(), bench.dropna()
    idx = a.index.intersection(b.index)
    if len(idx) < window + 1:
        return 0.0
    a, b = a.loc[idx], b.loc[idx]
    return float((a.iloc[-1] / a.iloc[-window - 1] - 1.0)
                 - (b.iloc[-1] / b.iloc[-window - 1] - 1.0))


def gap_risk(df: pd.DataFrame, window: int = 120) -> float:
    """
    95th percentile of absolute overnight gaps. Options positions carry gap
    risk that intraday stops cannot protect against, so short-premium sizing
    should respect this rather than assume continuous trading.
    """
    if len(df) < 20:
        return 0.0
    g = (df["open"] / df["close"].shift(1) - 1.0).abs().dropna().tail(window)
    return float(np.percentile(g, 95)) if len(g) else 0.0


def expected_move(spot: float, iv: float, dte: int) -> float:
    """
    One-standard-deviation move implied by the option market over `dte` days.
    This is the single most useful sanity check on a strike: a short strike
    inside the expected move is a coin flip dressed up as income.
    """
    return float(spot * iv * np.sqrt(max(dte, 0) / 365.0))
