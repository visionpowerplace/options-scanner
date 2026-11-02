"""
Synthetic provider: a self-contained market simulator.

Why this exists. An options scanner is a long pipeline -- indicators, regime
classification, strike selection, liquidity filters, greeks, sizing, scoring --
and testing it against the live market is slow, non-reproducible, and
impossible when a data feed is unreachable. This provider generates
deterministic, realistic data so the whole pipeline can be exercised and
verified end to end: trending and ranging names, a proper volatility surface
with put skew and term structure, bid/ask spreads that widen in the wings,
open interest concentrated near the money, and scheduled earnings.

It is for testing and demos. Run the scanner with --provider yahoo for real
money. The scanner labels every report with the provider it used.
"""
from __future__ import annotations

import zlib
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .base import CHAIN_COLUMNS, DataProvider


def stable_hash(text: str) -> int:
    """
    Deterministic across processes, unlike the builtin hash() for strings,
    which is salted per interpreter by PYTHONHASHSEED. A test fixture that
    silently changes every time you run it is not a fixture, so every seed
    derived from a symbol name goes through this.
    """
    return zlib.crc32(str(text).encode("utf-8"))


def strike_increment(price: float, liquidity: float = 0.7) -> float:
    """
    Approximate the real OCC strike ladder.

    Liquidity matters as much as price level. SPY at $560 lists $1-wide strikes,
    not the $10 a price-only rule would give it -- and that fine ladder is
    precisely what makes the big ETFs the only realistic vehicle for a small
    account, since it allows $1-wide spreads risking well under $100. Getting
    this wrong would make the simulator useless for exactly that case.
    """
    if liquidity >= 0.95:
        return 0.5 if price < 25 else 1.0
    if price < 15:
        return 0.5
    if price < 50:
        return 1.0
    if price < 100:
        return 2.5
    if price < 400:
        return 5.0
    return 10.0


# ticker -> (start_price, annual_drift, annual_vol, base_iv, liquidity 0..1, regime label)
DEFAULT_SPECS: dict[str, tuple] = {
    "SPY":  (560.0,  0.11, 0.14, 0.155, 1.00, "grind-up"),
    "QQQ":  (480.0,  0.15, 0.20, 0.225, 1.00, "grind-up"),
    "IWM":  (215.0,  0.04, 0.22, 0.240, 0.85, "range"),
    "AAPL": (228.0,  0.10, 0.24, 0.270, 0.95, "grind-up"),
    "MSFT": (420.0,  0.12, 0.23, 0.250, 0.95, "grind-up"),
    "NVDA": (135.0,  0.35, 0.50, 0.560, 1.00, "strong-up"),
    "AMD":  (145.0, -0.18, 0.52, 0.600, 0.90, "down"),
    "TSLA": (250.0,  0.05, 0.58, 0.640, 1.00, "range-wide"),
    "AMZN": (185.0,  0.14, 0.30, 0.320, 0.92, "grind-up"),
    "META": (560.0,  0.20, 0.33, 0.350, 0.90, "strong-up"),
    "GOOGL":(165.0,  0.08, 0.28, 0.300, 0.90, "range"),
    "JPM":  (225.0,  0.09, 0.21, 0.225, 0.80, "grind-up"),
    "XOM":  (118.0, -0.05, 0.24, 0.255, 0.75, "down"),
    "WMT":  (82.0,   0.13, 0.19, 0.205, 0.70, "grind-up"),
    "COIN": (210.0, -0.25, 0.70, 0.820, 0.85, "down-vol"),
    "INTC": (23.0,  -0.30, 0.45, 0.520, 0.80, "down"),
    "BA":   (155.0, -0.10, 0.38, 0.420, 0.75, "range"),
    "DIS":  (95.0,   0.06, 0.27, 0.290, 0.70, "range"),
    "PLTR": (42.0,   0.45, 0.62, 0.700, 0.88, "strong-up"),
    "SMCI": (38.0,  -0.40, 0.85, 0.950, 0.70, "down-vol"),
}


class SyntheticProvider(DataProvider):
    name = "synthetic"

    def __init__(self, seed: int = 20260501, specs: dict | None = None,
                 as_of: date | None = None, lookback: int = 400):
        self.specs = specs or DEFAULT_SPECS
        self.as_of = as_of or date.today()
        self.seed = seed
        self.lookback = lookback
        self._hist: dict[str, pd.DataFrame] = {}
        self._earn: dict[str, date] = {}

    # --------------------------------------------------------------- universe
    def universe(self) -> list[str]:
        return list(self.specs.keys())

    def _spec(self, ticker: str) -> tuple:
        if ticker in self.specs:
            return self.specs[ticker]
        rng = np.random.default_rng(stable_hash(ticker) % (2**31))
        return (float(rng.uniform(30, 300)), float(rng.normal(0.05, 0.2)),
                float(rng.uniform(0.2, 0.6)), 0.0, 0.7, "random")

    # ---------------------------------------------------------------- history
    def get_history(self, ticker: str, lookback_days: int = 400) -> pd.DataFrame:
        if ticker in self._hist:
            return self._hist[ticker].tail(lookback_days)
        s0, drift, vol, _, _, regime = self._spec(ticker)
        n = max(lookback_days, self.lookback)
        rng = np.random.default_rng(self.seed + stable_hash(ticker) % 100_000)
        dt = 1.0 / 252.0

        # Regime must be GROUND TRUTH, not a lucky draw. A pure random walk
        # with 50% vol routinely produces a 6-month downtrend under positive
        # drift, which makes it useless as a test fixture for a trend
        # classifier. So the path is built as:
        #     deterministic trend  +  stationary mean-reverting noise
        # anchored so the final bar equals the spec price. Daily returns keep
        # realistic magnitude and volatility clustering (the noise increments
        # carry it), but the regime the label claims is the regime present.
        total_move = {
            "strong-up": 0.62, "grind-up": 0.30, "down": -0.38,
            "down-vol": -0.55, "range": 0.0, "range-wide": 0.0, "random": drift,
        }.get(regime, drift)

        shocks = rng.standard_normal(n)
        # GARCH-style volatility clustering.
        v = np.empty(n)
        v[0] = vol
        for i in range(1, n):
            v[i] = np.sqrt(max(1e-6,
                0.92 * v[i - 1] ** 2 + 0.06 * vol ** 2 + 0.02 * (vol * shocks[i - 1]) ** 2))

        # Stationary AR(1) wiggle: daily increments have vol v*sqrt(dt), but the
        # level mean-reverts so it cannot overwhelm the trend.
        phi = 0.985
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = phi * x[i - 1] + v[i] * np.sqrt(dt) * shocks[i]

        t = np.linspace(0.0, 1.0, n)
        if regime.startswith("range"):
            amp = 0.17 if regime == "range" else 0.30
            trend = amp * np.sin(2.0 * np.pi * 1.75 * t)
        else:
            trend = total_move * t

        log_price = np.log(s0) + (trend - trend[-1]) + (x - x[-1])
        close = np.exp(log_price)

        intraday = np.abs(rng.standard_normal(n)) * v * np.sqrt(dt) * close * 0.75
        high = close + intraday * rng.uniform(0.3, 1.0, n)
        low = close - intraday * rng.uniform(0.3, 1.0, n)
        open_ = np.concatenate([[s0], close[:-1]]) * (1 + rng.standard_normal(n) * 0.002)
        high = np.maximum.reduce([high, close, open_])
        low = np.minimum.reduce([low, close, open_])
        base_vol = 6_000_000 * (0.4 + self._spec(ticker)[4])
        volume = (base_vol * np.exp(rng.standard_normal(n) * 0.35)).astype(np.int64)

        idx = pd.bdate_range(end=pd.Timestamp(self.as_of), periods=n)
        df = pd.DataFrame({"open": open_, "high": high, "low": low,
                           "close": close, "volume": volume}, index=idx)
        self._hist[ticker] = df
        return df.tail(lookback_days)

    def get_spot(self, ticker: str) -> float:
        h = self.get_history(ticker, 5)
        return float(h["close"].iloc[-1]) if not h.empty else 0.0

    # ----------------------------------------------------------- vol surface
    def _atm_iv(self, ticker: str, T: float) -> float:
        _, _, vol, base_iv, _, _ = self._spec(ticker)
        base = base_iv or (vol * 1.1)
        # Term structure: short-dated richer, flattening out the curve.
        term = 1.0 + 0.16 * np.exp(-6.0 * T) - 0.04 * min(T, 1.0)
        return float(base * term)

    def _iv(self, ticker: str, S: float, K: float, T: float) -> float:
        """ATM IV plus a realistic equity put skew in standardised moneyness."""
        atm = self._atm_iv(ticker, T)
        m = np.log(max(K, 1e-9) / S) / max(atm * np.sqrt(max(T, 1e-6)), 1e-6)
        m = float(np.clip(m, -4.0, 4.0))
        iv = atm * (1.0 - 0.075 * m + 0.022 * m * m)     # downside skew + smile
        return float(np.clip(iv, 0.03, 4.0))

    # --------------------------------------------------------------- expiries
    def get_expirations(self, ticker: str) -> list[date]:
        out, d = [], self.as_of + timedelta(days=1)
        while len(out) < 10 and d <= self.as_of + timedelta(days=250):
            if d.weekday() == 4:                       # Friday
                dte = (d - self.as_of).days
                is_monthly = 15 <= d.day <= 21
                if dte <= 60 or is_monthly:            # weeklies near, monthlies far
                    out.append(d)
            d += timedelta(days=1)
        return out

    def get_chain(self, ticker: str, expiry: date) -> pd.DataFrame:
        from ..core.bs import bs_price

        S = self.get_spot(ticker)
        dte = max((expiry - self.as_of).days, 1)
        T = dte / 365.0
        r, q = 0.0425, self.get_dividend_yield(ticker)
        liq = self._spec(ticker)[4]
        inc = strike_increment(S, liq)
        rng = np.random.default_rng(self.seed + dte + stable_hash(ticker) % 9_999)

        lo = np.floor(S * 0.65 / inc) * inc
        hi = np.ceil(S * 1.35 / inc) * inc
        strikes = np.arange(lo, hi + inc / 2, inc)

        rows = []
        for K in strikes:
            if K <= 0:
                continue
            iv = self._iv(ticker, S, float(K), T)
            moneyness = abs(np.log(float(K) / S))
            for kind in ("call", "put"):
                theo = bs_price(S, float(K), T, r, q, iv, kind)
                if theo < 0.01:
                    continue
                # Spread widens in the wings, with illiquidity, and in short tenors.
                rel = (0.012 + 0.10 * moneyness + 0.02 * np.exp(-dte / 12.0)) / max(liq, 0.2)
                half = max(0.01, min(theo * rel, max(0.02, theo * 0.45)))
                tick = 0.05 if theo >= 3.0 else 0.01
                bid = max(0.0, round((theo - half) / tick) * tick)
                ask = round((theo + half) / tick) * tick
                if ask <= bid:
                    ask = bid + tick
                # OI / volume peak at the money and decay in the wings.
                shape = float(np.exp(-(moneyness / 0.11) ** 2))
                term_w = float(np.exp(-abs(dte - 35) / 55.0))
                oi = int(max(0, rng.normal(9_000 * liq * shape * term_w,
                                           1_200 * liq * shape * term_w + 25)))
                vol_ = int(max(0, rng.normal(oi * 0.17, oi * 0.09 + 5)))
                rows.append({
                    "kind": kind, "expiry": expiry, "dte": dte, "strike": float(K),
                    "bid": round(bid, 2), "ask": round(ask, 2),
                    "mid": round((bid + ask) / 2.0, 4),
                    "last": round((bid + ask) / 2.0 * float(rng.uniform(0.97, 1.03)), 2),
                    "volume": vol_, "open_interest": oi, "iv_quoted": iv,
                    "contract": f"{ticker}{expiry:%y%m%d}{kind[0].upper()}"
                                f"{int(round(float(K) * 1000)):08d}",
                    "quote_stale": False,
                })
        df = pd.DataFrame(rows)
        return df if df.empty else df[df["mid"] > 0].reset_index(drop=True)

    # ----------------------------------------------------------------- extras
    def get_earnings_date(self, ticker: str) -> date | None:
        """ETFs get none; single names get a plausible upcoming date."""
        if ticker in ("SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "GLD", "TLT"):
            return None
        if ticker not in self._earn:
            rng = np.random.default_rng(self.seed + stable_hash(ticker) % 5_000)
            self._earn[ticker] = self.as_of + timedelta(days=int(rng.integers(3, 85)))
        return self._earn[ticker]

    def get_dividend_yield(self, ticker: str) -> float:
        return {"SPY": 0.012, "QQQ": 0.006, "IWM": 0.011, "AAPL": 0.005,
                "MSFT": 0.008, "JPM": 0.021, "XOM": 0.033, "WMT": 0.010,
                "INTC": 0.0, "DIS": 0.008}.get(ticker, 0.0)

    def get_risk_free_rate(self) -> float:
        return 0.0425
