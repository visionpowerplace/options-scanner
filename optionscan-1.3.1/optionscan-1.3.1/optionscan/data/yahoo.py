"""
Yahoo Finance provider (via yfinance). Free, no API key, good enough for daily
end-of-day scanning of liquid names.

Known limitations, handled explicitly rather than ignored:
  * Quotes are delayed ~15 minutes. Fine for a once-a-day scan; do NOT use for
    intraday execution -- always re-check the live quote in your broker.
  * Outside market hours bid/ask frequently come back as 0. Those rows are
    flagged `quote_stale` and the mid falls back to last price, and the
    liquidity filter can be told to reject them outright.
  * No implied-vol history, which is why IVHistoryStore exists.
  * Yahoo's own `impliedVolatility` field is unreliable on wide markets, so the
    scanner re-solves IV from the mid price itself and only uses Yahoo's number
    as a cross-check.
"""
from __future__ import annotations

import warnings
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from .base import CHAIN_COLUMNS, DataProvider
from .cache import DiskCache

warnings.filterwarnings("ignore", category=FutureWarning)


class YahooProvider(DataProvider):
    name = "yahoo"

    def __init__(self, cache: DiskCache | None = None,
                 history_ttl: float = 3600.0, chain_ttl: float = 300.0):
        try:
            import yfinance as yf
        except ImportError as exc:                       # pragma: no cover
            raise ImportError(
                "The Yahoo provider needs yfinance. Install it with:\n"
                "    pip install yfinance"
            ) from exc
        self._yf = yf
        self.cache = cache or DiskCache()
        self.history_ttl = history_ttl
        self.chain_ttl = chain_ttl
        self._tickers: dict[str, object] = {}

    def _t(self, ticker: str):
        if ticker not in self._tickers:
            self._tickers[ticker] = self._yf.Ticker(ticker)
        return self._tickers[ticker]

    # ------------------------------------------------------------------ prices
    def get_history(self, ticker: str, lookback_days: int = 400) -> pd.DataFrame:
        key = f"hist_{ticker}_{lookback_days}"
        hit = self.cache.get(key, self.history_ttl)
        if hit is not None:
            return hit
        period = "2y" if lookback_days > 370 else ("1y" if lookback_days > 200 else "6mo")
        df = self._t(ticker).history(period=period, auto_adjust=False, actions=False)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns={c: str(c).lower().replace(" ", "_") for c in df.columns})
        keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
        df = df[keep].dropna(subset=["close"]).tail(lookback_days)
        try:
            df.index = pd.to_datetime(df.index).tz_localize(None)
        except (TypeError, AttributeError):
            df.index = pd.to_datetime(df.index)
        self.cache.set(key, df)
        return df

    def get_spot(self, ticker: str) -> float:
        try:
            fi = self._t(ticker).fast_info
            for k in ("last_price", "lastPrice", "regular_market_price"):
                v = getattr(fi, k, None) if not isinstance(fi, dict) else fi.get(k)
                if v:
                    return float(v)
        except Exception:
            pass
        h = self.get_history(ticker, 10)
        return float(h["close"].iloc[-1]) if not h.empty else 0.0

    # ------------------------------------------------------------------ chains
    def get_expirations(self, ticker: str) -> list[date]:
        key = f"exps_{ticker}"
        hit = self.cache.get(key, self.chain_ttl)
        if hit is not None:
            return hit
        try:
            raw = self._t(ticker).options or ()
        except Exception:
            return []
        out = []
        for s in raw:
            try:
                out.append(datetime.strptime(str(s), "%Y-%m-%d").date())
            except ValueError:
                continue
        out.sort()
        self.cache.set(key, out)
        return out

    def get_chain(self, ticker: str, expiry: date) -> pd.DataFrame:
        key = f"chain_{ticker}_{expiry.isoformat()}"
        hit = self.cache.get(key, self.chain_ttl)
        if hit is not None:
            return hit
        try:
            oc = self._t(ticker).option_chain(expiry.isoformat())
        except Exception:
            return pd.DataFrame(columns=CHAIN_COLUMNS)

        frames = []
        for kind, raw in (("call", oc.calls), ("put", oc.puts)):
            if raw is None or len(raw) == 0:
                continue
            d = pd.DataFrame(raw).copy()
            d["kind"] = kind
            frames.append(d)
        if not frames:
            return pd.DataFrame(columns=CHAIN_COLUMNS)

        df = pd.concat(frames, ignore_index=True)
        dte = max((expiry - date.today()).days, 0)

        out = pd.DataFrame({
            "kind": df["kind"],
            "expiry": expiry,
            "dte": dte,
            "strike": pd.to_numeric(df.get("strike"), errors="coerce"),
            "bid": pd.to_numeric(df.get("bid"), errors="coerce").fillna(0.0),
            "ask": pd.to_numeric(df.get("ask"), errors="coerce").fillna(0.0),
            "last": pd.to_numeric(df.get("lastPrice"), errors="coerce").fillna(0.0),
            "volume": pd.to_numeric(df.get("volume"), errors="coerce").fillna(0.0),
            "open_interest": pd.to_numeric(df.get("openInterest"), errors="coerce").fillna(0.0),
            "iv_quoted": pd.to_numeric(df.get("impliedVolatility"), errors="coerce"),
            "contract": df.get("contractSymbol", pd.Series([""] * len(df))).astype(str),
        })

        # A two-sided market is required for a trustworthy mid. Where Yahoo
        # gives one (common after hours) fall back to last and flag the row.
        two_sided = (out["bid"] > 0) & (out["ask"] > 0) & (out["ask"] >= out["bid"])
        out["quote_stale"] = ~two_sided
        out["mid"] = np.where(two_sided, (out["bid"] + out["ask"]) / 2.0, out["last"])
        out = out[(out["strike"] > 0) & (out["mid"] > 0)].reset_index(drop=True)
        self.cache.set(key, out)
        return out

    # ----------------------------------------------------------------- extras
    def get_earnings_date(self, ticker: str) -> date | None:
        key = f"earn_{ticker}"
        hit = self.cache.get(key, 6 * 3600.0)
        if hit is not None:
            return hit if hit != "none" else None
        found = None
        try:
            cal = self._t(ticker).calendar
            cand = None
            if isinstance(cal, dict):
                cand = cal.get("Earnings Date")
            elif isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
                cand = cal.loc["Earnings Date"].tolist()
            if cand is not None:
                if not isinstance(cand, (list, tuple, np.ndarray)):
                    cand = [cand]
                ds = []
                for c in cand:
                    try:
                        ds.append(pd.Timestamp(c).date())
                    except Exception:
                        continue
                future = sorted(d for d in ds if d >= date.today())
                found = future[0] if future else (sorted(ds)[-1] if ds else None)
        except Exception:
            found = None
        if found is None:
            try:
                ed = self._t(ticker).get_earnings_dates(limit=8)
                if ed is not None and not ed.empty:
                    idx = pd.to_datetime(ed.index)
                    try:
                        idx = idx.tz_localize(None)
                    except (TypeError, AttributeError):
                        pass
                    future = sorted(d.date() for d in idx if d.date() >= date.today())
                    found = future[0] if future else None
            except Exception:
                found = None
        self.cache.set(key, found or "none")
        return found

    def get_dividend_yield(self, ticker: str) -> float:
        """
        Computed from the trailing 12 months of actual dividends rather than
        Yahoo's `dividendYield` field, whose units flip between percent and
        decimal across versions and would silently mis-price every option.
        """
        key = f"div_{ticker}"
        hit = self.cache.get(key, 24 * 3600.0)
        if hit is not None:
            return float(hit)
        y = 0.0
        try:
            divs = self._t(ticker).dividends
            if divs is not None and len(divs):
                idx = pd.to_datetime(divs.index)
                try:
                    idx = idx.tz_localize(None)
                except (TypeError, AttributeError):
                    pass
                recent = divs[idx >= (pd.Timestamp.today() - pd.Timedelta(days=365))]
                spot = self.get_spot(ticker)
                if spot > 0 and len(recent):
                    y = float(recent.sum()) / spot
        except Exception:
            y = 0.0
        y = min(max(y, 0.0), 0.25)
        self.cache.set(key, y)
        return y

    def get_risk_free_rate(self) -> float:
        """13-week T-bill (^IRX) as the short-rate proxy."""
        hit = self.cache.get("rfr", 12 * 3600.0)
        if hit is not None:
            return float(hit)
        r = 0.04
        try:
            h = self._yf.Ticker("^IRX").history(period="1mo")
            if h is not None and not h.empty:
                v = float(h["Close"].dropna().iloc[-1]) / 100.0
                if 0.0 <= v < 0.25:
                    r = v
        except Exception:
            pass
        self.cache.set("rfr", r)
        return r
