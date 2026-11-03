"""
Enriched option-chain view.

Raw vendor chains cannot be traded off directly. This layer:

  1. Re-solves implied volatility from the MID PRICE for every contract rather
     than trusting the vendor's IV field, which on wide or stale markets can be
     wildly wrong -- and every greek downstream inherits that error.
  2. Computes greeks consistently from that solved IV.
  3. Measures liquidity honestly: absolute and relative bid/ask spread, open
     interest, volume, and whether the quote is two-sided at all.
  4. Drops contracts that cannot be traded at a sane price, BEFORE strategy
     construction, so a beautiful-looking spread is never built on a quote you
     could not actually get filled at.

The relative-spread filter matters more than almost anything else in options.
A spread quoted 1.00/1.40 has a 33% round-trip cost against the mid. No
statistical edge survives that, so those contracts are removed rather than
scored.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .bs import greeks, implied_vol


@dataclass
class LiquidityRules:
    min_open_interest: int = 200
    min_volume: int = 0                 # today's volume; 0 early in the session
    max_spread_pct: float = 0.10        # (ask-bid)/mid
    max_spread_abs: float = 1.50        # dollars, caps the wide-market case
    min_bid: float = 0.05
    require_two_sided: bool = True
    min_mid: float = 0.10

    def describe(self) -> str:
        return (f"OI>={self.min_open_interest}, vol>={self.min_volume}, "
                f"spread<={self.max_spread_pct:.0%} and <=${self.max_spread_abs:.2f}, "
                f"bid>=${self.min_bid:.2f}")


class ChainView:
    """One expiry's calls and puts, enriched and filtered."""

    def __init__(self, ticker: str, spot: float, expiry: date, dte: int,
                 raw: pd.DataFrame, r: float, q: float,
                 rules: LiquidityRules | None = None):
        self.ticker = ticker
        self.spot = float(spot)
        self.expiry = expiry
        self.dte = int(dte)
        self.T = max(dte, 0) / 365.0
        self.r = r
        self.q = q
        self.rules = rules or LiquidityRules()
        self.raw_count = 0 if raw is None else len(raw)
        self.df = self._enrich(raw)
        self.rejected = self.raw_count - len(self.df)

    # ------------------------------------------------------------------
    def _enrich(self, raw: pd.DataFrame) -> pd.DataFrame:
        if raw is None or raw.empty or self.T <= 0:
            return pd.DataFrame()
        d = raw.copy()
        for col in ("bid", "ask", "mid", "last", "volume", "open_interest", "strike"):
            if col not in d.columns:
                d[col] = 0.0
            d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)
        if "quote_stale" not in d.columns:
            d["quote_stale"] = (d["bid"] <= 0) | (d["ask"] <= 0)

        d["spread_abs"] = (d["ask"] - d["bid"]).clip(lower=0.0)
        d["spread_pct"] = np.where(d["mid"] > 0, d["spread_abs"] / d["mid"], np.inf)

        # Re-solve IV from the mid price, then compute greeks from it.
        ivs, deltas, gammas, thetas, vegas = [], [], [], [], []
        for row in d.itertuples(index=False):
            kind = getattr(row, "kind")
            iv = implied_vol(float(row.mid), self.spot, float(row.strike),
                             self.T, self.r, self.q, kind)
            if iv is None:
                ivs.append(np.nan)
                deltas.append(np.nan); gammas.append(np.nan)
                thetas.append(np.nan); vegas.append(np.nan)
                continue
            g = greeks(self.spot, float(row.strike), self.T, self.r, self.q, iv, kind)
            ivs.append(iv); deltas.append(g.delta); gammas.append(g.gamma)
            thetas.append(g.theta); vegas.append(g.vega)

        d["iv"] = ivs
        d["delta"] = deltas
        d["gamma"] = gammas
        d["theta"] = thetas
        d["vega"] = vegas
        d["abs_delta"] = d["delta"].abs()

        # Extrinsic value -- what you actually pay for optionality.
        intr = np.where(d["kind"] == "call",
                        np.maximum(0.0, self.spot - d["strike"]),
                        np.maximum(0.0, d["strike"] - self.spot))
        d["intrinsic"] = intr
        d["extrinsic"] = (d["mid"] - intr).clip(lower=0.0)

        d = d.dropna(subset=["iv", "delta"])
        rl = self.rules
        ok = (
            (d["open_interest"] >= rl.min_open_interest)
            & (d["volume"] >= rl.min_volume)
            & (d["mid"] >= rl.min_mid)
            & (d["bid"] >= rl.min_bid)
            & ((d["spread_pct"] <= rl.max_spread_pct) | (d["spread_abs"] <= 0.05))
            & (d["spread_abs"] <= rl.max_spread_abs)
        )
        if rl.require_two_sided:
            ok &= ~d["quote_stale"].astype(bool)
        out = d[ok].copy()
        out["liq_score"] = self._liquidity_score(out)
        return out.sort_values(["kind", "strike"]).reset_index(drop=True)

    @staticmethod
    def _liquidity_score(d: pd.DataFrame) -> pd.Series:
        """0-100 tradability score: tight spread and deep open interest."""
        if d.empty:
            return pd.Series(dtype=float)
        sp = (1.0 - (d["spread_pct"] / 0.10).clip(0.0, 1.0)) * 100.0
        oi = (np.log1p(d["open_interest"]) / np.log1p(20_000)).clip(0.0, 1.0) * 100.0
        vo = (np.log1p(d["volume"]) / np.log1p(3_000)).clip(0.0, 1.0) * 100.0
        return (sp * 0.50 + oi * 0.32 + vo * 0.18).round(1)

    # ------------------------------------------------------------- accessors
    @property
    def empty(self) -> bool:
        return self.df.empty

    def side(self, kind: str) -> pd.DataFrame:
        return self.df[self.df["kind"] == kind]

    def atm_iv(self) -> float:
        """Average IV of the contracts closest to the money (both sides)."""
        if self.df.empty:
            return float("nan")
        d = self.df.assign(dist=(self.df["strike"] - self.spot).abs())
        return float(d.nsmallest(6, "dist")["iv"].mean())

    def nearest_delta(self, kind: str, target: float,
                      exclude_strikes: set | None = None) -> pd.Series | None:
        """Contract whose |delta| is closest to `target`."""
        s = self.side(kind)
        if exclude_strikes:
            s = s[~s["strike"].isin(exclude_strikes)]
        if s.empty:
            return None
        return s.iloc[(s["abs_delta"] - abs(target)).abs().argsort().iloc[0]]

    def at_or_beyond(self, kind: str, ref_strike: float, width: float) -> pd.Series | None:
        """
        The protective leg of a vertical: `width` dollars further out of the
        money than `ref_strike`, snapped to the nearest available strike that
        exists and passes the liquidity filter.
        """
        s = self.side(kind)
        if s.empty:
            return None
        target = ref_strike + width if kind == "call" else ref_strike - width
        cand = s[s["strike"] > ref_strike] if kind == "call" else s[s["strike"] < ref_strike]
        if cand.empty:
            return None
        return cand.iloc[(cand["strike"] - target).abs().argsort().iloc[0]]

    def strike_step(self) -> float:
        """Median gap between adjacent strikes -- the ladder's granularity."""
        ks = np.sort(self.df["strike"].unique())
        return float(np.median(np.diff(ks))) if len(ks) > 2 else 1.0

    def summary(self) -> dict:
        return {
            "ticker": self.ticker, "expiry": self.expiry, "dte": self.dte,
            "contracts_raw": self.raw_count, "contracts_tradable": len(self.df),
            "rejected": self.rejected, "atm_iv": self.atm_iv(),
            "median_spread_pct": float(self.df["spread_pct"].median()) if not self.df.empty else float("nan"),
        }
