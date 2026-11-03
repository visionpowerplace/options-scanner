"""
On-disk caches.

Two distinct jobs:

1. `DiskCache` -- a plain TTL cache so a rerun inside the same session does not
   re-hammer the data provider. Price history gets hours, chains get minutes.

2. `IVHistoryStore` -- the piece that makes IV RANK possible at all.
   Free data sources publish only *today's* implied vol, never its history, so
   IV rank (where today's IV sits in its own 1-year range) cannot be computed
   from a single call. This store appends one row per ticker per day, so the
   series builds itself up as you run the scanner. Until it holds enough
   observations the scanner falls back to an explicitly-labelled realised-vol
   proxy rather than quietly inventing a number.
"""
from __future__ import annotations

import csv
import json
import os
import pickle
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

DEFAULT_DIR = Path(os.environ.get("OPTIONSCAN_HOME", Path.home() / ".optionscan"))


class DiskCache:
    def __init__(self, root: Path | str = DEFAULT_DIR, enabled: bool = True):
        self.root = Path(root) / "cache"
        self.enabled = enabled
        if enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
        return self.root / f"{safe}.pkl"

    def get(self, key: str, ttl_seconds: float):
        if not self.enabled:
            return None
        p = self._path(key)
        try:
            if not p.exists() or (time.time() - p.stat().st_mtime) > ttl_seconds:
                return None
            with p.open("rb") as fh:
                return pickle.load(fh)
        except Exception:
            return None

    def set(self, key: str, value) -> None:
        if not self.enabled:
            return
        try:
            tmp = self._path(key).with_suffix(".tmp")
            with tmp.open("wb") as fh:
                pickle.dump(value, fh)
            tmp.replace(self._path(key))
        except Exception:
            pass

    def clear(self) -> int:
        n = 0
        if self.root.exists():
            for p in self.root.glob("*.pkl"):
                try:
                    p.unlink(); n += 1
                except Exception:
                    pass
        return n


class IVHistoryStore:
    """Append-only daily record of ATM implied vol and realised vol per ticker."""

    HEADER = ["date", "ticker", "atm_iv", "rv20", "rv60", "spot"]

    def __init__(self, root: Path | str = DEFAULT_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "iv_history.csv"
        if not self.path.exists():
            with self.path.open("w", newline="") as fh:
                csv.writer(fh).writerow(self.HEADER)
        self._cache: pd.DataFrame | None = None

    def record(self, ticker: str, atm_iv: float, rv20: float,
               rv60: float, spot: float, on: date | None = None) -> None:
        """Write today's observation, replacing any existing row for the same day."""
        on = on or date.today()
        df = self.load()
        if not df.empty:
            dup = (df["ticker"] == ticker) & (df["date"] == pd.Timestamp(on))
            if dup.any():
                df = df[~dup]
                self._rewrite(df)
        with self.path.open("a", newline="") as fh:
            csv.writer(fh).writerow([
                on.isoformat(), ticker,
                f"{atm_iv:.6f}" if atm_iv is not None else "",
                f"{rv20:.6f}" if rv20 is not None else "",
                f"{rv60:.6f}" if rv60 is not None else "",
                f"{spot:.4f}" if spot is not None else "",
            ])
        self._cache = None

    def _rewrite(self, df: pd.DataFrame) -> None:
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        out.to_csv(self.path, index=False, columns=self.HEADER)
        self._cache = None

    def load(self) -> pd.DataFrame:
        if self._cache is not None:
            return self._cache
        try:
            df = pd.read_csv(self.path)
            if df.empty:
                self._cache = df
                return df
            df["date"] = pd.to_datetime(df["date"])
            for c in ("atm_iv", "rv20", "rv60", "spot"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            self._cache = df
        except Exception:
            self._cache = pd.DataFrame(columns=self.HEADER)
        return self._cache

    def series(self, ticker: str, lookback_days: int = 365) -> pd.Series:
        """ATM IV series for one ticker over the lookback window."""
        df = self.load()
        if df.empty:
            return pd.Series(dtype=float)
        cutoff = pd.Timestamp(date.today()) - pd.Timedelta(days=lookback_days)
        sub = df[(df["ticker"] == ticker) & (df["date"] >= cutoff)]
        return sub.set_index("date")["atm_iv"].dropna().sort_index()

    def iv_rank(self, ticker: str, current_iv: float,
                lookback_days: int = 365, min_obs: int = 40) -> tuple[float | None, int]:
        """
        IV rank = (IV - min) / (max - min) over the window, in 0..100.
        Returns (rank, n_observations); rank is None until min_obs rows exist,
        so the caller can fall back to the labelled proxy instead of guessing.
        """
        s = self.series(ticker, lookback_days)
        n = len(s)
        if n < min_obs or current_iv is None:
            return None, n
        lo, hi = float(s.min()), float(s.max())
        if hi - lo < 1e-9:
            return 50.0, n
        return float(min(max((current_iv - lo) / (hi - lo) * 100.0, 0.0), 100.0)), n

    def iv_percentile(self, ticker: str, current_iv: float,
                      lookback_days: int = 365, min_obs: int = 40) -> tuple[float | None, int]:
        """IV percentile = % of days in the window with IV below today's."""
        s = self.series(ticker, lookback_days)
        n = len(s)
        if n < min_obs or current_iv is None:
            return None, n
        return float((s < current_iv).mean() * 100.0), n


class PositionLog:
    """
    Append-only record of what the scanner suggested, so you can audit it
    later against what actually happened. Honest performance tracking needs
    this: a scanner you cannot grade is a scanner you cannot trust.
    """

    def __init__(self, root: Path | str = DEFAULT_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "suggestions.jsonl"

    def append(self, records: list[dict]) -> None:
        with self.path.open("a") as fh:
            for r in records:
                fh.write(json.dumps(r, default=str) + "\n")

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame()
        rows = []
        with self.path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
        return pd.DataFrame(rows)
