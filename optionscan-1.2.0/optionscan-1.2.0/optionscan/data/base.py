"""Market-data provider contract. Swap the implementation, keep the scanner."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

# Standard chain columns every provider must return.
CHAIN_COLUMNS = [
    "kind", "expiry", "dte", "strike", "bid", "ask", "mid", "last",
    "volume", "open_interest", "iv_quoted", "contract",
]


@dataclass
class Snapshot:
    """Everything the scanner needs about one underlying at one point in time."""
    ticker: str
    spot: float
    history: pd.DataFrame                  # Date-indexed OHLCV, oldest first
    chains: dict[date, pd.DataFrame] = field(default_factory=dict)
    earnings_date: date | None = None
    dividend_yield: float = 0.0
    name: str = ""
    avg_dollar_volume: float = 0.0
    errors: list[str] = field(default_factory=list)


class DataProvider(ABC):
    """
    Implement this to plug in any data source (Yahoo, Polygon, Tradier, IBKR,
    Schwab, ThetaData...). The scanner only ever talks to this interface.
    """

    name: str = "base"

    @abstractmethod
    def get_history(self, ticker: str, lookback_days: int = 400) -> pd.DataFrame:
        """Daily OHLCV, oldest first, DatetimeIndex. Columns: open/high/low/close/volume."""

    @abstractmethod
    def get_expirations(self, ticker: str) -> list[date]:
        """Available option expiration dates, ascending."""

    @abstractmethod
    def get_chain(self, ticker: str, expiry: date) -> pd.DataFrame:
        """Calls and puts for one expiry, in CHAIN_COLUMNS form."""

    @abstractmethod
    def get_spot(self, ticker: str) -> float:
        """Last traded / current price of the underlying."""

    def get_earnings_date(self, ticker: str) -> date | None:
        """Next scheduled earnings date, if known. Override where available."""
        return None

    def get_dividend_yield(self, ticker: str) -> float:
        """Trailing annual dividend yield as a decimal. Override where available."""
        return 0.0

    def get_risk_free_rate(self) -> float:
        """Short-term risk-free rate as a decimal. Override where available."""
        return 0.04
