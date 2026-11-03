"""
Broker abstraction.

The trader talks only to this interface, so Alpaca can be swapped for Tradier,
Tastytrade or IBKR by writing one adapter. The offline simulator implements the
same interface, which is how the entire trading loop gets tested without a
network or an account.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import date, datetime


def occ_symbol(underlying: str, expiry: date, kind: str, strike: float) -> str:
    """
    OCC option symbol, the format every US broker speaks.
        AAPL  241220 C 00150000   ->  AAPL241220C00150000
    Root, expiry YYMMDD, C/P, then the strike x1000 padded to 8 digits.
    """
    return (f"{underlying.upper()}{expiry:%y%m%d}"
            f"{'C' if kind.lower().startswith('c') else 'P'}"
            f"{int(round(strike * 1000)):08d}")


def parse_occ(symbol: str) -> dict:
    """Inverse of occ_symbol -- used to read positions back from the broker."""
    s = symbol.strip().upper()
    strike = int(s[-8:]) / 1000.0
    kind = "call" if s[-9] == "C" else "put"
    ymd = s[-15:-9]
    root = s[:-15]
    return {"underlying": root, "expiry": date(2000 + int(ymd[:2]), int(ymd[2:4]),
                                               int(ymd[4:6])),
            "kind": kind, "strike": strike}


@dataclass
class Account:
    equity: float
    cash: float
    buying_power: float
    options_level: int = 0
    is_paper: bool = True
    raw: dict = field(default_factory=dict)


@dataclass
class Position:
    symbol: str                 # OCC symbol
    qty: int                    # negative = short
    avg_price: float            # per share
    market_price: float = 0.0
    unrealized_pl: float = 0.0
    raw: dict = field(default_factory=dict)

    @property
    def is_short(self) -> bool:
        return self.qty < 0


@dataclass
class OrderLeg:
    symbol: str
    side: str                   # 'buy' | 'sell'
    ratio: int = 1


@dataclass
class Order:
    id: str
    status: str                 # new | partially_filled | filled | canceled | rejected
    legs: list = field(default_factory=list)
    qty: int = 0
    filled_qty: int = 0
    limit_price: float = 0.0    # NET price: positive = debit paid, negative = credit
    filled_avg_price: float | None = None
    submitted_at: datetime | None = None
    tag: str = ""               # our own client reference
    raw: dict = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        return self.status in ("new", "accepted", "pending_new", "partially_filled")

    @property
    def is_done(self) -> bool:
        return self.status in ("filled", "canceled", "rejected", "expired")

    def as_dict(self):
        d = asdict(self)
        d["legs"] = [asdict(l) if not isinstance(l, dict) else l for l in self.legs]
        return d


@dataclass
class Quote:
    symbol: str
    bid: float
    ask: float
    last: float = 0.0

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.last

    @property
    def spread(self) -> float:
        return max(0.0, self.ask - self.bid)


class Broker(ABC):
    name = "base"
    is_paper = True

    @abstractmethod
    def get_account(self) -> Account: ...

    @abstractmethod
    def get_positions(self) -> list: ...

    @abstractmethod
    def get_orders(self, status: str = "open") -> list: ...

    @abstractmethod
    def get_quotes(self, symbols: list) -> dict:
        """OCC symbol -> Quote. The live check that a stale scan price still holds."""

    @abstractmethod
    def submit_multileg(self, legs: list, qty: int, limit_price: float,
                        tag: str = "") -> Order:
        """
        Submit a multi-leg (or single-leg) option order.

        `limit_price` is the NET price of the package, using the convention this
        package applies everywhere: POSITIVE means a debit you pay, NEGATIVE
        means a credit you receive.
        """

    @abstractmethod
    def replace_order(self, order_id: str, limit_price: float) -> Order:
        """Improve a working order's price. This is how an order gets worked."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool: ...

    def get_order(self, order_id: str) -> "Order | None":
        """One order by id. Default scans the full list; adapters may override."""
        for o in self.get_orders("all"):
            if o.id == order_id:
                return o
        for o in self.get_orders("open"):
            if o.id == order_id:
                return o
        return None

    def market_is_open(self) -> bool:
        return True
