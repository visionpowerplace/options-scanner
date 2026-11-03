"""
The trade journal.

This is the whole point of paper trading. Not "did it make money" -- a few
weeks of paper P&L is noise -- but the one question that decides whether the
scanner is worth trusting:

    did the high-scoring ideas actually do better than the low-scoring ones?

If score correlates with outcome, the scoring is measuring something real and
you can size by it. If it does not, the scanner is an expensive random number
generator and no amount of live trading will fix that. So every trade is
recorded with the score and probabilities it was opened on, and the report
groups outcomes by score bucket.

Stored as JSON so it survives between runs and is readable without tools.

SIGN CONVENTION, used everywhere: a net price is a DEBIT.
    positive = you pay, negative = you receive.
    P&L per contract = (-entry_net - exit_net) x 100
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path

import pandas as pd

MULT = 100


@dataclass
class JournalEntry:
    id: str
    ticker: str
    strategy: str
    opened_on: str
    expiry: str
    legs: list                      # [{symbol, side, ratio}]
    qty: int = 1

    # what the scanner believed at signal time
    score: float = 0.0
    pop: float = 0.0
    pop_at_rv: float = 0.0
    breakeven_win_rate: float = 0.0
    ev_at_rv: float = 0.0
    scan_net: float = 0.0           # price the scan assumed
    iv_rank: float | None = None
    iv_rv_ratio: float | None = None

    # execution
    entry_order_id: str = ""
    entry_net: float | None = None  # actual fill
    entry_slippage: float = 0.0     # fill minus scan price, in dollars/contract
    target_net: float = 0.0         # closing net that takes profit
    stop_net: float = 0.0           # closing net that stops out
    time_stop_dte: int = 21
    max_loss: float = 0.0

    status: str = "pending"         # pending | open | closed | abandoned
    exit_order_id: str = ""
    exit_net: float | None = None
    closed_on: str = ""
    exit_reason: str = ""
    pnl: float | None = None
    exit_attempts: int = 0
    pending_exit_reason: str = ""
    notes: list = field(default_factory=list)

    def as_dict(self):
        return asdict(self)

    def compute_pnl(self) -> float | None:
        if self.entry_net is None or self.exit_net is None:
            return None
        return round((-self.entry_net - self.exit_net) * self.qty * MULT, 2)


class Journal:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: list[JournalEntry] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text() or "[]")
        except (json.JSONDecodeError, OSError):
            raw = []
        fields = set(JournalEntry.__dataclass_fields__)
        self.entries = [JournalEntry(**{k: v for k, v in r.items() if k in fields})
                        for r in raw]

    def save(self) -> None:
        self.path.write_text(json.dumps([e.as_dict() for e in self.entries],
                                        indent=2, default=str))

    # ------------------------------------------------------------------
    def by_id(self, tid: str) -> JournalEntry | None:
        return next((e for e in self.entries if e.id == tid), None)

    def open_entries(self) -> list:
        return [e for e in self.entries if e.status in ("pending", "open")]

    def closed_entries(self) -> list:
        return [e for e in self.entries if e.status == "closed"]

    def has_open_in(self, ticker: str) -> bool:
        return any(e.ticker == ticker and e.status in ("pending", "open")
                   for e in self.entries)

    def add(self, entry: JournalEntry) -> JournalEntry:
        if self.by_id(entry.id) is None:
            self.entries.append(entry)
        return entry

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([e.as_dict() for e in self.entries])


# --------------------------------------------------------------------------
def grade(journal: Journal) -> str:
    """
    The report that answers the only question worth asking.

    Deliberately blunt about sample size: with fewer than ~30 closed trades
    none of these numbers mean anything, and saying so is more useful than a
    confident-looking table built on four trades.
    """
    L: list[str] = []
    A = L.append
    W = 86
    closed = journal.closed_entries()
    openn = journal.open_entries()

    A("=" * W)
    A("  PAPER TRADING SCORECARD".center(W))
    A("=" * W)
    A(f"  Closed trades: {len(closed)}      Still open: {len(openn)}")

    if not closed:
        A("")
        A("  Nothing has closed yet. Come back after a few weeks -- the scorecard")
        A("  needs completed trades, not open ones, because an open position's")
        A("  paper P&L tells you nothing until it is realised.")
        A("=" * W)
        return "\n".join(L)

    pnls = [e.pnl for e in closed if e.pnl is not None]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    total = sum(pnls)
    A(f"  Net P&L: ${total:+,.2f}      Win rate: {len(wins)/len(pnls):.0%}"
      f"      Avg win ${(sum(wins)/len(wins) if wins else 0):+,.2f}"
      f"      Avg loss ${(sum(losses)/len(losses) if losses else 0):+,.2f}")

    if len(closed) < 30:
        A("")
        A(f"  *** {len(closed)} trades is NOT a sample. Anything below is noise until you")
        A(f"      have 30+, and really 100+. Do not change the system based on this. ***")

    # ---------------- the real test: score vs outcome ----------------
    A("")
    A("-" * W)
    A("  DOES A HIGHER SCORE MEAN A BETTER OUTCOME?")
    A("-" * W)
    df = pd.DataFrame([{"score": e.score, "pnl": e.pnl,
                        "risk": max(e.max_loss, 1.0),
                        "win": 1 if (e.pnl or 0) > 0 else 0} for e in closed
                       if e.pnl is not None])
    if len(df) >= 4:
        df["bucket"] = pd.cut(df["score"], [0, 55, 65, 75, 100],
                              labels=["<55", "55-65", "65-75", "75+"])
        A(f"  {'SCORE':<10}{'N':>5}{'NET P&L':>12}{'AVG':>10}{'WIN RATE':>10}{'RETURN ON RISK':>17}")
        for b, g in df.groupby("bucket", observed=True):
            ror = (g["pnl"].sum() / g["risk"].sum() * 100) if g["risk"].sum() else 0
            A(f"  {str(b):<10}{len(g):>5}{g['pnl'].sum():>+12,.2f}{g['pnl'].mean():>+10,.2f}"
              f"{g['win'].mean():>10.0%}{ror:>16.1f}%")
        if df["score"].nunique() > 2:
            r = df["score"].corr(df["pnl"], method="spearman")
            A("")
            A(f"  Rank correlation between score and P&L: {r:+.2f}")
            A("  Reading: " + (
                "the score is predictive -- size by it" if r > 0.35 else
                "weak positive, keep collecting" if r > 0.1 else
                "no relationship -- the score is not earning its keep" if r > -0.1 else
                "INVERTED -- high scores are doing worse; stop and investigate"))
    else:
        A("  Not enough closed trades to bucket yet.")

    # ---------------- execution quality ----------------
    A("")
    A("-" * W)
    A("  EXECUTION: did the fills match what the scanner assumed?")
    A("-" * W)
    slips = [e.entry_slippage for e in closed + openn if e.entry_slippage]
    if slips:
        avg = sum(slips) / len(slips)
        A(f"  Average entry slippage: ${avg:+.2f} per contract across {len(slips)} fills")
        A(f"  Total given up to slippage: ${sum(slips):+,.2f}")
        A("")
        A("  This is the number that kills small-account option systems. If it is")
        A("  comparable to the average win, the edge is being eaten by execution and")
        A("  no amount of better signals will help -- you need wider spreads, more")
        A("  liquid underlyings, or more patience on the limit.")
    else:
        A("  No fills recorded yet.")

    # ---------------- why trades ended ----------------
    A("")
    A("-" * W)
    A("  HOW TRADES ENDED")
    A("-" * W)
    reasons: dict[str, list] = {}
    for e in closed:
        reasons.setdefault(e.exit_reason or "unknown", []).append(e.pnl or 0)
    for r, ps in sorted(reasons.items(), key=lambda kv: -len(kv[1])):
        A(f"  {r:<24}{len(ps):>4} trades   net ${sum(ps):>+10,.2f}   avg ${sum(ps)/len(ps):>+8,.2f}")

    A("")
    A("=" * W)
    return "\n".join(L)
