"""
Configuration. Every threshold the scanner uses lives here, nothing is buried
in the logic, and a YAML file can override any of it.

The defaults are deliberately conservative: liquid names only, defined-risk
structures preferred, 1% of the account risked per trade. Loosen them on
purpose, not by accident.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from pathlib import Path

from .core.chain import LiquidityRules
from .core.risk import AccountConfig


@dataclass
class DTEWindows:
    """
    Days-to-expiry windows per strategy family, chosen for specific reasons:

    credit 25-45  -- the sweet spot where theta is meaningful but gamma risk is
                     still manageable; exits happen at 21 DTE.
    debit  30-60  -- long enough for a directional move to develop without
                     paying for time you will not use.
    long   45-90  -- single options need time; buying 20-DTE calls is a
                     lottery ticket, not a strategy.
    condor 30-50  -- maximum theta per unit of gamma for a neutral position.
    """
    credit: tuple = (25, 45)
    debit: tuple = (30, 60)
    long: tuple = (45, 90)
    condor: tuple = (30, 50)


@dataclass
class DeltaTargets:
    """
    Strike selection by delta rather than by moneyness, so the same setting
    means the same thing on a $20 stock and a $600 stock.
    """
    long_option: float = 0.55       # slightly ITM: more delta per dollar, less theta burn
    debit_long: float = 0.60
    debit_short: float = 0.30
    credit_short: float = 0.25      # ~75% nominal win rate before costs
    condor_short: float = 0.17
    csp_short: float = 0.25
    cc_short: float = 0.30


@dataclass
class SpreadWidth:
    """Vertical width as a fraction of spot, clamped to sane dollar bounds."""
    pct_of_spot: float = 0.045
    min_dollars: float = 1.0
    max_dollars: float = 30.0

    def resolve(self, spot: float, strike_step: float) -> float:
        raw = max(self.min_dollars, min(self.max_dollars, spot * self.pct_of_spot))
        steps = max(1, round(raw / max(strike_step, 0.01)))
        return steps * max(strike_step, 0.01)


@dataclass
class UniverseFilters:
    min_price: float = 12.0
    max_price: float = 2000.0
    min_avg_dollar_volume: float = 25_000_000.0
    min_history_days: int = 120
    # Hard ceiling on what you will PAY to open a position, in dollars per
    # share (1.00 = $100 per contract). Applies to debit trades only; credit
    # trades receive money rather than pay it, so the risk budget governs them.
    # None = no ceiling. Set this on a small account: it is a more direct
    # constraint than the risk percentage and it cannot be satisfied by
    # accident.
    max_debit_per_share: float | None = None


@dataclass
class RegimeThresholds:
    iv_rich: float = 62.0            # IV rank at/above which we prefer selling premium
    iv_cheap: float = 32.0           # IV rank at/below which we prefer buying premium
    min_trend_score: float = 18.0
    min_trend_strength_directional: float = 40.0
    min_adx_directional: float = 18.0
    max_extension_atr: float = 3.0   # refuse directional entries beyond this
    neutral_band: float = 18.0
    # Minimum probability of profit (at realised vol) for ANY suggested trade.
    # This exists to stop the budget-fitting search from degenerating into
    # lottery tickets: when a structure is too expensive, the scanner narrows
    # it, and without a floor it will keep narrowing until it lands on a
    # 0.05-delta spread that costs almost nothing and wins almost never.
    # A 13:1 payoff at a 5% win rate is not a good trade, it is a raffle.
    # Small accounts hit this constantly, because cheap is all they can reach.
    min_pop: float = 0.25


@dataclass
class EarningsPolicy:
    """
    Earnings are the single biggest source of blown-up option trades, so the
    default is to avoid holding short premium through a report. An earnings
    event inside the expiry inflates IV (making the trade look attractive) and
    then gaps the underlying (making it a loser). Set
    `allow_through_earnings: true` only if you are deliberately trading events.
    """
    allow_through_earnings: bool = False
    block_credit_within_days: int = 3
    warn_window_days: int = 10
    score_penalty: float = 18.0


@dataclass
class ScoringWeights:
    regime_fit: float = 22.0
    premium_edge: float = 20.0
    expected_value: float = 18.0
    probability: float = 15.0
    liquidity: float = 15.0
    structure: float = 10.0


@dataclass
class OutputConfig:
    directory: str = "./output"
    max_ideas: int = 15
    max_per_ticker: int = 2
    min_score: float = 50.0
    html: bool = True
    csv: bool = True
    json: bool = True
    open_browser: bool = False


@dataclass
class Config:
    account: AccountConfig = field(default_factory=AccountConfig)
    liquidity: LiquidityRules = field(default_factory=LiquidityRules)
    dte: DTEWindows = field(default_factory=DTEWindows)
    deltas: DeltaTargets = field(default_factory=DeltaTargets)
    width: SpreadWidth = field(default_factory=SpreadWidth)
    filters: UniverseFilters = field(default_factory=UniverseFilters)
    regime: RegimeThresholds = field(default_factory=RegimeThresholds)
    earnings: EarningsPolicy = field(default_factory=EarningsPolicy)
    scoring: ScoringWeights = field(default_factory=ScoringWeights)
    output: OutputConfig = field(default_factory=OutputConfig)

    provider: str = "yahoo"
    benchmark: str = "SPY"
    tickers: list = field(default_factory=list)   # empty -> built-in universe
    max_workers: int = 6
    use_cache: bool = True
    strategies_enabled: list = field(default_factory=lambda: [
        "long_call", "long_put", "call_debit_spread", "put_debit_spread",
        "put_credit_spread", "call_credit_spread", "iron_condor",
        "cash_secured_put", "covered_call",
    ])

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        cfg = cls()
        for k, v in (d or {}).items():
            if not hasattr(cfg, k):
                continue
            cur = getattr(cfg, k)
            if is_dataclass(cur) and isinstance(v, dict):
                valid = {f.name for f in fields(cur)}
                for kk, vv in v.items():
                    if kk in valid:
                        if isinstance(getattr(cur, kk), tuple) and isinstance(vv, list):
                            vv = tuple(vv)
                        setattr(cur, kk, vv)
            else:
                setattr(cfg, k, v)
        return cfg

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        if not path:
            return cls()
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        text = p.read_text()
        if p.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml
                data = yaml.safe_load(text)
            except ImportError as exc:
                raise ImportError("YAML config needs PyYAML: pip install pyyaml") from exc
        else:
            data = json.loads(text)
        return cls.from_dict(data or {})

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        d = self.to_dict()
        if p.suffix.lower() in (".yaml", ".yml"):
            import yaml
            p.write_text(yaml.safe_dump(d, sort_keys=False, default_flow_style=False))
        else:
            p.write_text(json.dumps(d, indent=2, default=str))
