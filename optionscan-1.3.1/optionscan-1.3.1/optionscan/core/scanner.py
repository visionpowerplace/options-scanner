"""
Scan orchestration: universe in, ranked trade ideas out.

Error isolation is a design requirement, not a nicety. One delisted ticker, one
malformed chain or one network timeout must never kill a scan of ninety names,
so every underlying is processed independently and its failure is recorded and
reported rather than raised.
"""
from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

from ..config import Config
from ..data.base import DataProvider
from ..data.cache import IVHistoryStore
from . import indicators as ind
from . import playbook
from .chain import ChainView
from .regime import Regime, classify_trend, classify_vol
from .risk import AccountConfig, PositionSize, TradePlan, build_plan, portfolio_heat, size_position
from .scoring import ScoreCard, score_trade
from .strategies import (CALL_CREDIT, CALL_DEBIT, CASH_SECURED_PUT, COVERED_CALL,
                         IRON_CONDOR, LONG_CALL, LONG_PUT, PUT_CREDIT, PUT_DEBIT,
                         Trade, build_cash_secured_put, build_covered_call,
                         build_credit_spread, build_iron_condor,
                         build_long_option, build_vertical)


def _width_ladder(spot: float, step: float, cfg: Config, max_steps: int = 10) -> list:
    """Candidate vertical widths, widest first, bounded by the strike ladder."""
    cap = cfg.width.resolve(spot, step)
    n = max(1, min(max_steps, int(round(cap / max(step, 0.01)))))
    return [round(step * k, 4) for k in range(n, 0, -1)]


def _fit_to_budget(builder, knobs, budget: float, min_pop: float = 0.0):
    """
    Try structures from most to least expensive and return the first that is
    BOTH affordable and not a lottery ticket.

    This is what lets one config work for a $5k account and a $500k one. A
    $25-wide SPY credit spread risks ~$1,800 per contract: correct for a large
    account, untradeable on a small one. Rather than emitting an idea the user
    cannot size, the scanner narrows the structure until it fits.

    But narrowing has a floor. Keep shrinking a spread and it gets cheaper
    because it moves further out of the money -- eventually costing almost
    nothing and winning almost never. So affordability alone is not the test,
    and the two ways of failing are reported separately, because they call for
    opposite responses from the user:

        "too_expensive" -> the account cannot carry this structure
        "low_pop"       -> the account CAN carry it, but only in a version whose
                           odds are not worth taking

    Returns (trade, status) with status "ok", "too_expensive" or "low_pop".
    """
    tightest = None
    best_affordable = None
    for k in knobs:
        try:
            t = builder(k)
        except Exception:
            continue
        if t is None:
            continue
        tightest = t                     # knobs run expensive -> cheap
        if t.max_loss <= budget:
            if t.pop_at_rv >= min_pop:
                return t, "ok"
            if best_affordable is None or t.pop_at_rv > best_affordable.pop_at_rv:
                best_affordable = t
    if best_affordable is not None:
        return best_affordable, "low_pop"
    return tightest, "too_expensive"


@dataclass
class Idea:
    rank: int
    trade: Trade
    regime: Regime
    score: ScoreCard
    size: PositionSize
    plan: TradePlan

    def as_dict(self) -> dict:
        return {
            "rank": self.rank,
            "ticker": self.trade.ticker,
            "strategy": self.trade.strategy,
            "strategy_name": self.trade.pretty,
            "score": self.score.total,
            "spot": self.trade.spot,
            "expiry": self.trade.expiry,
            "dte": self.trade.dte,
            "order": self.trade.order_ticket(self.size.contracts),
            "net_cash": self.trade.net_cash,
            "entry_limit": self.plan.entry_limit,
            "entry_type": self.plan.entry_type,
            "max_profit": self.trade.max_profit,
            "max_loss": self.trade.max_loss,
            "breakevens": self.trade.breakevens,
            "pop": self.trade.pop,
            "pop_at_rv": self.trade.pop_at_rv,
            "required_pop": self.score.required_pop,
            "pop_edge": self.score.pop_edge,
            "p_touch_short": self.trade.p_touch_short,
            "ev_at_rv": self.trade.ev_at_rv,
            "ev_at_iv": self.trade.ev_at_iv,
            "contracts": self.size.contracts,
            "risk_dollars": self.size.risk_dollars,
            "capital_required": self.size.capital_required,
            "net_delta": self.trade.net_delta,
            "net_theta": self.trade.net_theta,
            "net_vega": self.trade.net_vega,
            "iv_rank": self.regime.vol.iv_rank,
            "iv_rank_source": self.regime.vol.iv_rank_source,
            "iv_rv_ratio": self.regime.vol.iv_rv_ratio,
            "atm_iv": self.regime.vol.atm_iv,
            "trend_label": self.regime.trend.label,
            "trend_score": self.regime.trend.score,
            "adx": self.regime.trend.adx,
            "earnings_in_days": self.regime.earnings_in_days,
            "liq_score": self.trade.liq_score,
            "profit_target_price": self.plan.profit_target_price,
            "stop_price": self.plan.stop_price,
            "time_stop_dte": self.plan.time_stop_dte,
            "components": self.score.components,
            "penalties": self.score.penalties,
            "reasons": self.score.reasons,
            "warnings": self.score.warnings + self.size.warnings,
        }


@dataclass
class ScanResult:
    as_of: datetime
    provider: str
    ideas: list = field(default_factory=list)
    diagnostics: list = field(default_factory=list)   # per-ticker outcome
    errors: list = field(default_factory=list)
    advisories: list = field(default_factory=list)
    data_coverage: float = 1.0      # share of the universe that returned usable data
    heat: dict = field(default_factory=dict)
    config: Config | None = None
    universe_size: int = 0
    scanned: int = 0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([i.as_dict() for i in self.ideas])

    def diagnostics_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.diagnostics)


# --------------------------------------------------------------------------
def _pick_expiry(expiries: list, window: tuple) -> date | None:
    """Expiry whose DTE sits in `window`, closest to the window's midpoint."""
    lo, hi = window
    today = date.today()
    cands = [(abs((e - today).days - (lo + hi) / 2.0), e)
             for e in expiries if lo <= (e - today).days <= hi]
    return min(cands)[1] if cands else None


def _window_for(strategy: str, cfg: Config) -> tuple:
    if strategy in (PUT_CREDIT, CALL_CREDIT, CASH_SECURED_PUT, COVERED_CALL):
        return cfg.dte.credit
    if strategy == IRON_CONDOR:
        return cfg.dte.condor
    if strategy in (LONG_CALL, LONG_PUT):
        return cfg.dte.long
    return cfg.dte.debit


class Scanner:
    def __init__(self, provider: DataProvider, cfg: Config | None = None,
                 iv_store: IVHistoryStore | None = None):
        self.provider = provider
        self.cfg = cfg or Config()
        self.iv_store = iv_store if iv_store is not None else IVHistoryStore()
        self._bench: pd.Series | None = None
        self._r = None

    # ------------------------------------------------------------------
    def _rate(self) -> float:
        if self._r is None:
            try:
                self._r = float(self.provider.get_risk_free_rate())
            except Exception:
                self._r = 0.04
        return self._r

    def _benchmark(self) -> pd.Series | None:
        if self._bench is None:
            try:
                h = self.provider.get_history(self.cfg.benchmark, 400)
                self._bench = h["close"] if h is not None and not h.empty else pd.Series(dtype=float)
            except Exception:
                self._bench = pd.Series(dtype=float)
        return self._bench if len(self._bench) else None

    # ------------------------------------------------------------------
    def scan_ticker(self, ticker: str) -> dict:
        """
        Process one underlying. Returns a diagnostic dict always, with an
        'ideas' key holding zero or more Idea objects. Never raises.
        """
        diag = {"ticker": ticker, "status": "ok", "ideas": [], "reason": "",
                "trend": "", "iv_rank": None, "candidates": [], "built": 0,
                "spot": None}
        try:
            hist = self.provider.get_history(ticker, 400)
            if hist is None or len(hist) == 0:
                # No bars at all almost always means the feed failed, not that
                # the symbol is uninteresting. Kept distinct from "no edge" so a
                # broken data source can never masquerade as a quiet market.
                diag.update(status="no_data",
                            reason="no price history returned -- data source "
                                   "unreachable or symbol not recognised")
                return diag
            if len(hist) < self.cfg.filters.min_history_days:
                diag.update(status="skipped",
                            reason=f"only {len(hist)} bars of history "
                                   f"(need {self.cfg.filters.min_history_days})")
                return diag

            spot = float(self.provider.get_spot(ticker)) or float(hist["close"].iloc[-1])
            diag["spot"] = round(spot, 2)
            f = self.cfg.filters
            if not (f.min_price <= spot <= f.max_price):
                diag.update(status="skipped",
                            reason=f"price ${spot:,.2f} outside "
                                   f"${f.min_price:g}-${f.max_price:g}")
                return diag

            adv = float((hist["close"] * hist["volume"]).tail(20).mean())
            if adv < f.min_avg_dollar_volume:
                diag.update(status="skipped",
                            reason=f"20-day dollar volume ${adv/1e6:,.1f}M below "
                                   f"${f.min_avg_dollar_volume/1e6:,.0f}M minimum")
                return diag

            expiries = self.provider.get_expirations(ticker)
            if not expiries:
                diag.update(status="no_data",
                            reason="no option expirations returned -- feed unreachable "
                                   "or no listed options")
                return diag

            try:
                q = float(self.provider.get_dividend_yield(ticker))
            except Exception:
                q = 0.0
            try:
                earn = self.provider.get_earnings_date(ticker)
            except Exception:
                earn = None
            earn_days = (earn - date.today()).days if earn else None

            r = self._rate()
            trend = classify_trend(hist, self._benchmark(), self.cfg)

            # Primary chain (the credit window) anchors the vol read.
            prim_exp = (_pick_expiry(expiries, self.cfg.dte.credit)
                        or _pick_expiry(expiries, self.cfg.dte.debit)
                        or expiries[min(3, len(expiries) - 1)])
            chains: dict[date, ChainView] = {}

            def view(exp: date) -> ChainView:
                if exp not in chains:
                    chains[exp] = ChainView(
                        ticker, spot, exp, (exp - date.today()).days,
                        self.provider.get_chain(ticker, exp), r, q, self.cfg.liquidity)
                return chains[exp]

            prim = view(prim_exp)
            if prim.empty:
                diag.update(status="skipped",
                            reason=f"no contracts passed the liquidity filter "
                                   f"({self.cfg.liquidity.describe()})")
                return diag

            atm_iv = prim.atm_iv()
            front = view(expiries[0]) if expiries[0] != prim_exp else prim
            back_exp = _pick_expiry(expiries, (80, 250)) or expiries[-1]
            back = view(back_exp) if back_exp != prim_exp else prim
            vol = classify_vol(hist, atm_iv, self.iv_store, ticker,
                               front.atm_iv() if not front.empty else None,
                               back.atm_iv() if not back.empty else None, self.cfg)

            rv_blend = ind.realized_vol(hist["close"], 20).iloc[-1]
            if pd.isna(rv_blend):
                rv_blend = atm_iv
            atr_v = float(ind.atr(hist, 14).iloc[-1])

            regime = Regime(ticker=ticker, spot=spot, trend=trend, vol=vol,
                            earnings_in_days=earn_days, avg_dollar_volume=adv,
                            gap_risk=ind.gap_risk(hist), atr=atr_v)
            diag["trend"] = trend.label
            diag["iv_rank"] = vol.iv_rank

            # Record today's observation so IV rank becomes real over time.
            try:
                self.iv_store.record(ticker, atm_iv, float(rv_blend),
                                     float(ind.realized_vol(hist["close"], 60).iloc[-1]), spot)
            except Exception:
                pass

            wanted = playbook.candidates(regime, self.cfg)
            diag["candidates"] = list(wanted)
            if not wanted:
                diag.update(status="no_edge", reason=playbook.no_edge_reason(regime, self.cfg))
                return diag

            acct = self.cfg.account
            slip, comm = acct.fill_slippage, acct.commission_per_contract
            ideas: list[Idea] = []

            for strat in wanted:
                exp = _pick_expiry(expiries, _window_for(strat, self.cfg))
                if exp is None:
                    continue
                cv = view(exp)
                if cv.empty:
                    continue
                step = cv.strike_step()
                widths = _width_ladder(spot, step, self.cfg)
                budget = acct.risk_dollars
                min_pop = self.cfg.regime.min_pop
                d = self.cfg.deltas
                t: Trade | None = None
                fit_status = "ok"
                try:
                    if strat == LONG_CALL or strat == LONG_PUT:
                        kind = "call" if strat == LONG_CALL else "put"
                        t, fit_status = _fit_to_budget(
                            lambda dl: build_long_option(cv, kind, dl, slip, comm, rv_blend),
                            [d.long_option, 0.45, 0.38, 0.30], budget, min_pop)
                    elif strat in (CALL_DEBIT, PUT_DEBIT):
                        kind = "call" if strat == CALL_DEBIT else "put"
                        # Cheaper variants: move the long leg further OTM, which
                        # lowers the debit (and the max loss) at the cost of delta.
                        pairs = [(d.debit_long, d.debit_short), (0.52, 0.28),
                                 (0.45, 0.25), (0.38, 0.20), (0.32, 0.16)]
                        t, fit_status = _fit_to_budget(
                            lambda pr: build_vertical(cv, kind, pr[0], pr[1],
                                                      slip, comm, rv_blend),
                            pairs, budget, min_pop)
                    elif strat in (PUT_CREDIT, CALL_CREDIT):
                        kind = "put" if strat == PUT_CREDIT else "call"
                        t, fit_status = _fit_to_budget(
                            lambda w: build_credit_spread(cv, kind, d.credit_short, w,
                                                          slip, comm, rv_blend),
                            widths, budget, min_pop)
                    elif strat == IRON_CONDOR:
                        t, fit_status = _fit_to_budget(
                            lambda w: build_iron_condor(cv, d.condor_short, w,
                                                        slip, comm, rv_blend),
                            widths, budget, min_pop)
                    elif strat == CASH_SECURED_PUT:
                        t = build_cash_secured_put(cv, d.csp_short, slip, comm, rv_blend)
                    elif strat == COVERED_CALL:
                        t = build_covered_call(cv, d.cc_short, slip, comm, rv_blend)
                except Exception as exc:
                    diag.setdefault("build_errors", []).append(f"{strat}: {exc}")
                    continue
                if t is None:
                    continue
                if fit_status == "too_expensive":
                    diag.setdefault("too_large", []).append(
                        f"{strat}: tightest version still risks ${t.max_loss:,.0f}/contract "
                        f"vs ${budget:,.0f} budget")
                    continue
                if fit_status == "low_pop":
                    diag.setdefault("lottery", []).append(
                        f"{strat}: only affordable at ${t.max_loss:,.0f}/contract with a "
                        f"{t.pop_at_rv:.0%} chance of profit (floor {min_pop:.0%}) -- "
                        f"cheap because it rarely pays, not because it is good value")
                    continue

                # Hard earnings block for short premium, before scoring.
                if (not self.cfg.earnings.allow_through_earnings
                        and t.strategy in (PUT_CREDIT, CALL_CREDIT, IRON_CONDOR,
                                           CASH_SECURED_PUT)
                        and earn_days is not None
                        and 0 <= earn_days <= t.dte + self.cfg.earnings.block_credit_within_days):
                    diag.setdefault("blocked", []).append(
                        f"{strat}: earnings in {earn_days}d inside {t.dte}-DTE expiry")
                    continue

                if t.pop_at_rv < self.cfg.regime.min_pop:
                    diag.setdefault("lottery", []).append(
                        f"{strat}: only {t.pop_at_rv:.0%} chance of profit "
                        f"(floor is {self.cfg.regime.min_pop:.0%}) -- cheap because it "
                        f"rarely pays, not because it is good value")
                    continue

                # Per-contract price ceiling (small-account constraint).
                cap = self.cfg.filters.max_debit_per_share
                if cap is not None and t.net_price > cap:
                    diag.setdefault("too_expensive", []).append(
                        f"{strat}: costs ${t.net_price:.2f}/share "
                        f"(${abs(t.net_cash):,.0f}/contract), over your "
                        f"${cap:.2f} ceiling")
                    continue

                diag["built"] += 1
                sc = score_trade(t, regime, self.cfg)
                if sc.total < self.cfg.output.min_score:
                    continue
                sz = size_position(t, acct, atm_iv=atm_iv)
                if sz.contracts < 1:
                    diag.setdefault("unsized", []).append(
                        f"{strat}: {sz.warnings[0] if sz.warnings else 'zero contracts'}")
                    continue
                pl = build_plan(t, regime, acct)
                ideas.append(Idea(0, t, regime, sc, sz, pl))

            ideas.sort(key=lambda i: i.score.total, reverse=True)
            diag["ideas"] = ideas[: self.cfg.output.max_per_ticker]
            if not diag["ideas"]:
                diag.update(status="filtered",
                            reason=f"built {diag['built']} structure(s); none cleared "
                                   f"score {self.cfg.output.min_score:g} / sizing limits")
            return diag

        except Exception as exc:
            diag.update(status="error", reason=f"{type(exc).__name__}: {exc}",
                        traceback=traceback.format_exc(limit=3))
            return diag

    # ------------------------------------------------------------------
    def scan(self, tickers: list, progress=None) -> ScanResult:
        res = ScanResult(as_of=datetime.now(), provider=self.provider.name,
                         config=self.cfg, universe_size=len(tickers))
        all_ideas: list[Idea] = []

        workers = max(1, int(self.cfg.max_workers))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(self.scan_ticker, t): t for t in tickers}
            done = 0
            for fut in as_completed(futs):
                t = futs[fut]
                done += 1
                try:
                    diag = fut.result()
                except Exception as exc:
                    diag = {"ticker": t, "status": "error",
                            "reason": f"{type(exc).__name__}: {exc}", "ideas": []}
                all_ideas.extend(diag.pop("ideas", []) or [])
                if diag["status"] == "error":
                    res.errors.append({"ticker": t, "reason": diag.get("reason", "")})
                res.diagnostics.append(diag)
                res.scanned += 1
                if progress:
                    progress(done, len(tickers), t, diag["status"])

        # A scan built on missing data is not a scan that found nothing. Saying
        # "no trades today" when the feed never answered would be the single most
        # misleading thing this tool could do, especially running unattended.
        no_data = sum(1 for d in res.diagnostics if d["status"] == "no_data")
        err = len(res.errors)
        broken = no_data + err
        if broken:
            share = broken / max(len(tickers), 1)
            res.data_coverage = round(1.0 - share, 3)
            msg = (f"DATA PROBLEM: {broken} of {len(tickers)} symbols returned no usable "
                   f"data ({share:.0%} of the universe). ")
            if share >= 0.5:
                msg += ("This scan is NOT complete and its results mean little. An empty "
                        "or short idea list below reflects missing data, not a quiet "
                        "market. Check your network and data source, then re-run.")
            else:
                msg += ("The remaining symbols scanned normally, but treat the idea list "
                        "as partial.")
            res.advisories.insert(0, msg)

        # Surface account-size constraints explicitly. "No ideas today" and
        # "your account is too small for these underlyings" are very different
        # messages, and conflating them would leave the user guessing.
        blocked = [d for d in res.diagnostics if d.get("too_large")]
        lottery = [d for d in res.diagnostics if d.get("lottery")]
        if blocked:
            cheapest = min(
                (float(x.split("risks $")[1].split("/")[0].replace(",", ""))
                 for d in blocked for x in d["too_large"] if "risks $" in x),
                default=0.0)
            need_pct = (cheapest / self.cfg.account.account_size * 100.0) if cheapest else 0.0
            res.advisories.append(
                f"{len(blocked)} underlying(s) produced valid structures that were too "
                f"large for your ${self.cfg.account.account_size:,.0f} account at "
                f"{self.cfg.account.risk_per_trade_pct:g}% risk per trade "
                f"(${self.cfg.account.risk_dollars:,.0f}). The cheapest rejected structure "
                f"risked ${cheapest:,.0f}/contract, which would need "
                f"risk_per_trade_pct >= {need_pct:.1f}%. Options trade in whole contracts, "
                f"so small accounts are genuinely limited to lower-priced underlyings and "
                f"narrow spreads -- that is a real constraint, not a setting to force.")

        if lottery:
            res.advisories.append(
                f"{len(lottery)} underlying(s) had structures your account could afford, "
                f"but only in versions with less than a "
                f"{self.cfg.regime.min_pop:.0%} chance of profit, so they were rejected. "
                f"This is the central difficulty of a small account: what is cheap enough "
                f"to buy is usually cheap BECAUSE it rarely pays. Lowering regime.min_pop "
                f"would surface them, which is the one setting you should not relax.")

        all_ideas.sort(key=lambda i: i.score.total, reverse=True)
        kept = all_ideas[: self.cfg.output.max_ideas]
        for n, i in enumerate(kept, 1):
            i.rank = n
        res.ideas = kept
        res.heat = portfolio_heat([i.size for i in kept], self.cfg.account)
        res.diagnostics.sort(key=lambda d: d["ticker"])
        return res
