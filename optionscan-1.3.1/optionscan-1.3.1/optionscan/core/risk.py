"""
Position sizing and trade management.

This module is the part that actually protects the account, and it is
deliberately the most conservative thing in the package.

Three rules it enforces:

1. SIZE FROM RISK, NOT FROM PREMIUM. Contract count comes from how many
   dollars you are willing to lose, divided by the real per-contract max loss
   (commissions and slippage included), never from how much premium looks
   collectible. Premium-selling blows up accounts precisely because a 90%
   win rate feels like a reason to size up.

2. SIZE UNDEFINED-RISK TRADES BY CAPITAL, NOT BY MAX LOSS. A cash-secured put's
   "max loss" is the strike to zero, which would size every position to zero
   contracts. Those are sized by collateral and an assignment-aware stress
   loss instead.

3. EVERY TRADE GETS AN EXIT BEFORE IT GETS AN ENTRY. Profit target, stop, time
   stop, and the underlying price that invalidates the thesis -- all computed
   up front, because nobody decides these well while losing money.

The management parameters follow the mainstream, research-backed conventions
for retail options: take credit trades off at ~50% of max credit, close before
gamma risk accelerates in the final three weeks, and cut long premium at a
fixed fraction of the debit. They are all configurable; none of them are
magic, and none of them turn a negative-expectancy trade positive.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

from .strategies import (CALL_CREDIT, CALL_DEBIT, CASH_SECURED_PUT, COVERED_CALL,
                         CREDIT_STRATEGIES, IRON_CONDOR, LONG_CALL, LONG_PUT, MULT,
                         PUT_CREDIT, PUT_DEBIT, Trade)

# A spread's net delta is small by design, so it is a poor guide to what the
# trade is actually betting on. A bear call credit spread can sit at delta
# -0.03 and still be a wholly bearish position: it loses if the stock rallies.
# Directional intent therefore comes from the structure itself.
_BULLISH = {LONG_CALL, CALL_DEBIT, PUT_CREDIT, CASH_SECURED_PUT}
_BEARISH = {LONG_PUT, PUT_DEBIT, CALL_CREDIT}
_NEUTRAL = {IRON_CONDOR, COVERED_CALL}


def trade_direction(trade: Trade) -> int:
    """+1 bullish, -1 bearish, 0 neutral -- by structure, with delta as a fallback."""
    if trade.strategy in _BULLISH:
        return 1
    if trade.strategy in _BEARISH:
        return -1
    if trade.strategy in _NEUTRAL:
        return 0
    return 1 if trade.net_delta > 0.05 else (-1 if trade.net_delta < -0.05 else 0)


@dataclass
class AccountConfig:
    account_size: float = 25_000.0
    risk_per_trade_pct: float = 1.0        # % of account risked on one trade
    max_portfolio_heat_pct: float = 6.0    # total % at risk across all open ideas
    max_capital_per_trade_pct: float = 20.0  # cap on buying power for one position
    max_contracts: int = 50
    commission_per_contract: float = 0.65
    fill_slippage: float = 0.40            # fraction of the bid/ask spread given up

    @property
    def risk_dollars(self) -> float:
        return self.account_size * self.risk_per_trade_pct / 100.0

    @property
    def heat_dollars(self) -> float:
        return self.account_size * self.max_portfolio_heat_pct / 100.0

    @property
    def capital_cap(self) -> float:
        return self.account_size * self.max_capital_per_trade_pct / 100.0


@dataclass
class PositionSize:
    contracts: int
    risk_dollars: float          # what you actually stand to lose
    capital_required: float      # buying power consumed
    pct_of_account_risked: float
    pct_of_account_capital: float
    limiting_factor: str
    stress_loss: float = 0.0     # loss in the adverse-but-plausible scenario
    warnings: list = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


@dataclass
class TradePlan:
    # Entry
    entry_limit: float            # the net price to work the order at (per share)
    entry_limit_worst: float      # do not pay/accept beyond this
    entry_type: str               # 'debit' | 'credit'
    entry_conditions: list = field(default_factory=list)

    # Exits
    profit_target_price: float = 0.0     # net price to close at
    profit_target_pnl: float = 0.0       # dollars per contract
    profit_target_label: str = ""
    stop_price: float = 0.0
    stop_pnl: float = 0.0
    stop_label: str = ""
    time_stop_dte: int = 21
    invalidation_level: float | None = None
    invalidation_label: str = ""
    exit_rules: list = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------
def _round_tick(x: float) -> float:
    """US option prices trade in $0.01 below $3 and $0.05 above (near enough)."""
    tick = 0.01 if abs(x) < 3.0 else 0.05
    return round(round(x / tick) * tick, 2)


def size_position(trade: Trade, acct: AccountConfig,
                  atm_iv: float | None = None, k_sigma: float = 2.0,
                  min_move_pct: float = 0.10, max_move_pct: float = 0.60) -> PositionSize:
    """
    Contracts to trade, and why that number.

    The adverse scenario used for sizing is scaled to the stock's OWN implied
    volatility, not a flat percentage. This matters enormously and a fixed
    percentage gets it dangerously wrong:

      A flat 12% adverse move barely reaches a 0.25-delta put on a 50%-vol
      stock over 30 days, so the modelled "stress loss" collapses towards zero
      and the sizer will happily recommend unlimited contracts on exactly the
      most volatile names in the universe. That is not a conservative error.

    So the stress move is `k_sigma` standard deviations of the option's own
    tenor -- 2 sigma by default -- floored at 10% and capped at 60%. On a 25%-vol
    name over 32 days that is ~15%; on a 51%-vol name it is ~30%, which is the
    whole point.

    Defined-risk structures are sized by max loss (already bounded). Cash-secured
    puts and covered calls are sized by this stress loss, because their
    theoretical max loss is the stock going to zero -- a number that would size
    every such position to zero contracts and is not the realistic risk anyway.
    """
    warnings: list[str] = []
    undefined = trade.strategy in (CASH_SECURED_PUT, COVERED_CALL)

    iv = atm_iv if (atm_iv and atm_iv > 0) else (
        sum(l.iv for l in trade.legs) / max(len(trade.legs), 1))
    horizon = max(trade.dte, 1) / 365.0
    sigma_move = k_sigma * iv * (horizon ** 0.5)
    stress_move_pct = max(min_move_pct, min(sigma_move, max_move_pct))

    # Test BOTH directions and take the worse: a structure's painful side is not
    # always the one its label suggests, and the bigger number is the honest one.
    down = trade.spot * (1.0 - stress_move_pct)
    up = trade.spot * (1.0 + stress_move_pct)
    stress = max(abs(min(0.0, trade.pnl(down))), abs(min(0.0, trade.pnl(up))))

    if undefined:
        sizing_loss = max(stress, 1.0)
        warnings.append(
            f"Undefined-risk structure: sized against a {stress_move_pct:.0%} adverse "
            f"move (= {k_sigma:g} standard deviations at this stock's {iv:.0%} implied "
            f"vol over {trade.dte} days), which is ${stress:,.0f}/contract -- NOT the "
            f"theoretical max loss of ${trade.max_loss:,.0f}. A gap beyond that move, "
            f"or assignment, still costs more than this number.")
    else:
        sizing_loss = max(trade.max_loss, 1.0)

    by_risk = int(acct.risk_dollars // sizing_loss)
    capital_each = max(trade.collateral, 1.0)
    by_capital = int(acct.capital_cap // capital_each)

    contracts = max(0, min(by_risk, by_capital, acct.max_contracts))
    limiting = ("risk limit" if by_risk <= by_capital else "capital limit")
    if contracts == acct.max_contracts and acct.max_contracts <= min(by_risk, by_capital):
        limiting = "max_contracts cap"

    if contracts == 0:
        if by_capital == 0:
            warnings.append(
                f"Position needs ${capital_each:,.0f} of buying power per contract but "
                f"your per-trade capital cap is ${acct.capital_cap:,.0f}. Too large for "
                f"this account -- skip it or use a narrower/cheaper structure.")
        else:
            warnings.append(
                f"One contract risks ${sizing_loss:,.0f}, more than your "
                f"${acct.risk_dollars:,.0f} per-trade limit. Skip it.")

    risk_total = contracts * sizing_loss
    cap_total = contracts * capital_each
    return PositionSize(
        contracts=contracts,
        risk_dollars=round(risk_total, 2),
        capital_required=round(cap_total, 2),
        pct_of_account_risked=round(risk_total / acct.account_size * 100.0, 2),
        pct_of_account_capital=round(cap_total / acct.account_size * 100.0, 2),
        limiting_factor=limiting,
        stress_loss=round(contracts * stress, 2),
        warnings=warnings,
    )


def build_plan(trade: Trade, regime=None, acct: AccountConfig | None = None,
               credit_target: float = 0.50, credit_stop_mult: float = 2.0,
               debit_target: float = 0.60, debit_stop: float = 0.50,
               long_target: float = 0.80, long_stop: float = 0.45,
               time_stop_credit: int = 21, time_stop_debit: int = 14) -> TradePlan:
    """
    Build the full entry/exit plan. Everything is expressed as a NET PRICE for
    the spread, so it can be entered as a single limit order and the exits can
    be placed as GTC orders the moment the fill confirms.
    """
    acct = acct or AccountConfig()
    net = trade.net_price                       # per share; >0 debit, <0 credit
    is_credit = net < 0
    entry = _round_tick(abs(net))
    conditions: list[str] = []
    rules: list[str] = []

    # --- entry price discipline -----------------------------------------
    if is_credit:
        worst = _round_tick(abs(net) * 0.90)
        conditions.append(
            f"Work a LIMIT order to SELL the spread for ${entry:.2f} credit. "
            f"Do not accept less than ${worst:.2f}. Start at the mid and walk "
            f"down a penny at a time; never sell at the natural bid.")
    else:
        worst = _round_tick(abs(net) * 1.10)
        conditions.append(
            f"Work a LIMIT order to BUY the spread for ${entry:.2f} debit. "
            f"Do not pay more than ${worst:.2f}. Start at the mid and walk up a "
            f"penny at a time; never buy at the natural ask.")

    conditions.append(
        "Place the order between 10:00 and 15:30 ET. The first and last 30 "
        "minutes have the widest spreads and the worst fills.")

    # --- entry timing driven by the regime ------------------------------
    if regime is not None:
        tr = regime.trend
        if trade_direction(trade) > 0 and tr.ext_atr > 2.0:
            conditions.append(
                f"CAUTION: price is {tr.ext_atr:.1f} ATR above its 21-EMA. Prefer waiting "
                f"for a pullback toward {trade.spot - regime.atr:.2f}-{trade.spot:.2f} "
                f"rather than entering extended.")
        if trade_direction(trade) < 0 and tr.ext_atr < -2.0:
            conditions.append(
                f"CAUTION: price is {abs(tr.ext_atr):.1f} ATR below its 21-EMA -- "
                f"stretched to the downside. Prefer a bounce toward "
                f"{trade.spot:.2f}-{trade.spot + regime.atr:.2f} before entering.")
        if tr.adx < 18 and trade.strategy in (LONG_CALL, LONG_PUT):
            conditions.append(
                f"ADX is {tr.adx:.0f} (no trend). Long premium needs movement -- "
                f"consider skipping or halving size.")
        if regime.earnings_in_days is not None and 0 <= regime.earnings_in_days <= trade.dte:
            conditions.append(
                f"EARNINGS in {regime.earnings_in_days} days, BEFORE this expiry. "
                f"Either close ahead of the report or size as an event trade.")

    # --- exits by family -------------------------------------------------
    if trade.strategy == IRON_CONDOR or trade.strategy in CREDIT_STRATEGIES:
        credit = abs(net)
        tgt_price = _round_tick(credit * (1.0 - credit_target))
        stop_price = _round_tick(credit * (1.0 + credit_stop_mult))
        plan_target_pnl = round(credit * credit_target * MULT - trade.commission, 2)
        plan_stop_pnl = round(-(credit * credit_stop_mult * MULT) - trade.commission, 2)
        label_t = f"close at {credit_target:.0%} of max credit"
        label_s = f"close if the spread costs {1 + credit_stop_mult:.1f}x the credit received"
        rules += [
            f"PROFIT: place a GTC limit to BUY the spread back at ${tgt_price:.2f} "
            f"as soon as you are filled. That captures {credit_target:.0%} of max "
            f"credit (${plan_target_pnl:,.0f}/contract) and removes the trade's worst "
            f"risk-reward period.",
            f"STOP: close if the spread trades at ${stop_price:.2f} "
            f"(loss ~${abs(plan_stop_pnl):,.0f}/contract).",
            f"TIME: close or roll at {time_stop_credit} DTE regardless of P&L. "
            f"Gamma risk accelerates sharply in the last three weeks -- that is where "
            f"small winners turn into large losers.",
        ]
        if trade.short_strikes:
            near = min(trade.short_strikes, key=lambda k: abs(k - trade.spot))
            rules.append(
                f"BREACH: if the underlying closes beyond your short strike "
                f"({near:g}), the trade is wrong. Close it; do not roll it down "
                f"into a falling stock.")
        time_stop = time_stop_credit
    elif trade.strategy in (LONG_CALL, LONG_PUT):
        debit = abs(net)
        tgt_price = _round_tick(debit * (1.0 + long_target))
        stop_price = _round_tick(debit * (1.0 - long_stop))
        plan_target_pnl = round(debit * long_target * MULT - trade.commission, 2)
        plan_stop_pnl = round(-(debit * long_stop * MULT) - trade.commission, 2)
        label_t = f"+{long_target:.0%} on the premium"
        label_s = f"-{long_stop:.0%} on the premium"
        rules += [
            f"PROFIT: sell at ${tgt_price:.2f} (+{long_target:.0%}, "
            f"${plan_target_pnl:,.0f}/contract). Consider scaling: sell half there and "
            f"trail the rest.",
            f"STOP: sell at ${stop_price:.2f} (-{long_stop:.0%}, "
            f"-${abs(plan_stop_pnl):,.0f}/contract).",
            f"TIME: exit by {time_stop_debit} DTE even if flat. Theta decay on a long "
            f"option becomes brutal inside two weeks.",
        ]
        time_stop = time_stop_debit
    else:                       # debit verticals
        debit = abs(net)
        maxp = trade.max_profit if trade.max_profit else debit * MULT
        tgt_price = _round_tick(debit + (trade.width - debit) * debit_target)
        stop_price = _round_tick(debit * (1.0 - debit_stop))
        plan_target_pnl = round((tgt_price - debit) * MULT - trade.commission, 2)
        plan_stop_pnl = round(-(debit * debit_stop * MULT) - trade.commission, 2)
        label_t = f"{debit_target:.0%} of max profit"
        label_s = f"-{debit_stop:.0%} on the debit"
        rules += [
            f"PROFIT: GTC limit to SELL the spread at ${tgt_price:.2f} "
            f"({debit_target:.0%} of max, ${plan_target_pnl:,.0f}/contract). Waiting for "
            f"the last few cents of a vertical means holding the worst "
            f"risk-reward of the whole trade.",
            f"STOP: close at ${stop_price:.2f} "
            f"(-${abs(plan_stop_pnl):,.0f}/contract).",
            f"TIME: close by {time_stop_debit} DTE; a debit spread that has not worked "
            f"by then usually will not.",
        ]
        time_stop = time_stop_debit

    # --- thesis invalidation on the UNDERLYING ---------------------------
    inval, inval_lbl = None, ""
    direction = trade_direction(trade)
    if regime is not None and regime.atr > 0:
        if direction > 0:
            inval = round(trade.spot - 1.8 * regime.atr, 2)
            inval_lbl = (f"bullish thesis fails on a close below {inval:.2f} "
                         f"(1.8 ATR down)")
        elif direction < 0:
            inval = round(trade.spot + 1.8 * regime.atr, 2)
            inval_lbl = (f"bearish thesis fails on a close above {inval:.2f} "
                         f"(1.8 ATR up)")
        else:
            lo = round(trade.spot - 1.8 * regime.atr, 2)
            hi = round(trade.spot + 1.8 * regime.atr, 2)
            inval = hi
            inval_lbl = (f"neutral thesis fails outside {lo:.2f}-{hi:.2f} "
                         f"(+/-1.8 ATR)")
        rules.append("UNDERLYING: " + inval_lbl + ".")

    rules.append("Never add to a loser. If the thesis breaks, take the loss and "
                 "look at the next scan.")

    return TradePlan(
        entry_limit=entry, entry_limit_worst=worst,
        entry_type="credit" if is_credit else "debit",
        entry_conditions=conditions,
        profit_target_price=tgt_price, profit_target_pnl=plan_target_pnl,
        profit_target_label=label_t,
        stop_price=stop_price, stop_pnl=plan_stop_pnl, stop_label=label_s,
        time_stop_dte=time_stop,
        invalidation_level=inval, invalidation_label=inval_lbl,
        exit_rules=rules,
    )


def portfolio_heat(sizes: list[PositionSize], acct: AccountConfig) -> dict:
    """Aggregate exposure if every suggestion were taken -- the number that
    stops a scanner from quietly recommending twenty correlated bullish bets."""
    total_risk = sum(s.risk_dollars for s in sizes)
    total_cap = sum(s.capital_required for s in sizes)
    return {
        "total_risk": round(total_risk, 2),
        "total_capital": round(total_cap, 2),
        "pct_risked": round(total_risk / acct.account_size * 100.0, 2),
        "pct_capital": round(total_cap / acct.account_size * 100.0, 2),
        "heat_limit_pct": acct.max_portfolio_heat_pct,
        "over_limit": total_risk > acct.heat_dollars,
    }
