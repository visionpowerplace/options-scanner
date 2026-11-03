"""
The daily trade sheet.

Design goal: everything needed to place and manage the trade, with nothing
needed from anywhere else. Strikes, expiry, the limit price to work, the exact
contract count for the configured account, the GTC orders to place on fill, the
price that invalidates the thesis -- and, deliberately, the reasons AND the
risks side by side. A scanner that shows only the bull case is a salesman.

Plain text on purpose: readable in a terminal, pasteable into a journal,
emailable without rendering, and diffable day over day.
"""
from __future__ import annotations

from datetime import datetime

W = 86


def _rule(ch: str = "-") -> str:
    return ch * W


def _wrap(text: str, indent: int = 6, width: int = W) -> str:
    out, line = [], " " * indent
    for word in str(text).split():
        if len(line) + len(word) + 1 > width:
            out.append(line.rstrip())
            line = " " * indent + word + " "
        else:
            line += word + " "
    out.append(line.rstrip())
    return "\n".join(out)


def _money(x) -> str:
    if x is None:
        return "unlimited"
    return f"${x:,.0f}"


def format_idea(idea) -> str:
    t, r, s, z, p = idea.trade, idea.regime, idea.score, idea.size, idea.plan
    L: list[str] = []
    A = L.append

    A("")
    A(_rule("="))
    A(f"#{idea.rank}  {t.ticker}  |  {t.pretty.upper()}  |  SCORE {s.total:.0f}/100")
    A(_rule("="))

    # ---------------- why / risks ----------------
    if s.reasons:
        A("")
        A("  WHY THIS SETUP")
        for x in s.reasons:
            A(_wrap("- " + x, 4))
    risks = list(s.warnings) + list(z.warnings) + list(r.vol.notes) + list(t.notes)
    if risks:
        A("")
        A("  RISKS AND CAVEATS")
        for x in risks:
            A(_wrap("- " + x, 4))

    # ---------------- the contracts ----------------
    A("")
    A(f"  THE TRADE{' ' * 34}{t.ticker} ${t.spot:,.2f}   ATR ${r.atr:,.2f}")
    A(_rule())
    for leg in t.legs:
        act = "SELL" if leg.action == "sell" else "BUY "
        A(f"   {act} {z.contracts:>3} x {leg.expiry:%d-%b-%y} {leg.strike:>9,.2f} "
          f"{leg.kind.upper():<4} @ ${leg.mid:>7,.2f}")
        A(f"        delta {leg.delta:+.3f}   IV {leg.iv:6.1%}   OI {leg.open_interest:>7,}"
          f"   bid/ask ${leg.bid:,.2f}/${leg.ask:,.2f}")
    if t.strategy == "covered_call":
        A(f"   HOLD {z.contracts * 100:>3} shares of {t.ticker} @ ${t.spot:,.2f}")
    kind = "CREDIT" if t.net_cash > 0 else "DEBIT"
    A(f"   NET: {kind} ${abs(t.net_cash):,.2f} per contract"
      + (f"   |   spread width ${t.width:,.2f}" if t.width else ""))
    A(f"   Order: {t.order_ticket(z.contracts)}")

    # ---------------- entry ----------------
    A("")
    A("  ENTRY")
    A(_rule())
    for c in p.entry_conditions:
        A(_wrap("- " + c, 4))

    # ---------------- position ----------------
    A("")
    A("  POSITION SIZE")
    A(_rule())
    A(f"   Contracts          {z.contracts}          (limited by: {z.limiting_factor})")
    A(f"   Dollars at risk    ${z.risk_dollars:>10,.0f}   ({z.pct_of_account_risked:.2f}% of account)")
    A(f"   Buying power       ${z.capital_required:>10,.0f}   ({z.pct_of_account_capital:.2f}% of account)")
    A(f"   Max profit         {_money(t.max_profit * z.contracts if t.max_profit else None):>11}"
      f"   per position")
    A(f"   Max loss           ${t.max_loss * z.contracts:>10,.0f}   per position")
    A(f"   Breakeven(s)       {', '.join(f'${b:,.2f}' for b in t.breakevens) or 'n/a'}")

    # ---------------- odds ----------------
    A("")
    A("  THE ODDS")
    A(_rule())
    A(f"   Probability of profit      {t.pop:>6.1%}   measured at IMPLIED vol "
      f"({r.vol.atm_iv:.1%}) -- the conservative read")
    A(f"   ... at realised vol        {t.pop_at_rv:>6.1%}   measured at the vol the stock "
      f"has actually shown ({r.vol.rv20:.1%})")
    A(f"   Break-even win rate        {s.required_pop:>6.1%}   what this payoff needs just "
      f"to break even")
    A(f"   Probability edge           {s.pop_edge:>+6.1%}   realised-vol win rate minus "
      f"break-even win rate")
    if t.p_touch_short is not None:
        A(f"   Chance of touching short   {t.p_touch_short:>6.1%}   before expiry (expect it "
          f"to look worse than it ends)")
    A(f"   Avg win / avg loss         ${t.exp_win:,.0f} / ${t.exp_loss:,.0f} per contract")
    A(f"   Expected value             ${t.ev_at_rv:>+6,.0f}   per contract at realised vol "
      f"(${t.ev_at_iv:+,.0f} at implied)")

    # ---------------- greeks & vol context ----------------
    A("")
    A("  GREEKS (per contract)        MARKET CONTEXT")
    A(_rule())
    A(f"   delta  {t.net_delta:>+8.3f}             trend      {r.trend.label.replace('_', ' ')}"
      f" (score {r.trend.score:+.0f}, ADX {r.trend.adx:.0f})")
    A(f"   theta  ${t.net_theta:>+7,.2f}/day         IV rank    {r.vol.iv_rank:.0f} "
      f"({r.vol.iv_rank_source}, {r.vol.iv_rank_obs} obs)")
    A(f"   vega   ${t.net_vega:>+7,.2f}/vol pt      IV / RV    {r.vol.iv_rv_ratio:.2f}x")
    A(f"   gamma  {t.net_gamma:>+8.4f}             earnings   "
      f"{'in ' + str(r.earnings_in_days) + ' days' if r.earnings_in_days is not None else 'none scheduled'}"
      f"   |   DTE {t.dte}")

    # ---------------- exits ----------------
    A("")
    A("  EXIT PLAN -- place these as GTC orders the moment you are filled")
    A(_rule())
    for rule in p.exit_rules:
        A(_wrap("- " + rule, 4))

    # ---------------- score ----------------
    A("")
    A("  SCORE BREAKDOWN")
    A(_rule())
    c = s.components
    A(f"   regime fit {c['regime_fit']:>5.0f}  |  premium edge {c['premium_edge']:>5.0f}  |  "
      f"expected value {c['expected_value']:>5.0f}")
    A(f"   probability {c['probability']:>4.0f}  |  liquidity   {c['liquidity']:>7.0f}  |  "
      f"structure      {c['structure']:>5.0f}")
    if s.penalties:
        A(f"   penalties: " + ", ".join(f"{k} -{v:g}" for k, v in s.penalties.items()))
    return "\n".join(L)


def format_report(res, cfg, show_diagnostics: bool = True,
                  show_detail: bool = True) -> str:
    L: list[str] = []
    A = L.append
    acct = cfg.account

    A(_rule("="))
    A("  DAILY OPTIONS SCAN".center(W))
    A(f"  {res.as_of:%A %d %B %Y, %H:%M}".center(W))
    A(_rule("="))
    A(f"  Data source      {res.provider}"
      + ("   *** SIMULATED DATA -- NOT A REAL MARKET ***"
         if res.provider == "synthetic" else "   (quotes are delayed; re-check live before ordering)"))
    A(f"  Universe         {res.universe_size} symbols scanned")
    A(f"  Account          ${acct.account_size:,.0f} at {acct.risk_per_trade_pct:g}% risk/trade "
      f"(${acct.risk_dollars:,.0f} max loss per position)")
    A(f"  Ideas found      {len(res.ideas)}"
      f"   (min score {cfg.output.min_score:g}, max {cfg.output.max_ideas})")
    if res.heat:
        h = res.heat
        A(f"  If you took all   risk ${h['total_risk']:,.0f} ({h['pct_risked']:.1f}% of account)"
          f"   capital ${h['total_capital']:,.0f} ({h['pct_capital']:.1f}%)"
          + ("   *** OVER YOUR HEAT LIMIT ***" if h["over_limit"] else ""))

    # ------------------------------------------------ summary table
    if res.ideas:
        A("")
        A(_rule("="))
        A("  RANKED SUMMARY")
        A(_rule("="))
        A(f"  {'#':>2} {'SYM':<6}{'STRATEGY':<24}{'SC':>3} {'DTE':>4} {'POP':>5} {'EDGE':>6} "
          f"{'CTR':>4} {'RISK$':>8} {'MAXP$':>8} {'EV$':>6}")
        A(f"  {_rule('-')[:W-2]}")
        for i in res.ideas:
            t = i.trade
            mp = ("unlim" if t.max_profit is None
                  else f"{t.max_profit * i.size.contracts:,.0f}")
            A(f"  {i.rank:>2} {t.ticker:<6}{t.pretty:<24}{i.score.total:>3.0f} {t.dte:>4} "
              f"{t.pop:>5.0%} {i.score.pop_edge:>+6.0%} {i.size.contracts:>4} "
              f"{i.size.risk_dollars:>8,.0f} {mp:>8} "
              f"{t.ev_at_rv * i.size.contracts:>+6,.0f}")
    else:
        A("")
        A(_rule("="))
        if getattr(res, "data_coverage", 1.0) < 0.5:
            A("  NO TRADES -- BUT THE DATA DID NOT ARRIVE")
            A(_rule("="))
            A(_wrap("Most of the universe returned no usable data, so this scan did not "
                    "really run. Do NOT read the empty list as 'the market offered "
                    "nothing today' -- it means the data source could not be reached. "
                    "Check your connection and re-run.", 2))
        else:
            A("  NO TRADES TODAY")
            A(_rule("="))
            A(_wrap("Nothing in the scanned universe met the configured thresholds. This is "
                    "a normal and correct outcome -- most days most underlyings offer no "
                    "edge, and forcing a trade on those days is how edge gets given back. "
                    "The per-symbol reasons are listed below.", 2))

    for a in res.advisories:
        A("")
        A("  ADVISORY")
        A(_wrap(a, 4))

    # ------------------------------------------------ detail
    if show_detail:
        for i in res.ideas:
            A(format_idea(i))

    # ------------------------------------------------ diagnostics
    if show_diagnostics:
        A("")
        A(_rule("="))
        A("  WHY EVERY OTHER SYMBOL WAS PASSED OVER")
        A(_rule("="))
        A(_wrap("Shown so the scan is auditable: you can see it considered each name and "
                "why it declined, rather than trusting a black box.", 2))
        A("")
        for d in sorted(res.diagnostics, key=lambda x: (x["status"], x["ticker"])):
            if d["status"] == "ok":
                continue
            A(f"  {d['ticker']:<7}{d['status']:<10}{(d.get('reason') or '')[:62]}")
            for key, label in (("too_large", "too large"), ("blocked", "blocked"),
                               ("too_expensive", "over price cap"),
                               ("lottery", "odds too poor"),
                               ("unsized", "unsized"), ("build_errors", "build error")):
                for item in (d.get(key) or [])[:2]:
                    A(f"  {'':<7}{'':<10}  {label}: {item[:58]}")
        if res.errors:
            A("")
            A("  ERRORS")
            for e in res.errors:
                A(f"  {e['ticker']:<8}{e['reason'][:70]}")

    # ------------------------------------------------ footer
    A("")
    A(_rule("="))
    A("  BEFORE YOU TRADE ANY OF THIS")
    A(_rule("="))
    for line in [
        "This is a screening tool, not advice, and not a prediction. It finds setups "
        "that are statistically favourable on the measures described above. Favourable "
        "is not the same as profitable: individual trades lose, and a positive expected "
        "value only shows up over a large number of them, taken at consistent size.",
        "Every price here comes from delayed quotes and a model. Re-check the live "
        "bid/ask in your broker before sending an order -- if the real market is "
        "materially worse than the price shown, the edge the scanner measured is not "
        "there and the trade is not the same trade.",
        "The expected-value and realised-vol probability figures assume implied "
        "volatility keeps exceeding realised volatility, which is true on average over "
        "time but false exactly when volatility expands. That is the main way this "
        "approach loses money, and no amount of scanning removes it.",
        "Short options carry assignment risk and gap risk that stops cannot protect "
        "against. Position sizing, not prediction, is what keeps a losing streak "
        "survivable.",
    ]:
        A(_wrap("- " + line, 4))
        A("")
    A(_rule("="))
    return "\n".join(L)
