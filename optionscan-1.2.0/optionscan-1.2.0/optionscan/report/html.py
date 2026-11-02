"""
Self-contained HTML report.

One file, no external requests, no CDN, no fonts to fetch. Open it from disk in
five years and it still renders. Light and dark are both authored (dark is a
selected set of steps, not an inverted flip), every chart is backed by a table
of the same numbers, and the reasons sit beside the risks on every card.
"""
from __future__ import annotations

import html as _h
from datetime import datetime

from .charts import HOVER_JS, payoff_chart, score_bars

CSS = """
:root{color-scheme:light dark}
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{background:var(--plane);color:var(--ink);
  font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;
  -webkit-font-smoothing:antialiased}
:root,:root[data-theme="light"]{
  --plane:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
  --grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);
  --profit:#2a78d6;--loss:#e34948;--accent:#2a78d6;--neutral:#f0efec;
  --good:#0ca30c;--warn:#fab219;--crit:#d03b3b;--chipbg:#f0efec}
@media (prefers-color-scheme:dark){
  :root:where(:not([data-theme="light"])){
  --plane:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
  --grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
  --profit:#3987e5;--loss:#e66767;--accent:#3987e5;--neutral:#383835;
  --good:#0ca30c;--warn:#fab219;--crit:#d03b3b;--chipbg:#2c2c2a}}
:root[data-theme="dark"]{
  --plane:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
  --grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
  --profit:#3987e5;--loss:#e66767;--accent:#3987e5;--neutral:#383835;
  --good:#0ca30c;--warn:#fab219;--crit:#d03b3b;--chipbg:#2c2c2a}

.wrap{max-width:1120px;margin:0 auto;padding:28px 20px 80px}
header.top{display:flex;justify-content:space-between;align-items:flex-start;
  gap:16px;flex-wrap:wrap;margin-bottom:6px}
h1{font-size:25px;margin:0 0 2px;letter-spacing:-.01em}
.sub{color:var(--ink2);font-size:13.5px}
button{font:inherit;font-size:13px;padding:7px 13px;border-radius:8px;
  border:1px solid var(--border);background:var(--surface);color:var(--ink);cursor:pointer}
.banner{margin:16px 0;padding:11px 14px;border-radius:9px;font-size:13.5px;
  border:1px solid var(--border);background:var(--surface);color:var(--ink2)}
.banner.sim{border-left:3px solid var(--warn)}
.banner.adv{border-left:3px solid var(--accent)}
.stats{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0 6px}
.stat{background:var(--surface);border:1px solid var(--border);border-radius:10px;
  padding:11px 15px;min-width:132px}
.stat .k{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
.stat .v{font-size:21px;margin-top:3px;letter-spacing:-.01em}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{padding:8px 9px;text-align:right;border-bottom:1px solid var(--grid)}
th{color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;
  letter-spacing:.05em;white-space:nowrap}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),
th:nth-child(3),td:nth-child(3){text-align:left}
td.num,th.num{font-variant-numeric:tabular-nums}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:12px;
  padding:4px 16px 8px;margin:14px 0}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);
  margin:22px 0 8px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:13px;
  margin:16px 0;overflow:hidden}
.card>.hd{display:flex;align-items:center;gap:12px;padding:14px 18px;
  border-bottom:1px solid var(--grid);flex-wrap:wrap}
.rank{font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
.sym{font-size:20px;font-weight:650;letter-spacing:-.01em}
.strat{font-size:14.5px;color:var(--ink2)}
.score{margin-left:auto;display:flex;align-items:baseline;gap:5px}
.score b{font-size:23px;letter-spacing:-.02em}
.score span{font-size:11px;color:var(--muted)}
.chips{display:flex;gap:6px;flex-wrap:wrap;padding:11px 18px 0}
.chip{font-size:11.5px;padding:3px 9px;border-radius:999px;background:var(--chipbg);
  color:var(--ink2);white-space:nowrap}
.body{display:grid;grid-template-columns:minmax(0,1.05fr) minmax(0,1fr);gap:4px 26px;
  padding:12px 18px 18px}
@media (max-width:860px){.body{grid-template-columns:1fr}}
.block{margin:12px 0 2px}
.block h3{font-size:11px;text-transform:uppercase;letter-spacing:.07em;
  color:var(--muted);margin:0 0 7px}
.kv{display:grid;grid-template-columns:auto 1fr;gap:3px 14px;font-size:13.5px}
.kv dt{color:var(--ink2)}
.kv dd{margin:0;text-align:right;font-variant-numeric:tabular-nums}
ul.notes{margin:0;padding-left:17px;font-size:13.5px;color:var(--ink2)}
ul.notes li{margin:4px 0}
ul.notes.risk li::marker{color:var(--crit)}
ul.notes.why li::marker{color:var(--accent)}
.legs{font-size:13px;width:100%}
.legs .side{font-weight:650;font-size:11.5px;letter-spacing:.04em}
.legs .sell{color:var(--loss)}.legs .buy{color:var(--profit)}
details{margin:8px 0;font-size:13px}
summary{cursor:pointer;color:var(--ink2);font-size:12px}
code{font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--chipbg);
  padding:2px 6px;border-radius:5px;display:inline-block;margin:3px 0}

svg.payoff{width:100%;height:auto;display:block;margin:2px 0 4px;overflow:visible}
.pf-grid{stroke:var(--grid);stroke-width:1}
.pf-zero{stroke:var(--axis);stroke-width:1.5}
.pf-curve{fill:none;stroke:var(--ink);stroke-width:2;stroke-linejoin:round}
.pf-area{stroke:none}
.pf-profit{fill:var(--profit);opacity:.17}
.pf-loss{fill:var(--loss);opacity:.17}
.pf-strike{stroke:var(--axis);stroke-width:1;stroke-dasharray:2 3;opacity:.8}
.pf-spot{stroke:var(--ink2);stroke-width:1.5;stroke-dasharray:5 3}
.pf-be{fill:var(--surface);stroke:var(--ink);stroke-width:2}
.pf-tick,.pf-xlab{fill:var(--muted);font-size:10px;font-variant-numeric:tabular-nums}
.pf-spotlab{fill:var(--ink2);font-size:10px}
.pf-rlab{font-size:10px;font-weight:700;letter-spacing:.09em}
.pf-rlab-p{fill:var(--profit)}.pf-rlab-l{fill:var(--loss)}
.pf-hover{opacity:0;transition:opacity .1s}
.pf-hover.on{opacity:1}
.pf-cross{stroke:var(--ink2);stroke-width:1}
.pf-dot{fill:var(--surface);stroke:var(--ink);stroke-width:2}
.pf-capture{cursor:crosshair}
.pf-tip{position:absolute;z-index:40;pointer-events:none;background:var(--surface);
  border:1px solid var(--border);border-radius:8px;padding:6px 10px;font:13px system-ui;
  color:var(--ink);box-shadow:0 4px 14px rgba(0,0,0,.14)}
.pf-tip b{display:block;font-variant-numeric:tabular-nums}
.pf-tip span{color:var(--ink2);font-size:12px;font-variant-numeric:tabular-nums}

.sb{font-size:12.5px}
.sb-row{display:flex;align-items:center;gap:8px;margin:4px 0}
.sb-lab{width:96px;color:var(--ink2);font-size:12px}
.sb-track{display:inline-block;height:9px;background:var(--neutral);border-radius:5px;
  position:relative;overflow:hidden}
.sb-fill{position:absolute;left:0;top:0;height:9px;background:var(--accent);
  border-radius:5px}
.sb-val{width:26px;text-align:right;font-variant-numeric:tabular-nums}
.sb-wt{width:30px;text-align:right;color:var(--muted);font-size:11px}
.foot{margin-top:34px;padding-top:18px;border-top:1px solid var(--grid);
  font-size:13px;color:var(--ink2)}
.foot h2{margin-top:0}
.foot li{margin:9px 0}
"""


def _e(x) -> str:
    return _h.escape(str(x))


def _pct(x, dp=0):
    return "n/a" if x is None else f"{x:.{dp}%}"


def _usd(x, dp=0):
    return "unlimited" if x is None else f"${x:,.{dp}f}"


def _idea_card(idea, i: int) -> str:
    t, r, s, z, p = idea.trade, idea.regime, idea.score, idea.size, idea.plan
    o: list[str] = []
    A = o.append

    A('<div class="card">')
    A(f'<div class="hd"><span class="rank">#{idea.rank}</span>'
      f'<span class="sym">{_e(t.ticker)}</span>'
      f'<span class="strat">{_e(t.pretty)}</span>'
      f'<span class="score"><b>{s.total:.0f}</b><span>/100</span></span></div>')

    chips = [
        f"{t.dte} DTE", f"exp {t.expiry:%d %b %y}",
        f"spot ${t.spot:,.2f}",
        f"IV rank {r.vol.iv_rank:.0f} ({r.vol.iv_rank_source})",
        f"IV/RV {r.vol.iv_rv_ratio:.2f}x",
        r.trend.label.replace("_", " "),
        f"ADX {r.trend.adx:.0f}",
    ]
    if r.earnings_in_days is not None:
        chips.append(f"earnings in {r.earnings_in_days}d")
    A('<div class="chips">' + "".join(f'<span class="chip">{_e(c)}</span>' for c in chips)
      + "</div>")

    A('<div class="body">')

    # ---------------- left column ----------------
    A("<div>")
    A('<div class="block"><h3>Profit &amp; loss at expiry '
      f'({z.contracts} contract{"s" if z.contracts != 1 else ""})</h3>')
    A(payoff_chart(t, z.contracts, chart_id=f"pf{i}"))
    rows = []
    for mult, lbl in ((0.90, "-10%"), (0.95, "-5%"), (1.0, "unchanged"),
                      (1.05, "+5%"), (1.10, "+10%")):
        px = t.spot * mult
        rows.append(f"<tr><td>{lbl}</td><td class='num'>${px:,.2f}</td>"
                    f"<td class='num'>${t.pnl(px) * z.contracts:+,.0f}</td></tr>")
    A('<details><summary>Show these numbers as a table</summary>'
      '<table><thead><tr><th>Move</th><th class="num">Price</th>'
      '<th class="num">P&amp;L</th></tr></thead><tbody>'
      + "".join(rows) + "</tbody></table></details>")
    A("</div>")

    A('<div class="block"><h3>The contracts</h3><table class="legs"><tbody>')
    for leg in t.legs:
        cls = "sell" if leg.action == "sell" else "buy"
        A(f'<tr><td><span class="side {cls}">{leg.action.upper()}</span></td>'
          f'<td>{z.contracts} x {leg.expiry:%d %b} {leg.strike:g} {leg.kind.upper()}</td>'
          f'<td class="num">${leg.mid:,.2f}</td>'
          f'<td class="num">d {leg.delta:+.2f}</td>'
          f'<td class="num">IV {leg.iv:.0%}</td>'
          f'<td class="num">OI {leg.open_interest:,}</td></tr>')
    if t.strategy == "covered_call":
        A(f'<tr><td><span class="side buy">HOLD</span></td>'
          f'<td>{z.contracts * 100} shares</td><td class="num">${t.spot:,.2f}</td>'
          f'<td colspan="3"></td></tr>')
    A("</tbody></table>")
    A(f'<code>{_e(t.order_ticket(z.contracts))}</code>')
    A(f'<div class="kv" style="margin-top:8px">'
      f'<dt>Net {"credit" if t.net_cash > 0 else "debit"} per contract</dt>'
      f'<dd>${abs(t.net_cash):,.2f}</dd>'
      f'<dt>Work the limit at</dt><dd>${p.entry_limit:,.2f} {p.entry_type}</dd>'
      f'<dt>Do not {"accept less than" if p.entry_type == "credit" else "pay more than"}</dt>'
      f'<dd>${p.entry_limit_worst:,.2f}</dd></div>')
    A("</div>")

    if s_reasons := s.reasons:
        A('<div class="block"><h3>Why this setup</h3><ul class="notes why">'
          + "".join(f"<li>{_e(x)}</li>" for x in s_reasons) + "</ul></div>")
    risks = list(s.warnings) + list(z.warnings) + list(r.vol.notes) + list(t.notes)
    if risks:
        A('<div class="block"><h3>Risks and caveats</h3><ul class="notes risk">'
          + "".join(f"<li>{_e(x)}</li>" for x in risks) + "</ul></div>")

    WTS = {"regime_fit": 22, "premium_edge": 20, "expected_value": 18,
           "probability": 15, "liquidity": 15, "structure": 10}
    A('<div class="block"><h3>Score breakdown</h3>'
      + score_bars(s.components, WTS)
      + (f'<div style="font-size:12px;color:var(--muted);margin-top:6px">penalties: '
         + ", ".join(f"{_e(k)} &minus;{v:g}" for k, v in s.penalties.items()) + "</div>"
         if s.penalties else "")
      + "</div>")
    A("</div>")

    # ---------------- right column ----------------
    A("<div>")
    A('<div class="block"><h3>Position &amp; risk</h3><div class="kv">'
      f'<dt>Contracts</dt><dd>{z.contracts}</dd>'
      f'<dt>At risk</dt><dd>${z.risk_dollars:,.0f} ({z.pct_of_account_risked:.2f}%)</dd>'
      f'<dt>Buying power</dt><dd>${z.capital_required:,.0f} ({z.pct_of_account_capital:.2f}%)</dd>'
      f'<dt>Max profit</dt><dd>{_usd(t.max_profit * z.contracts if t.max_profit else None)}</dd>'
      f'<dt>Max loss</dt><dd>${t.max_loss * z.contracts:,.0f}</dd>'
      f'<dt>Breakeven</dt><dd>{", ".join(f"${b:,.2f}" for b in t.breakevens) or "n/a"}</dd>'
      f'<dt>Sizing limited by</dt><dd>{_e(z.limiting_factor)}</dd>'
      "</div></div>")

    A('<div class="block"><h3>The odds</h3><div class="kv">'
      f'<dt>Win probability at implied vol</dt><dd>{_pct(t.pop, 1)}</dd>'
      f'<dt>... at realised vol</dt><dd>{_pct(t.pop_at_rv, 1)}</dd>'
      f'<dt>Break-even win rate needed</dt><dd>{_pct(s.required_pop, 1)}</dd>'
      f'<dt>Probability edge</dt><dd>{s.pop_edge:+.1%}</dd>'
      + (f'<dt>Chance of touching short strike</dt><dd>{_pct(t.p_touch_short, 1)}</dd>'
         if t.p_touch_short is not None else "")
      + f'<dt>Average win / loss</dt><dd>${t.exp_win:,.0f} / ${t.exp_loss:,.0f}</dd>'
      f'<dt>Expected value (realised vol)</dt><dd>${t.ev_at_rv:+,.0f}</dd>'
      f'<dt>Expected value (implied vol)</dt><dd>${t.ev_at_iv:+,.0f}</dd>'
      "</div></div>")

    A('<div class="block"><h3>Greeks per contract</h3><div class="kv">'
      f'<dt>Delta</dt><dd>{t.net_delta:+.3f}</dd>'
      f'<dt>Theta</dt><dd>${t.net_theta:+,.2f}/day</dd>'
      f'<dt>Vega</dt><dd>${t.net_vega:+,.2f}/vol pt</dd>'
      f'<dt>Gamma</dt><dd>{t.net_gamma:+.4f}</dd>'
      "</div></div>")

    A('<div class="block"><h3>Exit plan &mdash; place on fill</h3><ul class="notes">'
      + "".join(f"<li>{_e(x)}</li>" for x in p.exit_rules) + "</ul></div>")
    A("</div>")

    A("</div></div>")
    return "".join(o)


def render(res, cfg) -> str:
    acct = cfg.account
    o: list[str] = []
    A = o.append
    A("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>")
    A("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    A(f"<title>Options scan {res.as_of:%Y-%m-%d}</title>")
    A(f"<style>{CSS}</style></head><body><div class='wrap'>")

    A("<header class='top'><div><h1>Daily options scan</h1>"
      f"<div class='sub'>{res.as_of:%A %d %B %Y, %H:%M} &middot; "
      f"data source: {_e(res.provider)} &middot; {res.universe_size} symbols scanned"
      "</div></div><button id='themeToggle'>Dark mode</button></header>")

    if res.provider == "synthetic":
        A("<div class='banner sim'><b>Simulated data.</b> These prices come from the "
          "built-in market simulator, not a real market. Useful for checking the tool "
          "end to end; not tradeable. Run with <code>--provider yahoo</code> for live "
          "quotes.</div>")
    else:
        A("<div class='banner'>Quotes are delayed. Re-check the live bid/ask in your "
          "broker before sending any order: if the real market is materially worse than "
          "shown here, the measured edge is not there.</div>")
    for a in res.advisories:
        A(f"<div class='banner adv'>{_e(a)}</div>")

    h = res.heat or {}
    A("<div class='stats'>"
      f"<div class='stat'><div class='k'>Ideas</div><div class='v'>{len(res.ideas)}</div></div>"
      f"<div class='stat'><div class='k'>Account</div><div class='v'>${acct.account_size:,.0f}</div></div>"
      f"<div class='stat'><div class='k'>Risk / trade</div><div class='v'>{acct.risk_per_trade_pct:g}%</div></div>"
      f"<div class='stat'><div class='k'>Total at risk</div><div class='v'>${h.get('total_risk', 0):,.0f}</div></div>"
      f"<div class='stat'><div class='k'>Portfolio heat</div><div class='v'>{h.get('pct_risked', 0):.1f}%</div></div>"
      "</div>")

    if res.ideas:
        A("<h2>Ranked summary</h2><div class='panel'><table><thead><tr>"
          "<th>#</th><th>Symbol</th><th>Strategy</th><th class='num'>Score</th>"
          "<th class='num'>DTE</th><th class='num'>POP</th><th class='num'>Edge</th>"
          "<th class='num'>Contracts</th><th class='num'>At risk</th>"
          "<th class='num'>Max profit</th><th class='num'>EV</th></tr></thead><tbody>")
        for i in res.ideas:
            t = i.trade
            mp = ("unlimited" if t.max_profit is None
                  else f"${t.max_profit * i.size.contracts:,.0f}")
            A(f"<tr><td>{i.rank}</td><td><b>{_e(t.ticker)}</b></td><td>{_e(t.pretty)}</td>"
              f"<td class='num'>{i.score.total:.0f}</td><td class='num'>{t.dte}</td>"
              f"<td class='num'>{t.pop:.0%}</td><td class='num'>{i.score.pop_edge:+.0%}</td>"
              f"<td class='num'>{i.size.contracts}</td>"
              f"<td class='num'>${i.size.risk_dollars:,.0f}</td>"
              f"<td class='num'>{mp}</td>"
              f"<td class='num'>${t.ev_at_rv * i.size.contracts:+,.0f}</td></tr>")
        A("</tbody></table></div>")
        A("<h2>Trade sheets</h2>")
        for n, idea in enumerate(res.ideas):
            A(_idea_card(idea, n))
    else:
        if getattr(res, "data_coverage", 1.0) < 0.5:
            A("<h2>No trades &mdash; but the data did not arrive</h2><div class='panel'>"
              "<p>Most of the universe returned no usable data, so this scan did not "
              "really run. Do <b>not</b> read the empty list as &ldquo;the market offered "
              "nothing today&rdquo; &mdash; it means the data source could not be "
              "reached. Check your connection and re-run.</p></div>")
        else:
            A("<h2>No trades today</h2><div class='panel'><p>Nothing met the configured "
              "thresholds. This is a normal and correct outcome: most underlyings offer "
              "no edge on most days, and forcing a trade on those days is how edge gets "
              "given back. The reasons for each symbol are below.</p></div>")

    # ------------------------------------------------ diagnostics
    skipped = [d for d in res.diagnostics if d["status"] != "ok"]
    if skipped:
        A("<h2>Why every other symbol was passed over</h2><div class='panel'>"
          "<table><thead><tr><th>Symbol</th><th>Outcome</th><th>Reason</th>"
          "<th class='num'>Trend</th><th class='num'>IV rank</th></tr></thead><tbody>")
        for d in skipped:
            A(f"<tr><td><b>{_e(d['ticker'])}</b></td><td>{_e(d['status'])}</td>"
              f"<td>{_e(d.get('reason', ''))}</td>"
              f"<td class='num'>{_e(d.get('trend', '') or '-')}</td>"
              f"<td class='num'>{d['iv_rank'] if d.get('iv_rank') is not None else '-'}</td></tr>")
        A("</tbody></table></div>")

    # ------------------------------------------------ footer
    A("<div class='foot'><h2>Before you trade any of this</h2><ul>")
    for x in [
        "<b>This is a screening tool, not advice, and not a prediction.</b> It finds "
        "setups that are statistically favourable on the measures shown. Favourable is "
        "not profitable: individual trades lose, and positive expected value only "
        "shows up across many trades taken at consistent size.",
        "<b>Re-check every price live.</b> Everything here is built on delayed quotes "
        "and a pricing model. If the real market is materially worse than shown, the "
        "measured edge is not there and it is not the same trade.",
        "<b>The edge rests on an assumption that sometimes fails.</b> The realised-vol "
        "probability and expected-value figures assume implied volatility keeps "
        "exceeding realised volatility. That holds on average and breaks precisely "
        "when volatility expands, which is the main way this approach loses money.",
        "<b>Short options carry assignment and gap risk</b> that no stop order can "
        "protect against. Position sizing, not forecasting, is what makes a losing "
        "streak survivable.",
        "<b>Model limitations:</b> European-style Black-Scholes pricing is used for "
        "American-style contracts (early-exercise premium ignored), probabilities "
        "assume a zero-drift lognormal distribution, and real returns have fatter "
        "tails than that model implies. Tail events are therefore understated.",
    ]:
        A(f"<li>{x}</li>")
    A("</ul></div>")

    A(f"<script>{HOVER_JS}</script>")
    A("</div></body></html>")
    return "".join(o)
