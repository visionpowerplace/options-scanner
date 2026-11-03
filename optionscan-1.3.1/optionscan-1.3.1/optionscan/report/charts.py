"""
Inline SVG charts for the HTML report. No chart library, no CDN, no network --
the report must open from disk years from now and still render.

Colour decisions, and why they are not the finance-industry default:

  Profit/loss is POLARITY, so it gets a diverging encoding: two opposed hues
  with a neutral grey at zero. The pair used is BLUE (profit) / RED (loss)
  rather than the conventional green/red, because green-vs-red is the single
  worst pair for the ~8% of men with red-green colour vision deficiency -- for
  them the most important distinction in the whole report would vanish. The
  blue/red pair was validated at CVD Delta-E 21.6 (light) and 19.2 (dark),
  far above the 8.0 threshold.

  Colour is never the only channel regardless: profit and loss regions carry
  direct text labels, the zero line is drawn, and every chart is backed by a
  table of the same numbers.
"""
from __future__ import annotations

import json

# Diverging pair + chrome, light and dark. Validated with the palette checker.
LIGHT = {
    "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7",
    "profit": "#2a78d6", "loss": "#e34948", "neutral": "#f0efec",
    "accent": "#2a78d6",
}
DARK = {
    "surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
    "grid": "#2c2c2a", "axis": "#383835",
    "profit": "#3987e5", "loss": "#e66767", "neutral": "#383835",
    "accent": "#3987e5",
}


def _fmt(x: float) -> str:
    a = abs(x)
    if a >= 1000:
        return f"{x/1000:+.1f}k"
    return f"{x:+,.0f}"


def payoff_chart(trade, contracts: int = 1, width: int = 620, height: int = 210,
                 chart_id: str = "c0") -> str:
    """
    Expiry P&L curve for the position, scaled to the suggested contract count.

    The x-grid deliberately includes every strike and breakeven, so the
    piecewise-linear payoff is drawn exactly rather than approximated -- the
    corners land on real vertices instead of being rounded off between samples.
    """
    spot = trade.spot
    strikes = sorted({l.strike for l in trade.legs})
    lo = min([spot * 0.80] + [k * 0.94 for k in strikes])
    hi = max([spot * 1.20] + [k * 1.06 for k in strikes])

    xs = sorted(set(
        [lo + (hi - lo) * i / 140.0 for i in range(141)]
        + [k for k in strikes if lo <= k <= hi]
        + [b for b in trade.breakevens if lo <= b <= hi]
        + [spot]))
    pts = [(x, trade.pnl(x) * contracts) for x in xs]
    ys = [p for _, p in pts]
    ymin, ymax = min(ys), max(ys)
    pad = max((ymax - ymin) * 0.14, 1.0)
    ymin, ymax = ymin - pad, ymax + pad

    ml, mr, mt, mb = 56, 14, 16, 30
    pw, ph = width - ml - mr, height - mt - mb

    def sx(v: float) -> float:
        return ml + (v - lo) / (hi - lo) * pw

    def sy(v: float) -> float:
        return mt + (ymax - v) / (ymax - ymin) * ph

    y0 = sy(0.0)
    o: list[str] = []
    A = o.append
    A(f'<svg class="payoff" id="{chart_id}" viewBox="0 0 {width} {height}" '
      f'role="img" preserveAspectRatio="xMidYMid meet" '
      f'aria-label="Profit and loss at expiry for this position across underlying prices" '
      f'data-pts=\'{json.dumps([[round(x, 2), round(p, 1)] for x, p in pts])}\' '
      f'data-geom=\'{json.dumps({"ml": ml, "mr": mr, "mt": mt, "ph": ph, "pw": pw, "lo": lo, "hi": hi, "y0": y0, "ymin": round(ymin, 4), "ymax": round(ymax, 4)})}\'>')

    # y gridlines -- zero is always one of them and is labelled "0", because on a
    # P&L chart the breakeven line is the one the reader looks for first.
    ticks = {0.0, ymin + pad, ymax - pad}
    if ymax - pad > 0:
        ticks.add((ymax - pad) / 2.0)
    if ymin + pad < 0:
        ticks.add((ymin + pad) / 2.0)
    for v in sorted(ticks):
        yy = sy(v)
        A(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+pw}" y2="{yy:.1f}" '
          f'class="pf-grid"/>')
        A(f'<text x="{ml-8}" y="{yy+3.5:.1f}" class="pf-tick" text-anchor="end">'
          f'{"0" if abs(v) < 1e-9 else _fmt(v)}</text>')

    # profit / loss areas: clamping at zero makes the regions exact, because
    # every breakeven is already a vertex in the sample grid.
    for key, clamp in (("profit", lambda v: max(v, 0.0)), ("loss", lambda v: min(v, 0.0))):
        d = [f"M {sx(pts[0][0]):.1f} {y0:.1f}"]
        for x, v in pts:
            d.append(f"L {sx(x):.1f} {sy(clamp(v)):.1f}")
        d.append(f"L {sx(pts[-1][0]):.1f} {y0:.1f} Z")
        A(f'<path d="{" ".join(d)}" class="pf-area pf-{key}"/>')

    # zero line, then the curve on top
    A(f'<line x1="{ml}" y1="{y0:.1f}" x2="{ml+pw}" y2="{y0:.1f}" class="pf-zero"/>')
    curve = " ".join(f'{"M" if i == 0 else "L"} {sx(x):.1f} {sy(v):.1f}'
                     for i, (x, v) in enumerate(pts))
    A(f'<path d="{curve}" class="pf-curve"/>')

    # price context at the axis ends, so the strike labels have a scale to sit in
    A(f'<text x="{ml}" y="{mt+ph+13}" class="pf-xlab" text-anchor="start">'
      f'{lo:,.0f}</text>')
    A(f'<text x="{ml+pw}" y="{mt+ph+13}" class="pf-xlab" text-anchor="end">'
      f'{hi:,.0f}</text>')

    # strikes
    for k in strikes:
        if not lo <= k <= hi:
            continue
        A(f'<line x1="{sx(k):.1f}" y1="{mt}" x2="{sx(k):.1f}" y2="{mt+ph}" '
          f'class="pf-strike"/>')
        A(f'<text x="{sx(k):.1f}" y="{mt+ph+13}" class="pf-xlab" '
          f'text-anchor="middle">{k:g}</text>')

    # spot
    A(f'<line x1="{sx(spot):.1f}" y1="{mt}" x2="{sx(spot):.1f}" y2="{mt+ph}" '
      f'class="pf-spot"/>')
    A(f'<text x="{sx(spot):.1f}" y="{mt-4}" class="pf-spotlab" text-anchor="middle">'
      f'now {spot:,.2f}</text>')

    # breakevens
    for b in trade.breakevens:
        if lo <= b <= hi:
            A(f'<circle cx="{sx(b):.1f}" cy="{y0:.1f}" r="4" class="pf-be"/>')

    # direct region labels so colour is never the only channel
    if ymax > 0:
        A(f'<text x="{ml+6}" y="{mt+12}" class="pf-rlab pf-rlab-p">PROFIT</text>')
    if ymin < 0:
        A(f'<text x="{ml+6}" y="{mt+ph-6}" class="pf-rlab pf-rlab-l">LOSS</text>')

    # hover layer
    A(f'<g class="pf-hover" aria-hidden="true">'
      f'<line class="pf-cross" x1="0" y1="{mt}" x2="0" y2="{mt+ph}"/>'
      f'<circle class="pf-dot" r="4.5"/></g>')
    A(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="transparent" '
      f'class="pf-capture"/>')
    A("</svg>")
    return "".join(o)


def score_bars(components: dict, weights: dict, width: int = 300) -> str:
    """
    Horizontal bars for the six score components. One measure, one series, so a
    single hue is correct and no legend is needed -- the row labels name each
    bar directly. Weight is shown as text, not as a second visual channel.
    """
    rows = []
    bar_w = width - 150
    for k, v in components.items():
        pct = max(0.0, min(100.0, float(v))) / 100.0
        label = k.replace("_", " ")
        w = weights.get(k, 0)
        rows.append(
            f'<div class="sb-row"><span class="sb-lab">{label}</span>'
            f'<span class="sb-track" style="width:{bar_w}px">'
            f'<span class="sb-fill" style="width:{pct*bar_w:.1f}px"></span></span>'
            f'<span class="sb-val">{v:.0f}</span>'
            f'<span class="sb-wt">x{w:g}</span></div>')
    return '<div class="sb">' + "".join(rows) + "</div>"


HOVER_JS = r"""
// Crosshair + tooltip for every payoff chart. Plain DOM, no dependencies.
(function () {
  var tip = document.createElement('div');
  tip.className = 'pf-tip'; tip.style.display = 'none';
  document.body.appendChild(tip);

  document.querySelectorAll('svg.payoff').forEach(function (svg) {
    var pts, geom;
    try {
      pts = JSON.parse(svg.getAttribute('data-pts'));
      geom = JSON.parse(svg.getAttribute('data-geom'));
    } catch (e) { return; }
    var cap = svg.querySelector('.pf-capture');
    var cross = svg.querySelector('.pf-cross');
    var dot = svg.querySelector('.pf-dot');
    var hov = svg.querySelector('.pf-hover');
    if (!cap || !cross || !dot) return;

    function move(ev) {
      var r = svg.getBoundingClientRect();
      var cx = ev.clientX !== undefined ? ev.clientX
             : (ev.touches && ev.touches[0] ? ev.touches[0].clientX : 0);
      // Map client px -> viewBox units (the SVG scales with the card width).
      var vb = svg.viewBox.baseVal;
      var ux = (cx - r.left) / r.width * vb.width;
      var frac = (ux - geom.ml) / geom.pw;
      frac = Math.max(0, Math.min(1, frac));
      var price = geom.lo + (geom.hi - geom.lo) * frac;
      // Nearest sampled vertex -- exact on a piecewise-linear payoff.
      var best = 0, bd = Infinity;
      for (var i = 0; i < pts.length; i++) {
        var d = Math.abs(pts[i][0] - price);
        if (d < bd) { bd = d; best = i; }
      }
      var px = geom.ml + (pts[best][0] - geom.lo) / (geom.hi - geom.lo) * geom.pw;
      cross.setAttribute('x1', px); cross.setAttribute('x2', px);
      dot.setAttribute('cx', px);
      var yv = pts[best][1];
      var py = geom.mt + (geom.ymax - yv) / (geom.ymax - geom.ymin) * geom.ph;
      dot.setAttribute('cy', py);
      hov.classList.add('on');
      tip.style.display = 'block';
      tip.innerHTML = '<b>' + pts[best][0].toLocaleString(undefined,
            {minimumFractionDigits: 2, maximumFractionDigits: 2}) + '</b>'
          + '<span>' + (yv >= 0 ? '+' : '') + '$'
          + Math.round(yv).toLocaleString() + ' at expiry</span>';
      tip.style.left = Math.min(window.innerWidth - 150, cx + 12) + 'px';
      tip.style.top = (r.top + window.scrollY + 8) + 'px';
    }
    function leave() { hov.classList.remove('on'); tip.style.display = 'none'; }
    cap.addEventListener('mousemove', move);
    cap.addEventListener('mouseleave', leave);
    cap.addEventListener('touchmove', move, {passive: true});
    cap.addEventListener('touchend', leave);
  });

  var btn = document.getElementById('themeToggle');
  function effectiveDark() {
    var t = document.documentElement.getAttribute('data-theme');
    if (t) return t === 'dark';
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  }
  if (btn) btn.textContent = effectiveDark() ? 'Light mode' : 'Dark mode';
  if (btn) btn.addEventListener('click', function () {
    var root = document.documentElement;
    var now = root.getAttribute('data-theme');
    var next = now === 'dark' ? 'light' : (now === 'light' ? 'dark'
      : (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'light' : 'dark'));
    root.setAttribute('data-theme', next);
    btn.textContent = next === 'dark' ? 'Light mode' : 'Dark mode';
  });
})();
"""
