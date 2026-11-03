"""
Signal validation.

What this does and does not do, stated plainly, because an overstated backtest
is worse than none.

IT DOES test the two beliefs the scanner rests on, using only underlying price
history, with no look-ahead:

  1. DIRECTIONAL: does the trend score actually predict forward returns, and
     does a HIGHER score predict a BETTER outcome? Monotonicity across score
     buckets is the real test -- a signal that works only at its extremes is
     usually curve-fitting.

  2. VOLATILITY: is realised volatility mean-reverting? Selling premium after a
     volatility spike only works if elevated volatility tends to subside. This
     measures whether it does, per symbol, on real history.

IT DOES NOT backtest option P&L. Doing that honestly requires historical option
chains -- every strike's bid/ask on every past day -- which no free data source
provides. Reconstructing past option prices with a model and then "measuring"
profits against those same model prices tests the model, not the strategy, and
tends to produce impressive and meaningless equity curves. So it is not done
here, and the results below are reported as signal diagnostics rather than as
returns you could have earned.

Read the output as: is there a real edge in the signals this scanner uses on
these symbols? If bucket returns are flat or non-monotonic, the directional
side is not working on that universe and the premium-selling side is where any
edge lives.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .core import indicators as ind
from .core.regime import trend_score_series


@dataclass
class BacktestResult:
    horizons: tuple
    by_label: pd.DataFrame = field(default_factory=pd.DataFrame)
    by_bucket: pd.DataFrame = field(default_factory=pd.DataFrame)
    vol_mr: pd.DataFrame = field(default_factory=pd.DataFrame)
    tickers: list = field(default_factory=list)
    bars: int = 0
    notes: list = field(default_factory=list)


def _tstat(x: pd.Series) -> float:
    """
    t-statistic of the mean. Overlapping forward windows make returns
    autocorrelated, which inflates this, so it is a rough guide only -- treat
    |t| < 2 as noise and even |t| > 2 with suspicion on overlapping samples.
    """
    x = x.dropna()
    if len(x) < 10 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def run_backtest(provider, tickers: list, horizons=(5, 10, 21, 42),
                 lookback: int = 750) -> BacktestResult:
    res = BacktestResult(horizons=tuple(horizons), tickers=list(tickers))
    rows, vol_rows = [], []

    try:
        bench = provider.get_history("SPY", lookback + 80)["close"]
    except Exception:
        bench = None

    for tk in tickers:
        try:
            df = provider.get_history(tk, lookback + 80)
        except Exception:
            continue
        if df is None or len(df) < 260:
            continue
        res.bars += len(df)

        ts = trend_score_series(df, bench)
        close = df["close"]
        rv20 = ind.realized_vol(close, 20)
        rv_pct = ind.percent_rank(rv20, 252)

        fwd = {h: close.shift(-h) / close - 1.0 for h in horizons}
        # Forward realised vol, for the mean-reversion test.
        fwd_rv = {h: ind.realized_vol(close, h).shift(-h) for h in horizons}

        base = pd.DataFrame({
            "date": df.index,
            "ticker": tk,
            "score": ts["score"],
            "strength": ts["strength"],
            "label": ts["label"],
            "rv20": rv20,
            "rv_pct": rv_pct,
        })
        for h in horizons:
            base[f"fwd_{h}"] = fwd[h]
            base[f"fwdrv_{h}"] = fwd_rv[h]
        rows.append(base.dropna(subset=["score", "label"]))

    if not rows:
        res.notes.append("No symbol had enough history to test.")
        return res

    allr = pd.concat(rows, ignore_index=True)

    # ---------------------------------------------------- by regime label
    recs = []
    for label, grp in allr.groupby("label"):
        rec = {"regime": label, "n": len(grp)}
        for h in horizons:
            col = grp[f"fwd_{h}"].dropna()
            rec[f"mean_{h}d"] = col.mean() if len(col) else np.nan
            rec[f"hit_{h}d"] = (col > 0).mean() if len(col) else np.nan
            rec[f"t_{h}d"] = _tstat(col)
        recs.append(rec)
    res.by_label = pd.DataFrame(recs).set_index("regime")

    # ------------------------------------------- by score bucket (monotonicity)
    edges = [-100, -60, -40, -20, -5, 5, 20, 40, 60, 100]
    labels = [f"{a:+d}..{b:+d}" for a, b in zip(edges[:-1], edges[1:])]
    allr["bucket"] = pd.cut(allr["score"], bins=edges, labels=labels,
                            include_lowest=True)
    recs = []
    for b, grp in allr.groupby("bucket", observed=True):
        rec = {"score_bucket": str(b), "n": len(grp)}
        for h in horizons:
            col = grp[f"fwd_{h}"].dropna()
            rec[f"mean_{h}d"] = col.mean() if len(col) else np.nan
            rec[f"hit_{h}d"] = (col > 0).mean() if len(col) else np.nan
        recs.append(rec)
    res.by_bucket = pd.DataFrame(recs).set_index("score_bucket")

    # ------------------------------------------------ volatility mean reversion
    vq = pd.qcut(allr["rv_pct"].dropna(), 5,
                 labels=["lowest 20%", "20-40%", "40-60%", "60-80%", "highest 20%"],
                 duplicates="drop")
    tmp = allr.loc[vq.index].assign(vbucket=vq)
    recs = []
    for b, grp in tmp.groupby("vbucket", observed=True):
        rec = {"rv_percentile": str(b), "n": len(grp),
               "mean_rv20": grp["rv20"].mean()}
        for h in horizons:
            ratio = (grp[f"fwdrv_{h}"] / grp["rv20"]).replace([np.inf, -np.inf], np.nan).dropna()
            rec[f"fwd_rv_ratio_{h}d"] = ratio.mean() if len(ratio) else np.nan
            rec[f"fell_{h}d"] = (ratio < 1.0).mean() if len(ratio) else np.nan
        recs.append(rec)
    res.vol_mr = pd.DataFrame(recs).set_index("rv_percentile")
    return res


def format_backtest(res: BacktestResult) -> str:
    L: list[str] = []
    A = L.append
    W = 92
    A("=" * W)
    A("  SIGNAL VALIDATION".center(W))
    A("=" * W)
    A(f"  Symbols: {', '.join(res.tickers[:14])}"
      + (f" (+{len(res.tickers)-14} more)" if len(res.tickers) > 14 else ""))
    A(f"  Bars of history examined: {res.bars:,}")
    A("")
    A("  This validates the SIGNALS, using only underlying price history and no")
    A("  look-ahead. It is NOT an option-strategy backtest: honest option backtesting")
    A("  needs historical chains, which free data does not provide, and modelling past")
    A("  option prices then measuring profit against those same models would test the")
    A("  model rather than the strategy.")

    if res.by_label.empty:
        A("")
        A("  Not enough history to draw conclusions.")
        for n in res.notes:
            A(f"  {n}")
        return "\n".join(L)

    hs = res.horizons
    A("")
    A("=" * W)
    A("  1. FORWARD RETURNS BY TREND REGIME")
    A("=" * W)
    A("  Does the regime label say anything about what happens next?")
    A("")
    hdr = f"  {'REGIME':<19}{'N':>8}" + "".join(f"{str(h)+'d MEAN':>11}{'HIT':>7}" for h in hs)
    A(hdr)
    A("  " + "-" * (W - 4))
    order = ["strong_uptrend", "uptrend", "neutral", "downtrend", "strong_downtrend"]
    for lab in order:
        if lab not in res.by_label.index:
            continue
        r = res.by_label.loc[lab]
        line = f"  {lab:<19}{int(r['n']):>8,}"
        for h in hs:
            line += f"{r[f'mean_{h}d']:>10.2%}{r[f'hit_{h}d']:>7.0%}"
        A(line)
    A("")
    A("  t-statistics (|t| < 2 is noise; overlapping windows inflate these, so even")
    A("  larger values deserve suspicion):")
    for lab in order:
        if lab not in res.by_label.index:
            continue
        r = res.by_label.loc[lab]
        A(f"  {lab:<19}" + "  ".join(f"{h}d t={r[f't_{h}d']:+5.1f}" for h in hs))

    A("")
    A("=" * W)
    A("  2. MONOTONICITY -- the test that actually matters")
    A("=" * W)
    A("  A real signal gets steadily better as it gets stronger. If these rows are")
    A("  flat or jump around, the score is not measuring anything useful, however")
    A("  good the extreme buckets look.")
    A("")
    A(f"  {'SCORE BUCKET':<16}{'N':>8}" + "".join(f"{str(h)+'d MEAN':>11}{'HIT':>7}" for h in hs))
    A("  " + "-" * (W - 4))
    for b, r in res.by_bucket.iterrows():
        line = f"  {b:<16}{int(r['n']):>8,}"
        for h in hs:
            line += f"{r[f'mean_{h}d']:>10.2%}{r[f'hit_{h}d']:>7.0%}"
        A(line)

    # Rank correlation between bucket order and outcome.
    try:
        mid = res.by_bucket[f"mean_{hs[-1]}d"].dropna()
        rho = pd.Series(range(len(mid))).corr(pd.Series(mid.values), method="spearman")
        A("")
        A(f"  Rank correlation between score bucket and {hs[-1]}-day return: {rho:+.2f}")
        verdict = ("strong and monotonic" if rho > 0.7 else
                   "positive but noisy" if rho > 0.3 else
                   "weak -- treat directional signals with scepticism" if rho > -0.3 else
                   "INVERTED -- the score is pointing the wrong way on this universe")
        A(f"  Reading: {verdict}")
    except Exception:
        pass

    A("")
    A("=" * W)
    A("  3. VOLATILITY MEAN REVERSION -- the premium-selling assumption")
    A("=" * W)
    A("  Selling premium works because elevated volatility tends to subside. Ratios")
    A("  below 1.00 mean volatility fell over the following period. If the highest")
    A("  bucket does not show a ratio well below 1, premium selling has no tailwind")
    A("  on these symbols.")
    A("")
    A(f"  {'RV PERCENTILE':<16}{'N':>8}{'AVG RV':>9}"
      + "".join(f"{str(h)+'d RATIO':>11}{'FELL':>7}" for h in hs))
    A("  " + "-" * (W - 4))
    for b, r in res.vol_mr.iterrows():
        line = f"  {b:<16}{int(r['n']):>8,}{r['mean_rv20']:>8.1%}"
        for h in hs:
            line += f"{r[f'fwd_rv_ratio_{h}d']:>10.2f}{r[f'fell_{h}d']:>7.0%}"
        A(line)

    A("")
    A("=" * W)
    A("  HOW TO USE THIS")
    A("=" * W)
    for x in ["Treat it as a sanity check on the machinery, not a promise. Past signal "
              "behaviour is weak evidence about the future and these samples overlap.",
              "If section 2 is flat, the directional strategies have no demonstrated "
              "edge on your universe -- lean on the premium-selling side and size "
              "directional trades smaller.",
              "If section 3's top bucket sits well below 1.00, the volatility risk "
              "premium is present on these symbols, which is the foundation the credit "
              "strategies rest on.",
              "Re-run it on your own universe. Results differ a lot between liquid "
              "index ETFs and single high-volatility names."]:
        A("")
        words, line = x.split(), "  - "
        for w in words:
            if len(line) + len(w) + 1 > W - 2:
                A(line.rstrip()); line = "    " + w + " "
            else:
                line += w + " "
        A(line.rstrip())
    A("")
    A("=" * W)
    return "\n".join(L)
