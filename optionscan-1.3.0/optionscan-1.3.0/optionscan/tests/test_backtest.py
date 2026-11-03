"""
Two properties the backtest must have, or its numbers mean nothing:

  1. The vectorised trend score must equal the live classifier's score. If they
     differ, the backtest is validating a signal the scanner does not trade.
  2. No look-ahead. The score at bar t must be identical whether computed from
     the full series or from a series truncated at t -- i.e. future bars cannot
     change a past value.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
from optionscan.data.synthetic import SyntheticProvider
from optionscan.core.regime import classify_trend, trend_score_series

fails = []
p = SyntheticProvider()
bench = p.get_history('SPY', 400)['close']

print("=" * 94)
print("1. VECTORISED SCORE == LIVE CLASSIFIER (checked at the final bar of each symbol)")
print("=" * 94)
print(f"{'TKR':6}{'vector score':>14}{'live score':>12}{'diff':>8}   "
      f"{'vec label':<18}{'live label':<18}")
worst = 0.0
for tk in p.universe():
    df = p.get_history(tk, 400)
    live = classify_trend(df, bench)
    vec = trend_score_series(df, bench)
    vs, vl = float(vec['score'].iloc[-1]), vec['label'].iloc[-1]
    d = abs(vs - live.score)
    worst = max(worst, d)
    ok = d < 0.15 and vl == live.label
    if not ok: fails.append(f"{tk} mismatch")
    print(f"{tk:6}{vs:>14.2f}{live.score:>12.2f}{d:>8.3f}   {str(vl):<18}{live.label:<18}"
          f"{'' if ok else '  <-- MISMATCH'}")
print(f"\nworst score difference across {len(p.universe())} symbols: {worst:.4f}")

print("\n" + "=" * 94)
print("2. NO LOOK-AHEAD (score at bar t must not change when future bars are added)")
print("=" * 94)
tk = 'NVDA'
full = p.get_history(tk, 400)
vec_full = trend_score_series(full, bench)
checked = 0
for cut in (260, 300, 340, 380):
    trunc = full.iloc[:cut]
    vec_tr = trend_score_series(trunc, bench)
    a = float(vec_tr['score'].iloc[-1])
    b = float(vec_full['score'].iloc[cut - 1])
    ok = abs(a - b) < 1e-6
    checked += 1
    print(f"{'PASS' if ok else 'FAIL'}  bar {cut:>4}: truncated={a:+8.3f}  "
          f"from full series={b:+8.3f}  diff={a-b:+.2e}")
    if not ok: fails.append(f"look-ahead at bar {cut}")

print("\n" + "=" * 94)
print("3. BACKTEST RUNS AND PRODUCES COHERENT AGGREGATES")
print("=" * 94)
from optionscan.backtest import run_backtest
r = run_backtest(p, ['SPY','QQQ','NVDA','AMD','XOM','WMT','TSLA','DIS'],
                 horizons=(5,21), lookback=400)
print(f"PASS  examined {r.bars:,} bars across {len(r.tickers)} symbols")
print(f"PASS  regime table rows: {len(r.by_label)}   bucket rows: {len(r.by_bucket)}   "
      f"vol rows: {len(r.vol_mr)}")
ok = len(r.by_label) >= 3 and len(r.by_bucket) >= 5 and len(r.vol_mr) == 5
print(f"{'PASS' if ok else 'FAIL'}  aggregates populated")
if not ok: fails.append("backtest aggregates")
# hit rates must be valid probabilities
bad = []
for h in (5,21):
    for frame,col in ((r.by_label,f'hit_{h}d'),(r.by_bucket,f'hit_{h}d')):
        v = frame[col].dropna()
        if len(v) and (v.min() < 0 or v.max() > 1): bad.append(col)
print(f"{'PASS' if not bad else 'FAIL'}  all hit rates are valid probabilities in [0,1]")
if bad: fails.append("hit rate range")

print("\n" + "=" * 94)
print("ALL BACKTEST CHECKS PASSED" if not fails else f"{len(fails)} FAILURES: {fails}")
print("=" * 94)
sys.exit(1 if fails else 0)
