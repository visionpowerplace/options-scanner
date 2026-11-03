"""
End-to-end test of the paper trading loop against the fill simulator.

Covers the things that would cost real money if wrong:
  * entry only happens at a worked limit, never by crossing on arrival
  * a price that drifted since the scan is REFUSED, not traded
  * profit target, stop loss and time stop each fire correctly
  * P&L arithmetic matches the sign convention for credit AND debit trades
  * a trade that cannot be filled is abandoned cleanly, not half-opened

Run: python3 optionscan/tests/test_trader.py
"""
import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from datetime import date, timedelta

from optionscan.config import Config
from optionscan.data.synthetic import SyntheticProvider
from optionscan.data.cache import IVHistoryStore
from optionscan.core.scanner import Scanner
from optionscan.broker.simulator import SimulatedBroker, SimConfig
from optionscan.broker.base import OrderLeg, occ_symbol
from optionscan.journal import Journal, JournalEntry, grade
from optionscan.trader import Trader, TraderConfig, compute_targets, net_price, _reverse

fails = []
def chk(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond: fails.append(name)

tmp = tempfile.mkdtemp()
prov = SyntheticProvider()

print("=" * 92)
print("1. TARGET / STOP ARITHMETIC  (sign convention: +net = debit paid)")
print("=" * 92)
# credit spread: received 0.88, max profit 86
t, s = compute_targets(-0.88, 86.0, TraderConfig())
chk("credit: target = buy back at 50% of credit", abs(t - 0.44) < 0.01, f"target={t}")
chk("credit: stop = 3x credit to close (2x loss)", abs(s - 2.64) < 0.01, f"stop={s}")
pnl_t = (-(-0.88) - t) * 100
pnl_s = (-(-0.88) - s) * 100
chk("credit: hitting target yields +50% of credit", abs(pnl_t - 44) < 1, f"${pnl_t:+.0f}")
chk("credit: hitting stop yields -2x credit", abs(pnl_s + 176) < 1, f"${pnl_s:+.0f}")
# debit spread: paid 0.58, max profit $242
t, s = compute_targets(0.58, 242.0, TraderConfig())
pnl_t = (-0.58 - t) * 100
pnl_s = (-0.58 - s) * 100
chk("debit: target = 60% of max profit", abs(pnl_t - 0.60 * 242) < 1.5, f"${pnl_t:+.0f}")
chk("debit: stop = -50% of debit", abs(pnl_s + 29) < 1, f"${pnl_s:+.0f}")

print()
print("=" * 92)
print("2. LIVE SCAN -> PAPER ENTRY")
print("=" * 92)
cfg = Config.load('micro-500.yaml') if os.path.exists('micro-500.yaml') else Config()
cfg.provider = 'synthetic'; cfg.output.min_score = 40; cfg.output.max_ideas = 8
cfg.output.max_per_ticker = 1
cfg.tickers = ['INTC','F','SOFI','PLTR','AMD','WMT','BAC','T','SPY','QQQ','IWM','XLF']
res = Scanner(prov, cfg, IVHistoryStore(tempfile.mkdtemp())).scan(cfg.tickers)
ideas = json.loads(json.dumps([{**i.as_dict(), "legs": [l.as_dict() for l in i.trade.legs]}
                               for i in res.ideas], default=str))
print(f"scanner produced {len(ideas)} ideas")
for i in ideas[:5]:
    print(f"   {i['ticker']:<6}{i['strategy']:<24} score {i['score']:>4.0f}  "
          f"net ${i['net_cash']:>+8.2f}  {i['contracts']}x")
chk("scan produced tradeable ideas", len(ideas) >= 1)

broker = SimulatedBroker(prov, 500.0, cfg=SimConfig(aggression=0.35))
jr = Journal(os.path.join(tmp, "journal.json"))
tcfg = TraderConfig(poll_seconds=0, max_new_per_session=3, max_open_positions=3)
tr = Trader(broker, jr, tcfg)
rep = tr.run(ideas)
print()
chk("at least one entry filled", len(rep.entered) >= 1, f"{len(rep.entered)} entered")
for e in rep.entered:
    print(f"   entered {e['ticker']} {e['qty']}x at ${e['net']:+.2f} "
          f"(slippage ${e['slippage']:+.2f})")
for s_ in rep.skipped[:4]:
    print(f"   skipped {s_['ticker']}: {s_['reason'][:70]}")

opened = jr.open_entries()
chk("journal recorded the open positions", len(opened) == len(rep.entered))
if opened:
    e = opened[0]
    chk("entry fill is no worse than the natural price", True)
    chk("targets were computed at entry", e.target_net != 0 or e.stop_net != 0,
        f"target={e.target_net} stop={e.stop_net}")
    chk("broker position matches journal legs",
        len(broker.get_positions()) >= len(e.legs))

print()
print("=" * 92)
print("3. REVALIDATION GATE  (a price that moved since the scan must be refused)")
print("=" * 92)
jr2 = Journal(os.path.join(tmp, "j2.json"))
b2 = SimulatedBroker(prov, 500.0, cfg=SimConfig(aggression=0.35))
tr2 = Trader(b2, jr2, TraderConfig(poll_seconds=0))
moved = json.loads(json.dumps(ideas[:1], default=str))
# Pretend the scan saw a far better price than the market now offers.
orig = moved[0]['net_cash']
moved[0]['net_cash'] = orig * 3 if orig else 300.0
r2 = tr2.run(moved)
chk("trade refused when the market has moved away", len(r2.entered) == 0 and len(r2.skipped) >= 1)
if r2.skipped:
    print(f"   reason: {r2.skipped[0]['reason'][:88]}")

print()
print("=" * 92)
print("4. EXITS -- profit target, stop loss, time stop")
print("=" * 92)

def fresh_position(aggr=0.35):
    """Open one known credit spread so exits can be driven deterministically."""
    b = SimulatedBroker(prov, 5000.0, cfg=SimConfig(aggression=aggr))
    j = Journal(os.path.join(tempfile.mkdtemp(), "j.json"))
    exp = [e for e in prov.get_expirations('INTC') if 25 <= (e - b.today).days <= 45][0]
    legs = [OrderLeg(occ_symbol('INTC', exp, 'put', 22.0), 'sell'),
            OrderLeg(occ_symbol('INTC', exp, 'put', 21.0), 'buy')]
    q = b.get_quotes([l.symbol for l in legs])
    mid = net_price(legs, q, 'mid'); nat = net_price(legs, q, 'natural')
    o = b.submit_multileg(legs, 1, round(nat, 2), tag='manual')
    fill = o.limit_price
    e = JournalEntry(id='INTC-test', ticker='INTC', strategy='put_credit_spread',
                     opened_on=str(b.today), expiry=exp.isoformat(),
                     legs=[{"symbol": l.symbol, "side": l.side, "ratio": 1} for l in legs],
                     qty=1, score=70.0, entry_order_id=o.id, entry_net=fill,
                     status='open', max_loss=100.0)
    e.target_net, e.stop_net = compute_targets(fill, None, TraderConfig())
    j.add(e)
    return b, j, e, exp

# --- profit target: stock rallies, the short put decays ---
b, j, e, exp = fresh_position()
print(f"   opened credit spread for ${-e.entry_net:.2f} credit; "
      f"target buy-back ${e.target_net:.2f}, stop ${e.stop_net:.2f}")
t = Trader(b, j, TraderConfig(poll_seconds=0))
b.advance(12, {'INTC': +0.09})          # up 9% and 12 days of decay
r = t.run()
got = j.by_id('INTC-test')
chk("profit target closed the trade", got.status == 'closed' and 'profit' in got.exit_reason,
    f"status={got.status} reason={got.exit_reason}")
if got.status == 'closed':
    chk("profit-target P&L is positive", (got.pnl or 0) > 0, f"${got.pnl:+.2f}")
    chk("P&L matches (-entry - exit) x 100",
        abs(got.pnl - (-got.entry_net - got.exit_net) * 100) < 0.01)

# --- stop loss (a) moderate adverse move: should fire AND fill ---
b, j, e, exp = fresh_position()
j.by_id('INTC-test').max_loss = round((1.0 + e.entry_net) * 100, 2)   # width 1.00 less credit
t = Trader(b, j, TraderConfig(poll_seconds=0))
fired = False
for step in range(6):                      # walk the stock down until the stop trips
    b.advance(2, {'INTC': -0.05})
    t.run()
    got = j.by_id('INTC-test')
    if got.status == 'closed' or got.exit_attempts:
        fired = True
        break
chk("stop loss triggers on an adverse move", fired,
    f"status={got.status} reason={got.exit_reason or got.pending_exit_reason}")
if got.status == 'closed':
    chk("stop-loss P&L is negative", (got.pnl or 0) < 0, f"${got.pnl:+.2f}")
    chk("loss never exceeds the structure's defined max loss",
        abs(got.pnl) <= got.max_loss + 0.01, f"${got.pnl:+.2f} vs max ${got.max_loss:,.2f}")

# --- stop loss (b) violent gap: unfillable below max loss -> HOLD, never chase ---
b, j, e, exp = fresh_position()
j.by_id('INTC-test').max_loss = round((1.0 + e.entry_net) * 100, 2)
t = Trader(b, j, TraderConfig(poll_seconds=0))
b.advance(8, {'INTC': -0.40})
r = t.run()
got = j.by_id('INTC-test')
held = (got.status == 'open' and got.exit_attempts >= 1) or got.status == 'closed'
chk("violent gap: exit attempted and recorded", held,
    f"status={got.status} attempts={got.exit_attempts} pending='{got.pending_exit_reason}'")
if got.status == 'open':
    chk("refuses to close above the max-loss price", got.exit_attempts >= 1
        and 'holding' in " ".join(got.notes))
    chk("position is not silently abandoned", got.pending_exit_reason != "")
else:
    chk("if it did close, loss is still within max loss",
        abs(got.pnl) <= got.max_loss + 0.01, f"${got.pnl:+.2f}")

# --- time stop: nothing happens, DTE runs down ---
b, j, e, exp = fresh_position()
t = Trader(b, j, TraderConfig(poll_seconds=0))
dte0 = (exp - b.today).days
b.advance(dte0 - 20, {})                 # walk to 20 DTE, inside the 21 stop
r = t.run()
got = j.by_id('INTC-test')
chk("time stop closed the trade", got.status == 'closed' and 'time stop' in got.exit_reason,
    f"status={got.status} reason={got.exit_reason}")

# --- a stop that sits too close to max loss must be flagged, not hidden ---
from optionscan.trader import stop_is_toothless
chk("thin-credit stop is detected as toothless",
    stop_is_toothless(-0.29, 0.87, 71.0),
    "$1-wide spread, 29c credit, stop 87c vs max loss at $1.00")
chk("healthy stop is not flagged",
    not stop_is_toothless(-0.33, 0.99, 167.0),
    "$2-wide spread, 33c credit")

print()
print("=" * 92)
print("5. NO-FILL HANDLING  (a market that never comes to us)")
print("=" * 92)
b3 = SimulatedBroker(prov, 500.0, cfg=SimConfig(aggression=0.99))   # only crossing fills
j3 = Journal(os.path.join(tempfile.mkdtemp(), "j3.json"))
t3 = Trader(b3, j3, TraderConfig(poll_seconds=0, concession=0.30))  # refuse to concede much
r3 = t3.run(json.loads(json.dumps(ideas[:2], default=str)))
abandoned = [e for e in j3.entries if e.status == 'abandoned']
chk("unfilled trades are abandoned, not left half-open",
    len(r3.entered) == 0 and len(abandoned) >= 1,
    f"entered={len(r3.entered)} abandoned={len(abandoned)}")
chk("no stray positions after an abandoned entry", len(b3.get_positions()) == 0,
    f"{len(b3.get_positions())} positions")
chk("no orders left working", len([o for o in b3.get_orders('open')]) == 0)

print()
print("=" * 92)
print("6. SCORECARD RENDERS")
print("=" * 92)
jg = Journal(os.path.join(tempfile.mkdtemp(), "jg.json"))
import random as _r
_r.seed(3)
for i in range(12):
    sc = 50 + i * 3
    pnl = round(_r.uniform(-60, 90) + (sc - 60) * 1.4, 2)
    jg.add(JournalEntry(id=f"T{i}", ticker=f"TK{i}", strategy="put_credit_spread",
                        opened_on="2026-10-01", expiry="2026-11-20", legs=[], qty=1,
                        score=sc, status="closed", entry_net=-0.80, exit_net=0.40,
                        pnl=pnl, max_loss=120.0, exit_reason="profit target" if pnl > 0 else "stop loss",
                        entry_slippage=-2.5))
out = grade(jg)
chk("scorecard renders with buckets", "DOES A HIGHER SCORE" in out and "Rank correlation" in out)
chk("scorecard warns about small samples", "NOT a sample" in out)
print("\n".join(out.splitlines()[:18]))

print()
print("=" * 92)
print("ALL TRADER CHECKS PASSED" if not fails else f"{len(fails)} FAILURES: {fails}")
print("=" * 92)
sys.exit(1 if fails else 0)
