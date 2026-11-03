"""
Verifies the generic payoff engine against the textbook closed-form result for
every strategy. If these pass, max loss / max profit / breakeven / POP are
right, which is the difference between a risk number you can size from and a
number that will hurt you.

Run: python3 optionscan/tests/test_strategies.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from optionscan.data.synthetic import SyntheticProvider
from optionscan.core.chain import ChainView, LiquidityRules
from optionscan.core import strategies as ST
from optionscan.core.bs import prob_above

MULT = 100
fails = []
def chk(name, got, want, tol=0.02):
    ok = abs(got - want) < tol
    print(f"{'PASS' if ok else 'FAIL'}  {name:<52} got={got:>12.4f} want={want:>12.4f}")
    if not ok: fails.append(name)

SLIP, COMM = 0.4, 0.65
p = SyntheticProvider()
T = 'AAPL'
spot = p.get_spot(T)
exps = p.get_expirations(T)
e = [x for x in exps if 30 <= (x - p.as_of).days <= 50][0]
dte = (e - p.as_of).days
cv = ChainView(T, spot, e, dte, p.get_chain(T, e), 0.0425, 0.005,
               LiquidityRules(min_open_interest=100))
rv = 0.22
print(f"Underlying {T} spot={spot:.2f}  expiry={e} dte={dte}  "
      f"ATM IV={cv.atm_iv():.3f}  tradable contracts={len(cv.df)}\n")

# ---------------------------------------------------------------- LONG CALL
print("="*95); print("LONG CALL"); print("="*95)
t = ST.build_long_option(cv, 'call', 0.55, SLIP, COMM, rv)
L = t.legs[0]; debit = L.fill_price(SLIP)
print(f"  {t.order_ticket()}   debit/share={debit:.3f}  net_cash=${t.net_cash:,.2f}")
chk("long call max_loss = debit*100 + comm", t.max_loss, debit*MULT + t.commission)
print(f"PASS  long call max_profit is unlimited: {t.max_profit is None}")
if t.max_profit is not None: fails.append("long call unlimited")
chk("long call breakeven = K + debit + comm/100",
    t.breakevens[0], L.strike + debit + t.commission/MULT, 0.05)
chk("long call P&L at K", t.pnl(L.strike), -(debit*MULT) - t.commission)
chk("long call P&L at K+20", t.pnl(L.strike+20), (20-debit)*MULT - t.commission)

# ------------------------------------------------------- CALL DEBIT SPREAD
print("\n"+"="*95); print("BULL CALL DEBIT SPREAD"); print("="*95)
t = ST.build_vertical(cv, 'call', 0.60, 0.30, SLIP, COMM, rv)
lo, sh = t.legs[0], t.legs[1]
debit = lo.fill_price(SLIP) - sh.fill_price(SLIP)
print(f"  {t.order_ticket()}   width={t.width:g}  debit/share={debit:.3f}")
chk("debit spread max_loss = debit*100 + comm", t.max_loss, debit*MULT + t.commission)
chk("debit spread max_profit = (width-debit)*100 - comm",
    t.max_profit, (t.width-debit)*MULT - t.commission)
chk("debit spread breakeven = longK + debit + comm/100",
    t.breakevens[0], lo.strike + debit + t.commission/MULT, 0.05)
chk("debit spread P&L at/above short strike", t.pnl(sh.strike+50), t.max_profit)
chk("debit spread P&L below long strike", t.pnl(lo.strike-50), -t.max_loss)

# ------------------------------------------------------ PUT CREDIT SPREAD
print("\n"+"="*95); print("BULL PUT CREDIT SPREAD"); print("="*95)
t = ST.build_credit_spread(cv, 'put', 0.25, 10.0, SLIP, COMM, rv)
sh, lo = t.legs[0], t.legs[1]
credit = sh.fill_price(SLIP) - lo.fill_price(SLIP)
print(f"  {t.order_ticket()}   width={t.width:g}  credit/share={credit:.3f}  "
      f"net_cash=${t.net_cash:,.2f}")
chk("credit spread max_profit = credit*100 - comm", t.max_profit, credit*MULT - t.commission)
chk("credit spread max_loss = (width-credit)*100 + comm",
    t.max_loss, (t.width-credit)*MULT + t.commission)
chk("credit spread breakeven = shortK - credit + comm/100",
    t.breakevens[0], sh.strike - credit + t.commission/MULT, 0.05)
chk("credit spread P&L above short strike", t.pnl(sh.strike+50), t.max_profit)
chk("credit spread P&L below long strike", t.pnl(lo.strike-50), -t.max_loss)
# POP must equal the lognormal probability of finishing above the breakeven
be = t.breakevens[0]
chk("credit spread POP = P(S_T > breakeven)", t.pop,
    prob_above(spot, be, dte/365.0, cv.atm_iv(), 0.0), 0.002)
print(f"      short delta={abs(sh.delta):.3f} -> naive POP={1-abs(sh.delta):.3f}; "
      f"exact POP={t.pop:.3f}  (the difference is the credit, correctly accounted for)")

# ------------------------------------------------------------ IRON CONDOR
print("\n"+"="*95); print("IRON CONDOR"); print("="*95)
t = ST.build_iron_condor(cv, 0.18, 10.0, SLIP, COMM, rv)
sp, lp, sc, lc = t.legs
credit = (sp.fill_price(SLIP) - lp.fill_price(SLIP)
          + sc.fill_price(SLIP) - lc.fill_price(SLIP))
mw = max(abs(sp.strike-lp.strike), abs(sc.strike-lc.strike))
print(f"  {t.order_ticket()}")
print(f"  credit/share={credit:.3f}  max width={mw:g}  breakevens={t.breakevens}")
chk("condor max_profit = credit*100 - comm", t.max_profit, credit*MULT - t.commission)
chk("condor max_loss = (maxwidth-credit)*100 + comm", t.max_loss, (mw-credit)*MULT + t.commission)
chk("condor has exactly 2 breakevens", len(t.breakevens), 2, 0.5)
chk("condor lower BE = shortPut - credit + comm/100",
    t.breakevens[0], sp.strike - credit + t.commission/MULT, 0.05)
chk("condor upper BE = shortCall + credit - comm/100",
    t.breakevens[1], sc.strike + credit - t.commission/MULT, 0.05)
chk("condor P&L between short strikes = max profit",
    t.pnl((sp.strike+sc.strike)/2), t.max_profit)
pop_manual = (prob_above(spot, t.breakevens[0], dte/365.0, cv.atm_iv())
              - prob_above(spot, t.breakevens[1], dte/365.0, cv.atm_iv()))
chk("condor POP = P(BE_low < S_T < BE_high)", t.pop, pop_manual, 0.002)

# ------------------------------------------------------- CASH-SECURED PUT
print("\n"+"="*95); print("CASH-SECURED PUT"); print("="*95)
t = ST.build_cash_secured_put(cv, 0.25, SLIP, COMM, rv)
L = t.legs[0]; credit = L.fill_price(SLIP)
print(f"  {t.order_ticket()}   credit/share={credit:.3f}  collateral=${t.collateral:,.0f}")
chk("CSP max_profit = credit*100 - comm", t.max_profit, credit*MULT - t.commission)
chk("CSP breakeven = K - credit + comm/100", t.breakevens[0],
    L.strike - credit + t.commission/MULT, 0.05)
chk("CSP max_loss (stock to zero) = (K-credit)*100 + comm",
    t.max_loss, (L.strike-credit)*MULT + t.commission)
chk("CSP collateral = K*100 - credit*100", t.collateral, L.strike*MULT - credit*MULT)

# ----------------------------------------------------------- COVERED CALL
print("\n"+"="*95); print("COVERED CALL  (payoff includes the 100 shares)"); print("="*95)
t = ST.build_covered_call(cv, 0.28, SLIP, COMM, rv)
L = t.legs[0]; credit = L.fill_price(SLIP)
print(f"  {t.order_ticket()} + 100 shares @ {spot:.2f}   credit/share={credit:.3f}")
chk("CC max_profit = (K - S + credit)*100 - comm",
    t.max_profit, (L.strike - spot + credit)*MULT - t.commission)
chk("CC breakeven = S - credit + comm/100", t.breakevens[0],
    spot - credit + t.commission/MULT, 0.05)
chk("CC P&L at/above strike = max profit", t.pnl(L.strike+50), t.max_profit)
chk("CC P&L at spot = credit*100 - comm", t.pnl(spot), credit*MULT - t.commission)

# ------------------------------------------------- EV integrator cross-check
print("\n"+"="*95); print("EXPECTED VALUE SANITY"); print("="*95)
t = ST.build_credit_spread(cv, 'put', 0.25, 10.0, SLIP, COMM, rv)
print(f"  EV at implied vol = ${t.ev_at_iv:>8.2f}  <- market's own pricing, "
      f"should be ~ -(slippage+commission)")
print(f"  EV at realised vol= ${t.ev_at_rv:>8.2f}  <- the variance-risk-premium edge")
ok = -60.0 < t.ev_at_iv < 5.0
print(f"{'PASS' if ok else 'FAIL'}  EV at IV is near zero net of costs (not a fantasy number)")
if not ok: fails.append("ev_at_iv plausibility")
mn = min(t.pnl(s) for s in [x*spot/100 for x in range(50,160)])
ok2 = mn >= -t.max_loss - 0.01
print(f"{'PASS' if ok2 else 'FAIL'}  no price in [0.5x, 1.6x] loses more than stated max loss")
if not ok2: fails.append("max loss containment")

print("\n"+"="*95)
print("ALL STRATEGY CHECKS PASSED" if not fails else f"{len(fails)} FAILURES: {fails}")
print("="*95)
sys.exit(1 if fails else 0)
