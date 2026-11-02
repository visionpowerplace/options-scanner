"""Verification suite for the pricing core. Run: python3 -m optionscan.tests.test_bs"""
import math, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from optionscan.core.bs import (bs_price, greeks, implied_vol, prob_itm, prob_above,
                                prob_touch, expected_payoff, strike_for_delta, norm_ppf)

fails = []
def chk(name, got, want, tol=1e-6):
    ok = abs(got - want) < tol
    print(f"{'PASS' if ok else 'FAIL'}  {name:<46} got={got:.8f} want={want:.8f}")
    if not ok: fails.append(name)

print("=" * 86)
print("1. TEXTBOOK PRICE CHECKS  (Hull, Options Futures & Other Derivatives)")
print("=" * 86)
# Hull 7e Example 15.6: S=42, K=40, r=10%, sigma=20%, T=0.5 -> call 4.76, put 0.81
chk("Hull call S42 K40 T.5 r10% v20%", bs_price(42,40,0.5,0.10,0,0.20,'call'), 4.759422, 1e-5)
chk("Hull put  S42 K40 T.5 r10% v20%", bs_price(42,40,0.5,0.10,0,0.20,'put'),  0.808600, 1e-5)
# ATM zero-rate closed form: C = S*(2N(sigma*sqrt(T)/2)-1)
S,T,v = 100.0,1.0,0.20
chk("ATM zero-rate closed form", bs_price(S,S,T,0,0,v,'call'),
    S*(2*0.5*(1+math.erf((v*math.sqrt(T)/2)/math.sqrt(2)))-1), 1e-9)

print("\n" + "=" * 86)
print("2. PUT-CALL PARITY:  C - P = S*e^-qT - K*e^-rT   (must hold exactly)")
print("=" * 86)
for (S,K,T,r,q,v) in [(100,100,0.25,0.045,0.012,0.28),(57.3,65,0.08,0.05,0,0.62),
                      (410,380,1.5,0.03,0.02,0.19),(12.5,13,0.02,0.04,0.0,0.95)]:
    c = bs_price(S,K,T,r,q,v,'call'); p = bs_price(S,K,T,r,q,v,'put')
    chk(f"parity S={S} K={K} T={T}", c-p, S*math.exp(-q*T)-K*math.exp(-r*T), 1e-9)

print("\n" + "=" * 86)
print("3. GREEKS vs CENTRAL FINITE DIFFERENCE  (validates every analytic formula)")
print("=" * 86)
for kind in ('call','put'):
    S,K,T,r,q,v = 100.0,105.0,0.35,0.045,0.015,0.32
    g = greeks(S,K,T,r,q,v,kind)
    h=1e-4
    fd_delta=(bs_price(S+h,K,T,r,q,v,kind)-bs_price(S-h,K,T,r,q,v,kind))/(2*h)
    fd_gamma=(bs_price(S+h,K,T,r,q,v,kind)-2*bs_price(S,K,T,r,q,v,kind)+bs_price(S-h,K,T,r,q,v,kind))/(h*h)
    ht=1e-6
    fd_theta=-(bs_price(S,K,T+ht,r,q,v,kind)-bs_price(S,K,T-ht,r,q,v,kind))/(2*ht)/365.0
    hv=1e-6
    fd_vega=(bs_price(S,K,T,r,q,v+hv,kind)-bs_price(S,K,T,r,q,v-hv,kind))/(2*hv)/100.0
    hr=1e-7
    fd_rho=(bs_price(S,K,T,r+hr,q,v,kind)-bs_price(S,K,T,r-hr,q,v,kind))/(2*hr)/100.0
    chk(f"{kind} delta vs FD", g.delta, fd_delta, 1e-6)
    chk(f"{kind} gamma vs FD", g.gamma, fd_gamma, 1e-4)
    chk(f"{kind} theta/day vs FD", g.theta, fd_theta, 1e-6)
    chk(f"{kind} vega/vol-pt vs FD", g.vega, fd_vega, 1e-6)
    chk(f"{kind} rho/pct-pt vs FD", g.rho, fd_rho, 1e-6)

print("\n" + "=" * 86)
print("4. IMPLIED VOL ROUND TRIP  (price -> IV -> price, incl. deep wings)")
print("=" * 86)
worst=0.0
cases=[(100,100,.25,.3),(100,150,.08,.65),(100,60,1.0,.22),(55,56,.02,.9),
       (420,300,2.0,.18),(20,21,.12,.45),(100,250,.5,.80)]
for S,K,T,tv in cases:
    for kind in ('call','put'):
        px = bs_price(S,K,T,.045,.01,tv,kind)
        if px < 0.005: continue
        iv = implied_vol(px,S,K,T,.045,.01,kind)
        err = abs(iv-tv) if iv else 1.0
        worst=max(worst,err)
        print(f"{'PASS' if err<1e-5 else 'FAIL'}  IV {kind:4} S={S} K={K} T={T}  px={px:9.4f} "
              f"iv={iv if iv is None else round(iv,8)} true={tv}")
        if err>=1e-5: fails.append(f"iv {kind} {S}/{K}")
print(f"      worst absolute IV error across all cases: {worst:.2e}")
print(f"PASS  rejects arbitrage-violating price: {implied_vol(999,100,100,.25,.045,0,'call') is None}")
if implied_vol(999,100,100,.25,.045,0,'call') is not None: fails.append("iv bounds")
# A deep-ITM short-dated contract has ~zero extrinsic value -> no recoverable IV.
_zero_ext = bs_price(100,40,.05,.045,.01,.5,'call')
_r = implied_vol(_zero_ext,100,40,.05,.045,.01,'call')
print(f"PASS  returns None for zero-extrinsic contract (px={_zero_ext:.4f}): {_r is None}")
if _r is not None: fails.append("zero-extrinsic guard")

print("\n" + "=" * 86)
print("5. PROBABILITY MATH")
print("=" * 86)
chk("prob_itm ATM call ~ N(d2)", prob_itm(100,100,1,0,0,0.2,'call'), 0.460172, 1e-5)
chk("prob_itm call + put = 1", prob_itm(100,110,.5,.04,.01,.3,'call')+prob_itm(100,110,.5,.04,.01,.3,'put'), 1.0, 1e-12)
chk("prob_above zero-drift ATM = ~0.46", prob_above(100,100,1,0.2,0.0), 0.460172, 1e-5)
pt_up = prob_touch(100,110,0.5,0.30); pi_up = prob_above(100,110,0.5,0.30)
print(f"PASS  prob_touch(110) {pt_up:.4f} > prob_finish_above(110) {pi_up:.4f}  "
      f"ratio={pt_up/pi_up:.2f} (theory: ~2x)")
if not (1.7 < pt_up/pi_up < 2.3): fails.append("touch ratio")
chk("prob_touch at barrier = 1", prob_touch(100,100,0.5,0.3), 1.0, 1e-9)

print("\n" + "=" * 86)
print("6. EXPECTED-VALUE INTEGRATOR vs ANALYTIC BS  (Simpson accuracy check)")
print("=" * 86)
# Under drift r-q, discounted E[payoff] must reproduce the BS price exactly.
for S,K,T,r,q,v in [(100,105,.35,.045,.015,.32),(50,45,1.0,.04,0,.55),(250,260,.1,.05,.02,.25)]:
    for kind in ('call','put'):
        ev = expected_payoff(lambda x,K=K,kind=kind: max(0.,x-K) if kind=='call' else max(0.,K-x),
                             S,T,v,drift=r-q,nodes=[K])
        chk(f"EV integral = BS {kind} S={S} K={K}", math.exp(-r*T)*ev,
            bs_price(S,K,T,r,q,v,kind), 1e-8)

print("\n" + "=" * 86)
print("7. STRIKE-FROM-DELTA INVERSION  (round trip through greeks)")
print("=" * 86)
for td,kind in [(0.30,'put'),(0.20,'put'),(0.60,'call'),(0.16,'call'),(0.45,'call')]:
    K = strike_for_delta(100,0.1096,0.045,0.012,0.34,td,kind)
    d = abs(greeks(100,K,0.1096,0.045,0.012,0.34,kind).delta)
    chk(f"{kind} target delta {td}", d, td, 1e-7)

print("\n" + "=" * 86)
print(("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILURES: {fails}"))
print("=" * 86)
sys.exit(1 if fails else 0)
