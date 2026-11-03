"""
Black-Scholes-Merton pricing, greeks, implied-volatility solving and
probability / expected-value math for European-style equity options.

Conventions used throughout this package
----------------------------------------
S      spot price of the underlying
K      strike
T      time to expiry in YEARS (calendar-day based: dte / 365)
r      continuously-compounded risk-free rate (0.04 = 4%)
q      continuous dividend yield (0.015 = 1.5%)
sigma  annualised volatility as a decimal (0.32 = 32% IV)
kind   'call' or 'put'

Greeks are returned in trader units:
    delta   per $1 move in S            (-1 .. 1)
    gamma   delta change per $1 move in S
    theta   dollars per CALENDAR DAY (negative for long options)
    vega    dollars per 1 VOLATILITY POINT (i.e. IV 30% -> 31%)
    rho     dollars per 1 PERCENTAGE POINT change in r

American-exercise early-exercise premium is ignored. For liquid US equity
options this is immaterial for calls on low-dividend names and small for
short-dated puts; it is called out in the README as a known approximation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Callable, Literal

Kind = Literal["call", "put"]

SQRT_2PI = math.sqrt(2.0 * math.pi)
DAYS_PER_YEAR = 365.0
# Numerical floors. Options inside these bounds are treated as expiring/degenerate.
MIN_T = 1.0 / (DAYS_PER_YEAR * 24.0)   # one hour
MIN_SIGMA = 1e-6
MAX_SIGMA = 8.0                         # 800% IV ceiling for the solver


# --------------------------------------------------------------------------
# Normal distribution helpers
# --------------------------------------------------------------------------
def norm_cdf(x: float) -> float:
    """Standard normal CDF via erf (accurate to ~1e-15)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / SQRT_2PI


def norm_ppf(p: float) -> float:
    """
    Inverse standard normal CDF (Acklam's rational approximation,
    |relative error| < 1.15e-9). Used for strike-from-probability work.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"norm_ppf requires 0<p<1, got {p}")
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        z = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * z + c[1]) * z + c[2]) * z + c[3]) * z + c[4]) * z + c[5]) / \
               ((((d[0] * z + d[1]) * z + d[2]) * z + d[3]) * z + 1.0)
    if p > phigh:
        z = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * z + c[1]) * z + c[2]) * z + c[3]) * z + c[4]) * z + c[5]) / \
                ((((d[0] * z + d[1]) * z + d[2]) * z + d[3]) * z + 1.0)
    z = p - 0.5
    rr = z * z
    return (((((a[0] * rr + a[1]) * rr + a[2]) * rr + a[3]) * rr + a[4]) * rr + a[5]) * z / \
           (((((b[0] * rr + b[1]) * rr + b[2]) * rr + b[3]) * rr + b[4]) * rr + 1.0)


# --------------------------------------------------------------------------
# Core pricing
# --------------------------------------------------------------------------
def _d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float):
    vol_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_t
    return d1, d1 - vol_t


def intrinsic(S: float, K: float, kind: Kind) -> float:
    return max(0.0, S - K) if kind == "call" else max(0.0, K - S)


def bs_price(S: float, K: float, T: float, r: float, q: float,
             sigma: float, kind: Kind) -> float:
    """Black-Scholes-Merton price of a European option with continuous yield q."""
    if S <= 0.0 or K <= 0.0:
        return 0.0
    if T <= MIN_T or sigma <= MIN_SIGMA:
        # Degenerate: discounted intrinsic on the forward.
        fwd = S * math.exp((r - q) * max(T, 0.0))
        return math.exp(-r * max(T, 0.0)) * intrinsic(fwd, K, kind)
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    disc_r, disc_q = math.exp(-r * T), math.exp(-q * T)
    if kind == "call":
        return S * disc_q * norm_cdf(d1) - K * disc_r * norm_cdf(d2)
    return K * disc_r * norm_cdf(-d2) - S * disc_q * norm_cdf(-d1)


@dataclass(frozen=True)
class Greeks:
    price: float
    delta: float
    gamma: float
    theta: float   # per calendar day
    vega: float    # per 1 volatility POINT
    rho: float     # per 1 percentage point of r
    d1: float
    d2: float

    def as_dict(self) -> dict:
        return asdict(self)


def greeks(S: float, K: float, T: float, r: float, q: float,
           sigma: float, kind: Kind) -> Greeks:
    """Full greek set in trader units. Robust at the expiry/zero-vol boundary."""
    if T <= MIN_T or sigma <= MIN_SIGMA or S <= 0.0 or K <= 0.0:
        itm = (S > K) if kind == "call" else (S < K)
        d = (1.0 if kind == "call" else -1.0) if itm else 0.0
        return Greeks(intrinsic(S, K, kind), d, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    disc_r, disc_q = math.exp(-r * T), math.exp(-q * T)
    sqrt_t = math.sqrt(T)
    pdf_d1 = norm_pdf(d1)
    price = bs_price(S, K, T, r, q, sigma, kind)

    gamma = disc_q * pdf_d1 / (S * sigma * sqrt_t)
    vega_per_vol = S * disc_q * pdf_d1 * sqrt_t          # per 1.00 of sigma

    if kind == "call":
        delta = disc_q * norm_cdf(d1)
        theta_yr = (-S * disc_q * pdf_d1 * sigma / (2.0 * sqrt_t)
                    + q * S * disc_q * norm_cdf(d1)
                    - r * K * disc_r * norm_cdf(d2))
        rho_per_unit = K * T * disc_r * norm_cdf(d2)
    else:
        delta = -disc_q * norm_cdf(-d1)
        theta_yr = (-S * disc_q * pdf_d1 * sigma / (2.0 * sqrt_t)
                    - q * S * disc_q * norm_cdf(-d1)
                    + r * K * disc_r * norm_cdf(-d2))
        rho_per_unit = -K * T * disc_r * norm_cdf(-d2)

    return Greeks(
        price=price,
        delta=delta,
        gamma=gamma,
        theta=theta_yr / DAYS_PER_YEAR,
        vega=vega_per_vol / 100.0,
        rho=rho_per_unit / 100.0,
        d1=d1,
        d2=d2,
    )


# --------------------------------------------------------------------------
# Implied volatility
# --------------------------------------------------------------------------
def implied_vol(price: float, S: float, K: float, T: float, r: float, q: float,
                kind: Kind, tol: float = 1e-7, max_iter: int = 100) -> float | None:
    """
    Solve for IV from a market price.

    Newton-Raphson seeded with the Brenner-Subrahmanyam approximation, with a
    guaranteed bisection fallback. Returns None when the price is outside the
    no-arbitrage bounds (so the caller can discard a bad quote rather than
    silently trust a garbage IV).
    """
    if price is None or price <= 0.0 or S <= 0.0 or K <= 0.0 or T <= MIN_T:
        return None

    disc_r, disc_q = math.exp(-r * T), math.exp(-q * T)
    # No-arbitrage bounds for a European option.
    if kind == "call":
        lo_bound, hi_bound = max(0.0, S * disc_q - K * disc_r), S * disc_q
    else:
        lo_bound, hi_bound = max(0.0, K * disc_r - S * disc_q), K * disc_r
    if price < lo_bound - 1e-8 or price > hi_bound + 1e-8:
        return None
    # Extrinsic (time) value is what carries volatility information. When it
    # rounds to zero -- deep ITM or worthless far-OTM contracts -- vega is ~0 and
    # IV is genuinely unrecoverable, not small. Return None so the caller drops
    # the quote instead of trusting a meaningless number. The scanner never
    # trades these contracts anyway (they fail the delta and liquidity filters).
    if price - lo_bound < max(1e-4, 1e-6 * S):
        return None

    # Brenner-Subrahmanyam seed: sigma ~ sqrt(2pi/T) * price / S
    sigma = max(0.05, min(3.0, SQRT_2PI / math.sqrt(T) * price / S))

    for _ in range(max_iter):
        model = bs_price(S, K, T, r, q, sigma, kind)
        diff = model - price
        if abs(diff) < tol:
            return sigma
        v = S * disc_q * norm_pdf(_d1_d2(S, K, T, r, q, sigma)[0]) * math.sqrt(T)
        if v < 1e-10:
            break
        step = diff / v
        nxt = sigma - step
        if not (MIN_SIGMA < nxt < MAX_SIGMA) or not math.isfinite(nxt):
            break
        sigma = nxt

    # Bisection fallback -- monotone in sigma, so this always converges.
    lo, hi = MIN_SIGMA, MAX_SIGMA
    if bs_price(S, K, T, r, q, hi, kind) < price:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if bs_price(S, K, T, r, q, mid, kind) < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    sigma = 0.5 * (lo + hi)
    return sigma if MIN_SIGMA < sigma < MAX_SIGMA else None


def strike_for_delta(S: float, T: float, r: float, q: float, sigma: float,
                     target_delta: float, kind: Kind) -> float:
    """
    Invert the delta formula to get the strike with a given |delta|.
    Used to aim strike selection before snapping to the real chain.
    """
    td = abs(target_delta)
    td = min(max(td, 1e-4), 1.0 - 1e-4)
    disc_q = math.exp(-q * T)
    # delta_call = e^{-qT} N(d1)  ->  d1 = N^{-1}(delta / e^{-qT})
    ratio = min(max(td / disc_q, 1e-6), 1.0 - 1e-6)
    d1 = norm_ppf(ratio)
    if kind == "put":
        d1 = -d1
    return S * math.exp(-d1 * sigma * math.sqrt(T) + (r - q + 0.5 * sigma * sigma) * T)


# --------------------------------------------------------------------------
# Probability measures
# --------------------------------------------------------------------------
def prob_itm(S: float, K: float, T: float, r: float, q: float,
             sigma: float, kind: Kind) -> float:
    """Risk-neutral probability of finishing in the money ( = N(d2) for a call)."""
    if T <= MIN_T or sigma <= MIN_SIGMA:
        return 1.0 if intrinsic(S, K, kind) > 0 else 0.0
    _, d2 = _d1_d2(S, K, T, r, q, sigma)
    return norm_cdf(d2) if kind == "call" else norm_cdf(-d2)


def prob_above(S: float, level: float, T: float, sigma: float,
               drift: float = 0.0) -> float:
    """
    P(S_T > level) under a lognormal with annual drift `drift`.
    drift=0 gives the zero-growth ("real-world, no view") measure that is the
    honest default for probability-of-profit reporting.
    """
    if level <= 0:
        return 1.0
    if T <= MIN_T or sigma <= MIN_SIGMA:
        return 1.0 if S > level else 0.0
    z = (math.log(S / level) + (drift - 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return norm_cdf(z)


def prob_touch(S: float, barrier: float, T: float, sigma: float,
               drift: float = 0.0) -> float:
    """
    Probability the underlying TOUCHES `barrier` at any point before T
    (first-passage of geometric Brownian motion). Roughly 2x prob_itm for an
    OTM barrier -- the number that matters for short-strike tail risk.
    """
    if T <= MIN_T or sigma <= MIN_SIGMA or S <= 0 or barrier <= 0:
        return 0.0
    if abs(barrier - S) < 1e-12:
        return 1.0
    v = sigma * math.sqrt(T)
    mu = drift - 0.5 * sigma * sigma
    x = math.log(barrier / S)
    lam = mu / (sigma * sigma)
    if barrier > S:   # up-and-touch
        p = 1.0 - norm_cdf((x - mu * T) / v)
        p += math.exp(2.0 * lam * x) * norm_cdf((-x - mu * T) / v)
    else:             # down-and-touch
        p = norm_cdf((x - mu * T) / v)
        p += math.exp(2.0 * lam * x) * (1.0 - norm_cdf((-x - mu * T) / v))
    return min(max(p, 0.0), 1.0)


# --------------------------------------------------------------------------
# Expected value of an arbitrary payoff under a lognormal terminal law
# --------------------------------------------------------------------------
def expected_payoff(payoff: Callable[[float], float], S: float, T: float,
                    sigma: float, drift: float = 0.0,
                    nodes: "list[float] | None" = None,
                    n: int = 1201, width: float = 9.0) -> float:
    """
    E[payoff(S_T)] integrated against the lognormal density of S_T with
    annualised drift `drift` and volatility `sigma`.

    `nodes` must list the KINK POINTS of the payoff -- i.e. every strike in the
    position. Option payoffs are piecewise-linear with a corner at each strike,
    and Simpson's rule assumes a smooth integrand, so a corner sitting between
    grid points silently degrades the result from 4th-order to 2nd-order
    accuracy. Splitting the integral at each strike and putting a grid node
    exactly on the corner restores full accuracy (verified to <1e-9 against
    closed-form Black-Scholes in tests/test_bs.py).

    This is the engine behind both EV figures the scanner reports:
      * EV at implied vol  -- what the market is pricing. Zero by construction
        for a fairly-quoted spread; a sanity check on the quote.
      * EV at realised vol -- the volatility-risk-premium edge: what the trade
        is worth if the stock keeps moving the way it HAS been moving rather
        than the way options are priced. This is the only defensible source of
        positive expected value in a premium-selling strategy.
    """
    if T <= MIN_T or sigma <= MIN_SIGMA:
        return payoff(S)

    v = sigma * math.sqrt(T)
    mu = math.log(S) + (drift - 0.5 * sigma * sigma) * T
    lo, hi = mu - width * v, mu + width * v

    # Integration boundaries: domain edges plus every in-range payoff kink.
    bounds = [lo, hi]
    for k in (nodes or []):
        if k and k > 0:
            lk = math.log(k)
            if lo < lk < hi:
                bounds.append(lk)
    bounds = sorted(set(bounds))

    span = hi - lo
    total = 0.0
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = b - a
        if seg <= 0:
            continue
        # Allocate grid points proportionally to segment width; keep it odd.
        m = max(21, int(n * seg / span) | 1)
        h = seg / (m - 1)
        acc = 0.0
        for i in range(m):
            x = a + i * h
            w = 1.0 if i in (0, m - 1) else (4.0 if i % 2 == 1 else 2.0)
            acc += w * payoff(math.exp(x)) * norm_pdf((x - mu) / v)
        total += acc * h / 3.0
    return total / v
