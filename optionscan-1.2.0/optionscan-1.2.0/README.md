# optionscan

A daily options scanner. It screens a universe of liquid underlyings, decides
whether each one calls for buying premium, selling premium, or nothing at all,
picks the actual contracts, and writes a complete trade plan: strikes, expiry,
the limit price to work, how many contracts your account can carry, the profit
target and stop to place on fill, and the price that says the thesis is wrong.

**Read this first.** No scanner can promise profitable trades, and this one
does not. What it does is apply a consistent, auditable process: find setups
that are statistically favourable on measures you can inspect, size them so no
single loss hurts, and show the risks beside the reasons. Favourable is not the
same as profitable. Individual trades lose. A positive expected value only
shows up across many trades taken at consistent size. This is a screening tool,
not financial advice.

---

## Quick start

```bash
pip install -r requirements.txt

# See it work immediately, offline, on simulated data:
python -m optionscan scan --provider synthetic

# Live scan against your own account settings:
python -m optionscan scan --account 25000 --risk 1.0

# Narrow it down:
python -m optionscan scan --tickers AAPL,MSFT,SPY,NVDA
python -m optionscan scan --universe liquid --max-ideas 8
```

Every run writes four files to `./output/`:

| File | What it is |
|---|---|
| `scan_YYYY-MM-DD.html` | Full dashboard: payoff charts, trade sheets, score breakdowns. Open in a browser. |
| `scan_YYYY-MM-DD.txt` | The same trade sheets as plain text. Readable in a terminal, pasteable into a journal. |
| `scan_YYYY-MM-DD.csv` | One row per idea, for a spreadsheet. |
| `scan_YYYY-MM-DD.json` | Everything, including legs, greeks and the full plan, for programmatic use. |

Start with `--provider synthetic`. It runs with no network and no API key
against a built-in market simulator, so you can see the whole output and tune
your config before pointing it at real money. Reports from it are clearly
marked as simulated.

---

## What it actually does

### 1. Reads the regime, on two independent axes

**Trend** — direction *and*, separately, strength. Direction alone is not
enough: a +40% move that happened in three gaps is not a trend you can buy a
45-day call on. Strength blends ADX, EMA alignment, the R-squared of the
log-price regression, and momentum confirmation, so a disorderly spike is not
mistaken for a trend.

**Volatility** — is premium rich or cheap *right now* relative to how much the
stock is actually moving. This is the axis most retail scanners ignore, and it
decides whether you should be a buyer or a seller. Buying 60% implied vol on a
stock realising 30% loses money even when you get the direction right.

### 2. Picks the structure from the 2×2

|  | **IV cheap / normal** | **IV rich / extreme** |
|---|---|---|
| **Uptrend** | Buy premium: long call, call debit spread | Sell premium: put credit spread, cash-secured put |
| **Downtrend** | Buy premium: long put, put debit spread | Sell premium: call credit spread |
| **Neutral** | **Nothing — no edge. Sit out.** | Sell both sides: iron condor |

"No edge" is a real, frequent, correct answer. A neutral tape with cheap
premium offers nothing worth trading, and the scanner says so instead of
manufacturing an idea. Some days the honest output is an empty list, and the
report tells you why each symbol was passed over.

### 3. Picks contracts that can actually be traded

Implied volatility is **re-solved from the mid price** for every contract
rather than trusted from the data feed, because vendor IV fields are unreliable
on wide markets and every greek downstream inherits that error. Contracts are
then filtered on open interest, volume, bid/ask spread as a percentage of mid,
and whether the market is two-sided at all — *before* any spread is built, so a
good-looking structure is never priced off a quote you could not get filled at.

A spread quoted 1.00 / 1.40 costs 33% round-trip against the mid. No
statistical edge survives that, so those contracts are removed rather than
scored.

### 4. Sizes it to your account

Contract count comes from how many dollars you are willing to lose divided by
the real per-contract max loss — commissions and slippage included — never from
how much premium looks collectible. Spread width adapts: if a $25-wide SPY
spread risks more than your budget, the scanner narrows the structure until it
fits and tells you when even the tightest version does not.

### 5. Writes the exits before the entry

Profit target, stop, time stop, and the underlying price that invalidates the
thesis, all as net prices you can enter as GTC orders the moment you are
filled. Nobody decides these well while losing money.

---

## The two numbers that matter most

Most of the report is self-explanatory. These two deserve an explanation,
because they are where the real thinking is.

### Probability of profit is reported twice, deliberately

```
Win probability at implied vol     79.8%    <- the conservative read
... at realised vol                90.2%    <- what the trade is betting on
Break-even win rate needed         81.6%    <- what this payoff requires
Probability edge                   +8.6%
```

Measured at **implied** volatility, a fairly-priced spread *always* has a win
probability below the rate it needs to break even. That is not a quirk; it is
no-arbitrage. If the market's own volatility made a structure favourable, it
would be free money. Scoring credit spreads at implied vol would therefore
reject every one ever written — which is obviously wrong, because people do
make money selling premium.

The reason they do is that implied volatility usually *exceeds* what the stock
goes on to realise. The real bet behind a credit spread is "this will move less
than options are pricing." So both numbers are shown, and the gap between them
**is the trade's thesis**, stated out loud rather than buried. The scanner
scores the realised-vol edge and flags any trade that leans on it too heavily.

This is also the main way the approach loses money: the assumption holds on
average and breaks precisely when volatility expands.

### Break-even win rate, not max-profit ratios

The obvious formula — `max_loss / (max_loss + max_profit)` — is wrong for any
structure whose theoretical maximum is not realistically reachable. A long
put's "max profit" assumes the stock goes to zero, which would give it a 5%
break-even win rate and rank it above everything else on an edge it does not
have.

So the scanner uses the conditional expected win and loss instead:

```
break-even win rate = E[loss | it loses] / (E[loss | it loses] + E[win | it wins])
```

Well defined for every structure, bounded or not, and it is the number a win
rate actually has to beat. Both conditional averages are shown on each trade
sheet as "Average win / loss".

---

## Scoring

Each idea gets 0–100 from six components, all shown on the trade sheet with
their weights, plus explicit penalties. Nothing is hidden: if you cannot see
why an idea ranked where it did, you cannot tell a real edge from a bug.

| Component | Weight | What it measures |
|---|---|---|
| Regime fit | 22 | Does the structure match what the market is doing? |
| Premium edge | 20 | Are you on the right side of volatility pricing? |
| Expected value | 18 | EV at realised vol per dollar at risk, annualised |
| Probability | 15 | Realised-vol win rate minus the break-even win rate |
| Liquidity | 15 | Spread tightness and open-interest depth, judged by the **worst** leg |
| Structure | 10 | Credit-to-width, risk-reward, or yield on committed capital |

Penalties are applied for earnings inside the expiry, wide spreads, thin open
interest, an estimated rather than true IV rank, and over-reliance on the
volatility assumption.

---

## IV rank warms up over ~40 runs

IV rank — where today's implied vol sits in its own one-year range — is the
right measure of whether premium is rich, but computing it needs **implied
volatility history**, which no free data source publishes. A single API call
gives you today's IV and nothing else.

So the scanner keeps its own. Each run appends one row per symbol to
`~/.optionscan/iv_history.csv`, and the series builds itself up as you run it.
Until there are about 40 observations, the report shows an estimate derived
from the IV/realised-vol ratio, labelled **`rv_proxy`**, with a small score
penalty. After that it switches to a true IV rank, labelled **`history`**.

```bash
python -m optionscan history     # see how close each symbol is
```

An estimate you can see is worth more than a confident number you cannot audit,
which is why the source is printed on every idea. The IV/RV ratio works from
day one regardless, and is arguably the more direct measure anyway: it compares
what options cost against how much the stock is actually moving. Equity options
typically carry a 10–25% premium to realised vol, so ~1.15 is *fair*, not cheap.

---

## Validating the signals

```bash
python -m optionscan backtest --tickers SPY,QQQ,AAPL,MSFT,NVDA
```

This tests the two beliefs the scanner rests on, using only underlying price
history, with no look-ahead (a property verified in the test suite):

1. **Directional** — does the trend score predict forward returns, and does a
   *higher* score predict a *better* outcome? Monotonicity across score buckets
   is the real test; a signal that works only at its extremes is usually
   curve-fitting.
2. **Volatility** — is realised volatility mean-reverting? Selling premium
   after a volatility spike only works if elevated volatility tends to subside.

**It does not backtest option P&L**, and that omission is deliberate. Honest
option backtesting needs historical chains — every strike's bid/ask on every
past day — which free data does not provide. Reconstructing past option prices
with a model and then measuring profits against those same model prices tests
the model, not the strategy, and reliably produces impressive, meaningless
equity curves. You will see plenty of tools that do it. This one does not.

---

## Running it daily

**cron (macOS / Linux)** — `run_daily.sh` is ready to use:

```bash
crontab -e
45 16 * * 1-5  /full/path/to/run_daily.sh >> /full/path/to/scan.log 2>&1
```

**GitHub Actions** — `.github/workflows/daily-scan.yml` runs it in the cloud
every weekday with nothing installed locally, caches the IV history between
runs, and keeps each report as a downloadable artifact.

**Windows** — Task Scheduler, action:
`python -m optionscan scan --account 25000 --risk 1.0`, started in the project
folder.

Run it **after the close or before the open**, not intraday. End-of-day quotes
are settled, the scanner is built on daily bars, and you get the evening to
place orders rather than chasing a moving market.

---

## Configuration

```bash
python -m optionscan init-config my.yaml      # writes every setting with defaults
python -m optionscan scan --config my.yaml
```

Defaults are deliberately conservative: liquid names only, defined-risk
structures preferred, 1% of the account risked per trade, no short premium held
through earnings. Loosen them on purpose, not by accident. The settings worth
knowing:

```yaml
account:
  account_size: 25000
  risk_per_trade_pct: 1.0        # max loss on one position, as % of account
  max_portfolio_heat_pct: 6.0    # total at risk if you took every idea
  max_capital_per_trade_pct: 20  # buying-power cap per position
  commission_per_contract: 0.65
  fill_slippage: 0.40            # fraction of the bid/ask spread assumed given up

liquidity:
  min_open_interest: 200
  max_spread_pct: 0.10           # the single most important filter in options
  require_two_sided: true        # reject one-sided quotes (common after hours)

deltas:
  credit_short: 0.25             # ~75% nominal win rate before costs
  condor_short: 0.17
  debit_long: 0.60

earnings:
  allow_through_earnings: false  # earnings are the top cause of blown-up trades

output:
  min_score: 50                  # raise it to see fewer, better ideas
  max_ideas: 15
  max_per_ticker: 2
```

Universe presets for `--universe`: `default` (86 symbols), `liquid`, `etf`,
`mega`, `highvol`, `small`.

---

## Small accounts: a real constraint, not a setting to force

Options trade in whole contracts, and a single $5-wide spread on a $400 stock
risks ~$350. On a $25,000 account at 1% risk per trade, that is already over
budget. The scanner reports this plainly rather than silently dropping those
names, and tells you what risk percentage would be needed.

The honest answers are: trade lower-priced underlyings and narrow spreads, or
accept a larger risk percentage as a deliberate decision. Not: raise the
setting until the ideas appear.

---

## Known limitations

These are the things most likely to make the output wrong. They are listed
because you should know them, not because they are unusual.

- **Delayed quotes.** The Yahoo provider is ~15 minutes delayed, and outside
  market hours bid/ask often come back as zero (those rows are flagged and
  filtered). Always re-check the live market before sending an order. If the
  real spread is materially worse than shown, the measured edge is not there.
- **European pricing for American options.** Black-Scholes ignores the
  early-exercise premium. Immaterial for calls on low-dividend names, small but
  real for in-the-money puts and around dividends.
- **Lognormal probabilities.** All probability and EV figures assume a
  zero-drift lognormal distribution. Real returns have fatter tails, so **tail
  risk is understated** — the scanner's worst-case numbers are optimistic about
  how often the worst case happens.
- **No assignment modelling.** Short options can be assigned early, especially
  around ex-dividend dates. Cash-secured puts and covered calls are sized
  against a volatility-scaled adverse move, but a gap beyond it costs more.
- **No calendar or diagonal spreads.** Their P&L at the front expiry depends on
  modelling the back month's value, which needs assumptions about the volatility
  term structure that would not be visible to you in the output. Left out rather
  than guessed at.
- **Earnings dates come from the data feed** and are sometimes wrong or
  missing. Verify before holding short premium through a report.
- **The backtest validates signals, not strategy returns** — see above.
- **Correlation is not modelled.** Portfolio heat sums dollars at risk; it does
  not know that ten bullish tech trades are one trade. Diversify by hand.

---

## Auditing the scanner

Every idea is appended to `~/.optionscan/suggestions.jsonl` with its score,
probabilities and entry price. In three months you can ask the only question
that matters: did the high-scoring ideas actually do better than the
low-scoring ones? A scanner you cannot grade is a scanner you cannot trust.

---

## Tests

```bash
python optionscan/tests/test_bs.py           # pricing, greeks, IV, probabilities
python optionscan/tests/test_strategies.py   # every strategy's P&L math
python optionscan/tests/test_backtest.py     # no look-ahead, vectorised == live
```

The pricing core is verified against textbook values, put-call parity (exact to
1e-9), every greek against central finite differences, implied-vol round trips
through deep wings, and the expected-value integrator against closed-form
Black-Scholes (1e-8). Every strategy's max profit, max loss, breakevens and POP
are checked against their closed-form formulas, and the identity
`E[win]·P(win) − E[loss]·P(lose) = EV` is verified to reconstruct to the cent —
which validates the quadrature, the probabilities and the conditional
expectations together.

---

## Project layout

```
optionscan/
  cli.py              command line
  config.py           every threshold, overridable by YAML
  universe.py         scan universes
  backtest.py         signal validation
  core/
    bs.py             Black-Scholes, greeks, IV solver, probability, EV integrator
    indicators.py     EMA/RSI/ATR/ADX/MACD, realised and Parkinson volatility
    regime.py         trend and volatility classification (+ vectorised for backtests)
    chain.py          chain enrichment, IV re-solve, liquidity filtering
    strategies.py     structures and the generic payoff engine
    playbook.py       the regime -> strategy matrix
    risk.py           position sizing and trade management
    scoring.py        the six-component score
    scanner.py        orchestration
  data/
    base.py           provider interface -- implement to add a data source
    yahoo.py          Yahoo Finance (free, delayed)
    synthetic.py      offline market simulator for testing
    cache.py          disk cache, IV history store, suggestion log
  report/
    console.py        plain-text trade sheets
    html.py           self-contained dashboard
    charts.py         inline SVG payoff charts
    exports.py        CSV / JSON
```

Adding a different data source (Polygon, Tradier, IBKR, Schwab, ThetaData) means
implementing the five methods in `data/base.py`. Nothing else changes — the
scanner only ever talks to that interface.

---

## A note on the chart colours

Profit and loss are drawn in **blue and red**, not the conventional green and
red. Green-versus-red is the single worst pair for the roughly 8% of men with
red-green colour vision deficiency: for them, the most important distinction in
the whole report would disappear. The blue/red pair was validated at a
colour-difference of 21.6 (light mode) and 19.2 (dark), far above the threshold
of 8. Colour is never the only channel regardless — profit and loss regions are
labelled in text, and every chart has a table view.

---

## Disclaimer

This software is for research and education. It is not investment advice, not a
recommendation to buy or sell anything, and not a prediction. Options trading
involves substantial risk of loss and is not suitable for all investors; you can
lose more than you deposit with short options. The author and this tool accept
no liability for trading losses. Verify every price and every number
independently before risking money. If you are unsure whether options are
appropriate for you, consult a licensed financial adviser.
