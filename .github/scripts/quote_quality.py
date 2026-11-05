"""Decide whether a scan ran against live quotes.

After the close the feed returns zero bid/ask, so nearly every contract
fails the liquidity filter and the scan reports one idea while still
looking perfectly healthy. Publishing a scan like that overwrites a good
earlier one with a worse picture, so the workflow withholds it instead.

Usage: quote_quality.py <scan json>
Prints: <ok|dead> <unusable> <scanned> <ideas>
"""
import json
import sys

data = json.load(open(sys.argv[1]))
diag = data.get("diagnostics") or []
ideas = len(data.get("ideas") or [])

unusable = 0
for item in diag:
    reason = item.get("reason") or ""
    status = item.get("status")
    if "liquidity filter" in reason or status in ("no_data", "error"):
        unusable += 1

# Live quotes leave a minority of names unusable. When most of the universe
# has no tradable two-sided market, the feed is asleep -- not the market.
if not diag:
    verdict = "ok" if ideas else "dead"
else:
    verdict = "dead" if unusable / len(diag) > 0.5 else "ok"

print(verdict, unusable, len(diag), ideas)
