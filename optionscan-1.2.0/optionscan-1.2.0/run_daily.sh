#!/usr/bin/env bash
# Daily scan wrapper, suitable for cron.
#
# Install (runs at 4:45pm ET on weekdays, after the close):
#   crontab -e
#   45 16 * * 1-5  /full/path/to/run_daily.sh >> /full/path/to/scan.log 2>&1
#
# Run it AFTER the close (or before the open) rather than intraday: end-of-day
# quotes are settled, the scanner is built around daily bars, and you get the
# whole evening to place orders for the next session rather than chasing a
# moving market.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Point PY at a virtualenv interpreter if you use one.
PY="${PY:-python3}"

ACCOUNT="${ACCOUNT:-25000}"
RISK="${RISK:-1.0}"
UNIVERSE="${UNIVERSE:-liquid}"

echo "=== optionscan $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
"$PY" -m optionscan scan \
    --provider yahoo \
    --universe "$UNIVERSE" \
    --account "$ACCOUNT" \
    --risk "$RISK" \
    --out "$HERE/output" \
    --quiet

# Optional: email yourself the plain-text sheet.
# TODAY=$(date +%F)
# mail -s "Options scan $TODAY" you@example.com < "$HERE/output/scan_$TODAY.txt"
