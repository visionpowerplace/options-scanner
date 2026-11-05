#!/usr/bin/env bash
# Run the scan, then judge whether its quotes were alive enough to publish.
# Writes quality=<ok|dead|none> to $GITHUB_OUTPUT. Usage: scan.sh <package dir>
set -uo pipefail

DIR="$1"

ARGS="--provider yahoo --quiet --universe ${UNIVERSE:-liquid}"
ARGS="$ARGS --account ${ACCOUNT_SIZE:-500} --risk ${RISK_PCT:-20}"
if [ -f "$DIR/micro-500.yaml" ]; then ARGS="$ARGS --config micro-500.yaml"; fi

( cd "$DIR" && python -m optionscan scan $ARGS ) || exit 1

J=$(ls -t "$DIR"/output/scan_*.json 2>/dev/null | head -1)
if [ -z "$J" ]; then
  echo "quality=none" >> "$GITHUB_OUTPUT"
  echo "No scan output produced."
  exit 0
fi

set -- $(python .github/scripts/quote_quality.py "$J")
echo "quality=$1" >> "$GITHUB_OUTPUT"
echo "$2 of $3 symbols had no usable market -> quotes $1 ; ideas: $4"
