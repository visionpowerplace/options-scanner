#!/usr/bin/env bash
# Copy the newest scan into public/ (the dashboard) and state/ (what the
# trader reads), then commit the state. Usage: publish.sh <package dir>
set -uo pipefail

DIR="$1"

mkdir -p public state
cp "$DIR"/output/* public/ 2>/dev/null || true

L=$(ls -t public/scan_*.html 2>/dev/null | head -1)
if [ -n "$L" ]; then cp "$L" public/index.html; fi

J=$(ls -t public/scan_*.json 2>/dev/null | head -1)
if [ -n "$J" ]; then cp "$J" state/latest_scan.json; fi

ls -la public state

git config user.name "options-scanner"
git config user.email "actions@github.com"
git add state/ || true
if git diff --cached --quiet; then
  echo "No scan changes."
else
  git commit -m "scan $(date -u +%Y-%m-%d)"
  git pull --rebase --autostash || true
  git push
fi
