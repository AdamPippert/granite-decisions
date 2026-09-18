#!/usr/bin/env bash
set -euo pipefail
cd /var/home/adam/Development/granite-decisions/experiments/overnight-20260917
# Independent in-container timeout survives loss of the SSH client.
gd_remaining=$(( $(date -d '2026-09-18T05:00:00-07:00' +%s) - $(date +%s) - 10 ))
if (( gd_remaining <= 0 )); then echo 'Deadline already passed'; exit 1; fi
export PYTHONPATH="$PWD/src"
exec timeout --signal=TERM --kill-after=10s "${gd_remaining}s" \
  ../../.venv-finetune/bin/python -u scripts/train_overnight.py \
  --data data-ready --out run --steps 400 --eval-every 50 \
  --train-until 2026-09-18T04:30:00-07:00 --finish-until 2026-09-18T04:55:00-07:00
