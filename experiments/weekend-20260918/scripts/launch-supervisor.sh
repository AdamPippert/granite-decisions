#!/usr/bin/env bash
set -euo pipefail
gd_root=/var/home/adam/Development/granite-decisions/experiments/weekend-20260918
export PYTHONPATH="$gd_root/runtime/experiments/overnight-20260917/src"
exec /var/home/adam/Development/granite-decisions/.venv-finetune/bin/python -u \
  "$gd_root/runtime/experiments/weekend-20260918/scripts/supervise.py" \
  --root "$gd_root" > "$gd_root/supervisor.log" 2>&1
