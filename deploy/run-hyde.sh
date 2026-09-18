#!/usr/bin/env bash
# Run inside the extracted project on Fedora; results stay in a new run directory.
set -euo pipefail
gd_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$gd_root"
export GD_NATIVE_ROOT="${GD_NATIVE_ROOT:-$gd_root/.native}"
gd_venv="${GD_VENV:-$gd_root/.venv}"
gd_models="${GD_MODEL_DIR:-$gd_root/models}"
mkdir -p "$gd_root/runs"
gd_report="$(mktemp -d "$gd_root/runs/hyde.XXXXXXXX")"
trap 'gd_exit=$?; printf "%s\n" "$gd_exit" > "$gd_report/exit-code.txt"; printf "Run evidence: %s (exit %s)\n" "$gd_report" "$gd_exit"' EXIT

run_step() {
  local gd_step="$1"
  shift
  printf 'Running %s\n' "$gd_step"
  "$@" 2>&1 | tee "$gd_report/$gd_step.log"
}

run_step preflight bash deploy/hyde-native.sh inspect
run_step build bash deploy/hyde-native.sh build
if [[ ! -e "$gd_venv" ]]; then
  run_step venv python3 -m venv "$gd_venv"
fi
[[ -x "$gd_venv/bin/python" ]] || { echo 'GD_VENV is not a Python environment.' >&2; exit 2; }
run_step install "$gd_venv/bin/python" -m pip install -e '.[serve,test]'
run_step tests "$gd_venv/bin/python" -m unittest discover -s tests -v
run_step weights "$gd_venv/bin/python" scripts/download_granite.py --directory "$gd_models" --reuse
run_step native-smoke "$gd_venv/bin/python" scripts/smoke_native.py \
  --server "$GD_NATIVE_ROOT/build-rocm/bin/llama-server" \
  --gguf "$gd_models/granite-4.1-3b-Q4_K_M.gguf" \
  --out "$gd_report/smoke.json" --log-dir "$gd_report/server-logs"
printf 'Native ROCm interface checks passed. This is not a domain-quality benchmark.\n'
