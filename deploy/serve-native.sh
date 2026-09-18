#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 3 ]]; then
  echo 'Usage: bash deploy/serve-native.sh /absolute/llama-server /absolute/granite.gguf embedding|baseline' >&2
  exit 2
fi
gd_server="$1"
gd_model="$2"
gd_mode="$3"
[[ -x "$gd_server" && -f "$gd_model" ]] || { echo 'Server or model missing.' >&2; exit 2; }
case "$gd_mode" in
  embedding) gd_port="${GD_LLAMA_PORT:-8091}"; gd_extra=(--embedding --pooling last) ;;
  baseline) gd_port="${GD_LLAMA_PORT:-8092}"; gd_extra=() ;;
  *) echo 'Mode must be embedding or baseline.' >&2; exit 2 ;;
esac
# Keep context and embedding microbatch consistent. Larger values require rebind.
# LLAMA_API_KEY may be supplied via environment; do not print it.
exec "$gd_server" --model "$gd_model" --host 127.0.0.1 --port "$gd_port" \
  --ctx-size 4096 --batch-size 4096 --ubatch-size 4096 --parallel 1 \
  --n-gpu-layers 99 --no-context-shift --no-webui --cors-origins localhost "${gd_extra[@]}"
