#!/usr/bin/env bash
# Run inside the ROCm Toolbx or an equivalent Linux development environment.
set -euo pipefail
cd "$(dirname "$0")/.."
gd_venv="${GD_FINETUNE_VENV:-$PWD/.venv-finetune}"
if [[ ! -e "$gd_venv" ]]; then python3.12 -m venv "$gd_venv"; fi
"$gd_venv/bin/python" -m pip install 'torch==2.11.0' --index-url https://download.pytorch.org/whl/rocm7.2
"$gd_venv/bin/python" -m pip install 'transformers==4.57.6' 'peft==0.18.1' 'accelerate==1.12.0' -e .
"$gd_venv/bin/python" - <<'PY'
import json, torch
assert torch.version.hip and torch.cuda.is_available(), 'A working ROCm PyTorch device is required'
x = torch.randn(256, 256, device='cuda', requires_grad=True)
loss = (x @ x.T).square().mean()
loss.backward()
torch.cuda.synchronize()
assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
print(json.dumps({'torch': torch.__version__, 'hip': torch.version.hip,
                  'gpu': torch.cuda.get_device_name(0), 'forward_backward': 'passed'}))
PY
