#!/usr/bin/env bash
# Fedora Atomic host: dedicated user Toolbx, no host package layering/reboot.
set -euo pipefail
cd "$(dirname "$0")/.."
gd_container="granite-decisions-rocm"
if ! podman container exists "$gd_container"; then
  toolbox create --container "$gd_container" --release 44 --assumeyes
fi
toolbox run --container "$gd_container" sudo dnf install -y \
  gcc gcc-c++ cmake git python3-pip python3-devel python3.12 \
  rocm-hip-devel hipblas-devel rocblas-devel rocminfo
toolbox run --container "$gd_container" bash deploy/run-hyde.sh
# Optional: toolbox run --container granite-decisions-rocm bash deploy/setup-finetune.sh
