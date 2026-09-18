#!/usr/bin/env bash
set -euo pipefail
# Run on superrouter. SSH starts a detached supervisor inside the existing toolbox.
gd_remote=/var/home/adam/Development/granite-decisions/experiments/weekend-20260918
gd_ssh=(ssh -F /dev/null -i /home/adam/.ssh/homelab-hosts -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=15 adam@hyde-fedora.tail4df14e.ts.net)
case "${1:-}" in
  start)
    "${gd_ssh[@]}" "test ! -e $gd_remote/launch.json && test ! -e $gd_remote/run && test ! -e $gd_remote/CANCEL && podman exec --detach --user adam granite-decisions-rocm /usr/bin/bash $gd_remote/runtime/experiments/weekend-20260918/scripts/launch-supervisor.sh"
    # Podman detach only acknowledges spawning. Verify trainer launch separately.
    for gd_attempt in {1..12}; do
      if "${gd_ssh[@]}" "test -f $gd_remote/launch.json && cat $gd_remote/supervisor-status.json"; then exit 0; fi
      sleep 5
    done
    echo 'Detached launch did not produce a receipt; inspect hyde supervisor and data checks.' >&2
    exit 1
    ;;
  stop)
    "${gd_ssh[@]}" "touch $gd_remote/CANCEL"
    ;;
  status)
    "${gd_ssh[@]}" "if test -f $gd_remote/supervisor-status.json; then cat $gd_remote/supervisor-status.json; else echo 'Not started'; fi; if test -f $gd_remote/run/status.json; then cat $gd_remote/run/status.json; fi"
    ;;
  *) echo 'Usage: dispatch.sh start|stop|status' >&2; exit 2;;
esac
