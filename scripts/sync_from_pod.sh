#!/usr/bin/env bash
# Pull results back from the runpod box (activations, probe tables, figures, journal). Idempotent.
# Usage: scripts/sync_from_pod.sh            (reads POD_SSH from .env, e.g. "root@1.2.3.4 -p 12345")
set -euo pipefail
cd "$(dirname "$0")/.."
POD_SSH=${POD_SSH:-$(grep -E '^POD_SSH=' .env | cut -d= -f2-)}
[ -n "$POD_SSH" ] || { echo "set POD_SSH in .env (root@<ip> -p <port>)"; exit 1; }
HOST=${POD_SSH%% *}; PORT=$(echo "$POD_SSH" | sed -n 's/.*-p \([0-9]*\).*/\1/p'); PORT=${PORT:-22}
for d in data/processed figures journal; do
  rsync -avz -e "ssh -p $PORT" "$HOST:/workspace/mats12/$d/" "$d/"
done
echo "synced from $HOST"
