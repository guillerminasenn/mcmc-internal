#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_PY="$SCRIPT_DIR/run.py"
ROOT_DIR="$SCRIPT_DIR/../../.."
ENV_BIN="$ROOT_DIR/.multip-env/bin"

GRID_COUNT=${1:-""}
REPLICATE_COUNT=${2:-20}
M=${3:-10}
N_ITERS=${4:-500}
LOG_DIR="$SCRIPT_DIR/logs"

mkdir -p "$LOG_DIR"

if [[ -z "$GRID_COUNT" ]]; then
  GRID_COUNT=$(
    GRID_HINT=$((REPLICATE_COUNT * M)) \
    "$ENV_BIN/python" - <<'PY'
import os
hint = int(os.environ.get("GRID_HINT", "1"))
workers = max(1, min(hint, os.cpu_count() or 1))
print(workers)
PY
  )
fi

if [[ "$GRID_COUNT" -lt 1 ]]; then
  echo "GRID_COUNT must be >= 1" >&2
  exit 1
fi

if [[ "$REPLICATE_COUNT" -lt 1 || "$M" -lt 1 || "$N_ITERS" -lt 1 ]]; then
  echo "REPLICATE_COUNT, M, and N_ITERS must be >= 1" >&2
  exit 1
fi

echo "Launching $GRID_COUNT workers (replicates=$REPLICATE_COUNT, M=$M, n_iters=$N_ITERS)..."
for i in $(seq 0 $((GRID_COUNT - 1))); do
  LOG_FILE="$LOG_DIR/independent_worker_${i}.log"
  echo "  worker $i -> $LOG_FILE"
  "$ENV_BIN/python" -u "$RUN_PY" \
    --replicate-count "$REPLICATE_COUNT" \
    --M "$M" \
    --n-iters "$N_ITERS" \
    --grid-count "$GRID_COUNT" \
    --grid-index "$i" > "$LOG_FILE" 2>&1 &
done

echo "Done. Use 'jobs' to see running workers, or 'tail -f logs/independent_worker_0.log' to watch one."
