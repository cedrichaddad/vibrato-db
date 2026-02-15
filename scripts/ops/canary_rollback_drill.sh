#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-./vibrato_data}"
COLLECTION="${COLLECTION:-default}"
PORT="${PORT:-18081}"

echo "[canary] creating pre-canary snapshot"
cargo run --release -- snapshot-create --data-dir "${DATA_DIR}" --collection "${COLLECTION}"

echo "[canary] starting canary server on :${PORT}"
cargo run --release -- serve-v2 \
  --data-dir "${DATA_DIR}" \
  --collection "${COLLECTION}" \
  --dim 128 \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --checkpoint-interval-secs 30 \
  --compaction-interval-secs 180 \
  --public-health-metrics true &
SERVER_PID=$!
trap 'kill ${SERVER_PID} >/dev/null 2>&1 || true' EXIT

echo "[canary] waiting for readiness"
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${PORT}/v2/health/ready" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "[canary] run your validation traffic now (72h in production)."
echo "[canary] to rollback:"
echo "  1) stop canary process"
echo "  2) cargo run --release -- snapshot-restore --data-dir \"${DATA_DIR}\" --snapshot-dir <snapshot_dir>"
echo "  3) restart previous known-good binary"
