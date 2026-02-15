#!/usr/bin/env bash
set -euo pipefail

export VIBRATO_SOAK_SECS="${VIBRATO_SOAK_SECS:-86400}"
export VIBRATO_SOAK_SEED="${VIBRATO_SOAK_SEED:-42}"

echo "[soak] running mixed-workload soak for ${VIBRATO_SOAK_SECS}s seed=${VIBRATO_SOAK_SEED}"
cargo test --release --test v2_soak_mixed_workload_e2e -- --ignored --nocapture --test-threads=1
