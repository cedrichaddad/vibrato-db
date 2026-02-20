#!/usr/bin/env bash
set -euo pipefail

export VIBRATO_SOAK_SECS="${VIBRATO_SOAK_SECS:-259200}"
export VIBRATO_SOAK_SEED="${VIBRATO_SOAK_SEED:-42}"
export VIBRATO_SOAK_TARGET_OPS_PER_SEC="${VIBRATO_SOAK_TARGET_OPS_PER_SEC:-500}"
export VIBRATO_SOAK_SAMPLE_INTERVAL_SECS="${VIBRATO_SOAK_SAMPLE_INTERVAL_SECS:-60}"

echo "[soak] running mixed-workload soak for ${VIBRATO_SOAK_SECS}s seed=${VIBRATO_SOAK_SEED} target_ops_per_sec=${VIBRATO_SOAK_TARGET_OPS_PER_SEC}"
cargo test --release --test v2_soak_mixed_workload_e2e -- --ignored --nocapture --test-threads=1
