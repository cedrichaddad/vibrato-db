#!/usr/bin/env bash
set -euo pipefail

export VIBRATO_CRASH_SEEDS="${VIBRATO_CRASH_SEEDS:-10}"

echo "[crash-matrix] running seeds=${VIBRATO_CRASH_SEEDS}"
cargo test --test v2_crash_matrix_kill9_e2e -- --ignored --nocapture --test-threads=1
