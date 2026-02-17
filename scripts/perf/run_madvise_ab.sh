#!/usr/bin/env bash
set -euo pipefail

export VIBRATO_ENFORCE_MADVISE_GATE="${VIBRATO_ENFORCE_MADVISE_GATE:-0}"

echo "[perf] madvise A/B gate enforce=${VIBRATO_ENFORCE_MADVISE_GATE}"
cargo test --release --test v2_madvise_ab_perf_e2e -- --ignored --nocapture --test-threads=1
