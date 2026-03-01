#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "[bench_clean_run] removing criterion outputs"
rm -rf target/criterion

echo "[bench_clean_run] running cargo bench --bench hnsw --bench simd"
cargo bench --bench hnsw --bench simd "$@"
