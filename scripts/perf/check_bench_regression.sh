#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASELINE_FILE="$ROOT_DIR/benchmarks/baseline/hnsw_simd_baseline.tsv"
THRESHOLD_PCT="${BENCH_REGRESSION_THRESHOLD_PCT:-5}"

if [[ ! -f "$BASELINE_FILE" ]]; then
  echo "baseline file not found: $BASELINE_FILE" >&2
  exit 2
fi

cd "$ROOT_DIR"

tmp_out="$(mktemp)"
trap 'rm -f "$tmp_out"' EXIT

echo "[check_bench_regression] running cargo bench --bench hnsw --bench simd"
cargo bench --bench hnsw --bench simd "$@" | tee "$tmp_out"

to_ns() {
  local value="$1"
  local unit="$2"
  awk -v v="$value" -v u="$unit" 'BEGIN {
    mult = 1
    if (u == "ns") mult = 1
    else if (u == "us" || u == "µs") mult = 1000
    else if (u == "ms") mult = 1000000
    else if (u == "s") mult = 1000000000
    else {
      print "nan"
      exit 0
    }
    printf "%.6f", (v * mult)
  }'
}

declare -A baseline_ns
while IFS=$'\t' read -r name median_ns; do
  [[ -z "${name}" || "${name}" == \#* ]] && continue
  baseline_ns["$name"]="$median_ns"
done < "$BASELINE_FILE"

declare -A observed_ns
while read -r name median unit; do
  ns="$(to_ns "$median" "$unit")"
  if [[ "$ns" != "nan" ]]; then
    observed_ns["$name"]="$ns"
  fi
done < <(
  awk '
    $2 == "time:" && $3 ~ /^\[/ {
      name = $1
      median = $5
      unit = $6
      print name, median, unit
    }
  ' "$tmp_out"
)

failed=0
echo "[check_bench_regression] threshold=${THRESHOLD_PCT}%"
for name in "${!baseline_ns[@]}"; do
  base="${baseline_ns[$name]}"
  obs="${observed_ns[$name]:-}"
  if [[ -z "$obs" ]]; then
    echo "MISSING: $name not found in benchmark output" >&2
    failed=1
    continue
  fi
  if awk -v o="$obs" -v b="$base" -v t="$THRESHOLD_PCT" 'BEGIN { exit (o > b * (1.0 + t / 100.0)) ? 0 : 1 }'; then
    pct="$(awk -v o="$obs" -v b="$base" 'BEGIN { printf "%.2f", ((o - b) / b) * 100.0 }')"
    echo "REGRESSION: $name observed_ns=$obs baseline_ns=$base delta_pct=+$pct" >&2
    failed=1
  else
    pct="$(awk -v o="$obs" -v b="$base" 'BEGIN { printf "%.2f", ((o - b) / b) * 100.0 }')"
    echo "OK: $name observed_ns=$obs baseline_ns=$base delta_pct=$pct"
  fi
done

if [[ "$failed" -ne 0 ]]; then
  echo "[check_bench_regression] FAILED" >&2
  exit 1
fi

echo "[check_bench_regression] PASSED"
