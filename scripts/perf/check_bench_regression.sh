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

baseline_parsed="$(mktemp)"
observed_parsed="$(mktemp)"
observed_retry_parsed="$(mktemp)"
failed_names="$(mktemp)"
tmp_out="$(mktemp)"
tmp_out_retry="$(mktemp)"
trap 'rm -f "$baseline_parsed" "$observed_parsed" "$observed_retry_parsed" "$failed_names" "$tmp_out" "$tmp_out_retry"' EXIT

awk -F'\t' '
  NF >= 2 && $1 !~ /^#/ {
    print $1 "\t" $2
  }
' "$BASELINE_FILE" > "$baseline_parsed"

parse_criterion_output() {
  local input_file="$1"
  local output_file="$2"
  awk '
  function to_ns(v, u) {
    gsub(/[\[\]]/, "", u)
    if (u == "ns") return v
    if (u == "us" || u == "µs") return v * 1000
    if (u == "ms") return v * 1000000
    if (u == "s") return v * 1000000000
    return -1
  }
  function emit_metric(name, median, unit) {
    ns = to_ns(median, unit)
    if (ns >= 0) {
      printf "%s\t%.6f\n", name, ns
    }
  }
  # Criterion sometimes prints benchmark name on a dedicated line, and `time:`
  # on the next line (e.g. some SIMD benches).
  NF == 1 && $1 ~ /^[^[:space:]]+\/[^[:space:]]+$/ {
    pending_name = $1
    next
  }
  # Single-line format:
  # name time: [low unit mid unit high unit]
  $2 == "time:" && $3 ~ /^\[/ {
    emit_metric($1, $5, $6)
    pending_name = ""
    next
  }
  # Two-line format:
  # name
  # time: [low unit mid unit high unit]
  $1 == "time:" && $2 ~ /^\[/ && pending_name != "" {
    emit_metric(pending_name, $4, $5)
    pending_name = ""
    next
  }
  ' "$input_file" > "$output_file"
}

collect_regressions() {
  local observed_file="$1"
  local output_failed_names="$2"
  local fail_count=0
  : > "$output_failed_names"

  echo "[check_bench_regression] threshold=${THRESHOLD_PCT}%"
  while IFS=$'\t' read -r name base; do
    [[ -z "${name}" ]] && continue
    obs="$(awk -F'\t' -v key="$name" '$1 == key { print $2; exit }' "$observed_file")"
    if [[ -z "$obs" ]]; then
      echo "MISSING: $name not found in benchmark output" >&2
      echo "$name" >> "$output_failed_names"
      fail_count=1
      continue
    fi
    if awk -v o="$obs" -v b="$base" -v t="$THRESHOLD_PCT" 'BEGIN { exit (o > b * (1.0 + t / 100.0)) ? 0 : 1 }'; then
      pct="$(awk -v o="$obs" -v b="$base" 'BEGIN { printf "%.2f", ((o - b) / b) * 100.0 }')"
      echo "REGRESSION: $name observed_ns=$obs baseline_ns=$base delta_pct=+$pct" >&2
      echo "$name" >> "$output_failed_names"
      fail_count=1
    else
      pct="$(awk -v o="$obs" -v b="$base" 'BEGIN { printf "%.2f", ((o - b) / b) * 100.0 }')"
      echo "OK: $name observed_ns=$obs baseline_ns=$base delta_pct=$pct"
    fi
  done < "$baseline_parsed"

  return "$fail_count"
}

echo "[check_bench_regression] running cargo bench --bench hnsw --bench simd (pass 1)"
cargo bench --bench hnsw --bench simd "$@" | tee "$tmp_out"
parse_criterion_output "$tmp_out" "$observed_parsed"

if collect_regressions "$observed_parsed" "$failed_names"; then
  echo "[check_bench_regression] PASSED"
  exit 0
fi

echo "[check_bench_regression] pass 1 had regressions; rerunning once to filter thermal/noise variance"
echo "[check_bench_regression] running cargo bench --bench hnsw --bench simd (pass 2)"
cargo bench --bench hnsw --bench simd "$@" | tee "$tmp_out_retry"
parse_criterion_output "$tmp_out_retry" "$observed_retry_parsed"

persistent_fail=0
while IFS= read -r name; do
  [[ -z "$name" ]] && continue
  base="$(awk -F'\t' -v key="$name" '$1 == key { print $2; exit }' "$baseline_parsed")"
  obs="$(awk -F'\t' -v key="$name" '$1 == key { print $2; exit }' "$observed_retry_parsed")"
  if [[ -z "$base" || -z "$obs" ]]; then
    echo "PERSISTENT MISSING: $name not found on rerun" >&2
    persistent_fail=1
    continue
  fi
  if awk -v o="$obs" -v b="$base" -v t="$THRESHOLD_PCT" 'BEGIN { exit (o > b * (1.0 + t / 100.0)) ? 0 : 1 }'; then
    pct="$(awk -v o="$obs" -v b="$base" 'BEGIN { printf "%.2f", ((o - b) / b) * 100.0 }')"
    echo "PERSISTENT REGRESSION: $name observed_ns=$obs baseline_ns=$base delta_pct=+$pct" >&2
    persistent_fail=1
  else
    pct="$(awk -v o="$obs" -v b="$base" 'BEGIN { printf "%.2f", ((o - b) / b) * 100.0 }')"
    echo "RECOVERED: $name observed_ns=$obs baseline_ns=$base delta_pct=$pct"
  fi
done < <(sort -u "$failed_names")

if [[ "$persistent_fail" -ne 0 ]]; then
  echo "[check_bench_regression] FAILED" >&2
  exit 1
fi

echo "[check_bench_regression] PASSED (after rerun confirmation)"
