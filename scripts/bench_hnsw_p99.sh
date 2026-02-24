#!/usr/bin/env bash
set -euo pipefail

RUNS="${1:-5}"
if ! [[ "$RUNS" =~ ^[0-9]+$ ]] || [ "$RUNS" -lt 1 ]; then
  echo "usage: $0 [runs>=1]" >&2
  exit 1
fi

declare -a p99_20=()
declare -a p99_50=()
declare -a p99_100=()

for i in $(seq 1 "$RUNS"); do
  echo "=== hnsw_p99 run $i/$RUNS ==="
  while IFS= read -r line; do
    ef="${line%% *}"
    ef="${ef#ef=}"
    p99_field="$(printf '%s\n' "$line" | awk '{print $4}')"
    p99="${p99_field#p99=}"
    p99="${p99%us}"
    case "$ef" in
      20) p99_20+=("$p99") ;;
      50) p99_50+=("$p99") ;;
      100) p99_100+=("$p99") ;;
    esac
    echo "$line"
  done < <(cargo bench -p vibrato-core --bench hnsw_p99 | awk '/^ef=/{print}')
done

median() {
  local -a arr=("$@")
  local n="${#arr[@]}"
  if [ "$n" -eq 0 ]; then
    echo "nan"
    return
  fi
  local sorted
  sorted="$(printf '%s\n' "${arr[@]}" | sort -g)"
  local mid=$((n / 2))
  if [ $((n % 2)) -eq 1 ]; then
    printf '%s\n' "$sorted" | sed -n "$((mid + 1))p"
  else
    local a b
    a="$(printf '%s\n' "$sorted" | sed -n "${mid}p")"
    b="$(printf '%s\n' "$sorted" | sed -n "$((mid + 1))p")"
    awk -v x="$a" -v y="$b" 'BEGIN { printf "%.2f\n", (x + y) / 2.0 }'
  fi
}

echo
echo "=== Median p99 across $RUNS runs ==="
echo "ef=20  p99=$(median "${p99_20[@]}")us"
echo "ef=50  p99=$(median "${p99_50[@]}")us"
echo "ef=100 p99=$(median "${p99_100[@]}")us"
