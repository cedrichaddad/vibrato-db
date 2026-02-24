#!/usr/bin/env bash
set -euo pipefail

RUNS="${1:-5}"
if ! [[ "$RUNS" =~ ^[0-9]+$ ]] || [ "$RUNS" -lt 1 ]; then
  echo "usage: $0 [runs>=1]" >&2
  exit 1
fi

declare -a p99_unfiltered=()
declare -a p99_filtered_tag_all=()
declare -a p99_filtered_combo=()

for i in $(seq 1 "$RUNS"); do
  echo "=== query_filter_p99 run $i/$RUNS ==="
  while IFS= read -r line; do
    case_name="$(printf '%s\n' "$line" | awk '{print $1}')"
    case_name="${case_name#case=}"
    p99_field="$(printf '%s\n' "$line" | awk '{print $4}')"
    p99="${p99_field#p99=}"
    p99="${p99%us}"

    case "$case_name" in
      unfiltered) p99_unfiltered+=("$p99") ;;
      filtered_tag_all) p99_filtered_tag_all+=("$p99") ;;
      filtered_combo) p99_filtered_combo+=("$p99") ;;
    esac
    echo "$line"
  done < <(cargo bench --bench query_filter_p99 | awk '/^case=/{print}')
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
echo "case=unfiltered       p99=$(median "${p99_unfiltered[@]}")us"
echo "case=filtered_tag_all p99=$(median "${p99_filtered_tag_all[@]}")us"
echo "case=filtered_combo   p99=$(median "${p99_filtered_combo[@]}")us"
