#!/usr/bin/env bash
# Run from the reservoir_computing repository root.
set -euo pipefail

python_bin="${PYTHON_BIN:-python}"
output_root=results/pass_gqc/mpsk_coupled_followup
ep_table=results/passive_gqc/device_ep_candidates.csv
dry_run=false
case "${1:-}" in
  --dry-run) dry_run=true ;;
  '') ;;
  *) printf 'Usage: %s [--dry-run]\n' "$0" >&2; exit 2 ;;
esac

if [[ ! -f src/run_pass_gqc_mpsk.py || ! -r "$ep_table" ]]; then
  printf 'Run from the reservoir_computing repository root.\n' >&2
  exit 1
fi

is_complete() {
  local output="$1" order="$2" condition="$3" xi2="$4" seed="$5" dimers="$6" inductance="$7"
  [[ -s "$output/selected_test.csv" ]] || return 1
  "$python_bin" - "$output/selected_test.csv" "$order" "$condition" "$xi2" "$seed" "$dimers" "$inductance" <<'PY'
import csv
import math
import sys

path, order, condition, xi2, seed, dimers, inductance = sys.argv[1:]
task = f"{order}psk_{condition}" + ("_snr20" if condition == "awgn" else "")
try:
    with open(path, newline="") as stream:
        rows = list(csv.DictReader(stream))
    complete = len(rows) == 2 and {row["method"] for row in rows} == {"reservoir", "input_only"}
    complete = complete and all(
        row["task"] == task and row["mode"] == "graphene" and row["design_index"] == "5"
        and float(row["xi2"]) == float(xi2) and row["seed"] == seed
        and row["n_dimers"] == dimers and row["routing"] == "interleaved" and row["ordering"] == "age"
        and float(row["coupling_l_nh"]) == float(inductance) and float(row["coupling_r_ohm"]) == 1
        and int(row["n_test"]) == 20 * int(order) and math.isfinite(float(row["test"]))
        for row in rows
    )
except (KeyError, ValueError, OSError):
    complete = False
sys.exit(0 if complete else 1)
PY
}

missing=0
while read -r order condition xi2; do
  for dimers in 2 4; do
    case "$dimers" in
      2) inductance=2 ;;
      4) inductance=8 ;;
    esac
    for seed in 42 43 44; do
      output="$output_root/${dimers}dimers/${order}psk/$condition/design_5/xi2_$xi2/seed_$seed"
      if is_complete "$output" "$order" "$condition" "$xi2" "$seed" "$dimers" "$inductance"; then
        printf 'Skip completed: %s\n' "$output"
        continue
      fi
      missing=$((missing + 1))
      printf 'Pending: %s\n' "$output"
      if "$dry_run"; then
        continue
      fi

      "$python_bin" -u -m src.run_pass_gqc_mpsk \
        --ep-table "$ep_table" --design-indices 5 --xi2 "$xi2" --seeds "$seed" --modes graphene \
        --n-dimers "$dimers" --routing interleaved --ordering age \
        --coupling-l-nh "$inductance" --coupling-r-ohm 1 \
        --cycles-per-symbol 0.5 --repeats 20 \
        --psk-orders "$order" --conditions "$condition" --snr-db 20 \
        --output "$output"
      if ! is_complete "$output" "$order" "$condition" "$xi2" "$seed" "$dimers" "$inductance"; then
        printf 'Incomplete test results: %s\n' "$output" >&2
        exit 1
      fi
    done
  done
done <<'CASES'
4 awgn 1
4 isi 1
8 awgn 2
8 isi 1
CASES
printf 'Jobs requiring a run: %s\n' "$missing"
