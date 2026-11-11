#!/usr/bin/env bash
# Run from the reservoir_computing repository root.
set -euo pipefail

python_bin="${PYTHON_BIN:-python}"
ep_table="${EP_TABLE:-results/pass_gqc_r2_extended/ep_table.csv}"
output_root="${1:-results/pass_gqc/single_sweep}"
property_cycles="${PROPERTY_CYCLES:-0.5}"
benchmark_cycles="${BENCHMARK_CYCLES:-0.5}"
xor_parity_cycles="${XOR_PARITY_CYCLES:-1.0}"

properties=(
  memory_capacity memory_capacity_null
  nonlinear_memory_capacity nonlinear_memory_capacity_null
  cross_memory_capacity cross_memory_capacity_null
  kernel_rank generalization_rank effective_dimension
  voltage_decay fading_memory_convergence
)
benchmarks=(xor parity sine_cube narma10 henon rossler mackey_glass)
common=(
  --ep-table "$ep_table"
  --design-indices 37 5
  --xi2 0.5 1 2 4 8 12
  --modes graphene matched_linear
  --seeds 42 43 44
  --n-dimers 1
  --symbols 3000 --washout 300 --train-ratio 0.8
  --amplitude 0.001 --k 8 --points-per-symbol 16 --ridge 1e-6
)

command -v "$python_bin" >/dev/null || {
  printf 'Python executable not found: %s\n' "$python_bin" >&2
  exit 1
}
[[ -r "$ep_table" ]] || {
  printf 'EP table not found: %s\n' "$ep_table" >&2
  exit 1
}
for task in "${properties[@]}" "${benchmarks[@]}"; do
  [[ -f "src/run_pass_gqc_${task}.py" ]] || {
    printf 'Missing entry point: src/run_pass_gqc_%s.py\n' "$task" >&2
    exit 1
  }
done
if [[ -e "$output_root" ]]; then
  printf 'Output path already exists. Choose a new path: %s\n' "$output_root" >&2
  exit 1
fi

mkdir -p "$output_root"
run_task() {
  local group="$1" task="$2" cycles="$3"
  shift 3
  local destination="$output_root/$group/$task"
  mkdir -p "$output_root/logs/$group"
  printf '\n[%s] %s -> %s\n' "$group" "$task" "$destination"
  "$python_bin" -u -m "src.run_pass_gqc_$task" "${common[@]}" \
    --cycles-per-symbol "$cycles" --output "$destination" "$@" \
    2>&1 | tee "$output_root/logs/$group/$task.log"
}

for task in "${properties[@]}"; do
  run_task properties "$task" "$property_cycles"
done
for task in "${benchmarks[@]}"; do
  cycles="$benchmark_cycles"
  case "$task" in
    xor|parity) cycles="$xor_parity_cycles" ;;
  esac
  run_task benchmarks "$task" "$cycles" --validation-ratio 0.1
done
printf '\nCompleted 11 properties and 7 benchmarks in %s\n' "$output_root"
