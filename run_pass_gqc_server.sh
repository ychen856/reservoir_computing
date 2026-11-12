#!/usr/bin/env bash
# Run from the repository root on Linux.
set -euo pipefail

if (( $# < 5 )); then
  printf 'Usage: bash %s TASK SEED DESIGN XI2 MODE [task-specific Python options]\n' "$0" >&2
  exit 2
fi

task="$1"
seed="$2"
design="$3"
xi2="$4"
mode="$5"
shift 5

if [[ ! "$task" =~ ^[a-z][a-z0-9_]*$ || ! -f "src/run_pass_gqc_${task}.py" ]]; then
  printf 'Unknown task or wrong working directory: %s\n' "$task" >&2
  exit 2
fi
if [[ ! "$seed" =~ ^[0-9]+$ || ! "$design" =~ ^[0-9]+$ ||
      ! "$xi2" =~ ^[0-9]+([.][0-9]+)?$ ||
      ( "$mode" != graphene && "$mode" != matched_linear ) ]]; then
  printf 'Use nonnegative integer seed/design, decimal xi2, and graphene or matched_linear.\n' >&2
  exit 2
fi

python_bin="${PYTHON_BIN:-python}"
ep_table="${EP_TABLE:-results/passive_gqc/device_ep_candidates.csv}"
output_root="${OUTPUT_ROOT:-results/pass_gqc/server}"
n_dimers="${N_DIMERS:-2}"
cycles=0.5
case "$task" in
  xor|parity) cycles=1 ;;
esac

command -v "$python_bin" >/dev/null
if [[ ! -r "$ep_table" ]]; then
  printf 'EP table not found: %s\n' "$ep_table" >&2
  exit 1
fi

# Set shard-defining options last so task-specific options cannot override them.
args=(-u -m "src.run_pass_gqc_$task" "$@"
      --ep-table "$ep_table" --design-indices "$design"
      --xi2 "$xi2" --seeds "$seed" --modes "$mode"
      --n-dimers "$n_dimers" --routing interleaved --ordering age
      --cycles-per-symbol "${CYCLES_PER_SYMBOL:-$cycles}")

# Different task options or EP tables get independent outputs.
ep_hash="$(sha256sum < "$ep_table")"
config_hash="$(printf '%s\0' "$ep_hash" "$python_bin" "${args[@]}" | sha256sum)"
job="$output_root/${n_dimers}dimers/$task/seed_$seed/design_$design/xi2_$xi2/$mode/${config_hash%% *}"
mkdir -p "$job"
exec 9>"$job/.lock"
if ! flock -n 9; then
  printf 'Already running: %s\n' "$job" >&2
  exit 1
fi

if [[ -L "$job/completed" ]]; then
  printf 'Already completed: %s/completed/results\n' "$job"
  exit 0
fi

attempt="$(mktemp -d "$job/attempt.XXXXXXXX")"
printf '%q ' "$python_bin" "${args[@]}" --output "$attempt/results" > "$attempt/command.txt"
printf '\n' >> "$attempt/command.txt"
printf 'Running in %s\n' "$attempt"
"$python_bin" "${args[@]}" --output "$attempt/results" 2>&1 | tee "$attempt/run.log"
ln -s "${attempt##*/}" "$job/completed"
printf 'Completed: %s/completed/results\n' "$job"
