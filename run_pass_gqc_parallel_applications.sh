#!/usr/bin/env bash
# Run from the reservoir_computing repository root.
set -euo pipefail

python_bin="${PYTHON_BIN:-python}"
ep_table="${EP_TABLE:-results/passive_gqc/device_ep_candidates.csv}"
n_dimers="${N_DIMERS:-2}"
output_root="${1:-results/pass_gqc/applications_parallel_${n_dimers}dimers}"
tasks=(mpsk pam4 waveform_classification)

if [[ ! "$n_dimers" =~ ^([2-9]|[1-9][0-9]+)$ ]]; then
  printf 'N_DIMERS must be an integer of at least 2.\n' >&2
  exit 2
fi
if [[ ! -r "$ep_table" ]]; then
  printf 'EP table not found: %s\n' "$ep_table" >&2
  exit 1
fi

common=(--ep-table "$ep_table" --n-dimers "$n_dimers" --routing interleaved --ordering age)
for task in "${tasks[@]}"; do
  printf 'Running %s with %s interleaved dimers\n' "$task" "$n_dimers"
  "$python_bin" -u -m "src.run_pass_gqc_${task}" "${common[@]}" \
    --output "$output_root/$task"
done