#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"
# Limit BLAS oversubscription. Users can override these environment values.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
repo="${REPO_ROOT:-$script_dir}"
output="${OUTPUT_ROOT:-results/pass_gqc/mpsk_d5_open_20reps}"
ep_table="${EP_TABLE:-results/pass_gqc_r2_extended/ep_table.csv}"
mode="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
  all|features|readout|plan) ;;
  *) printf 'Usage: bash run_gqc_mpsk_20reps.sh [all|features|readout|plan] [feature-runner options]\n' >&2; exit 2 ;;
esac
for option in "$@"; do
  case "$option" in
    --repo|--repo=*|--output|--output=*|--ep-table|--ep-table=*)
      printf 'Use REPO_ROOT, OUTPUT_ROOT and EP_TABLE environment variables with the shell launcher.\n' >&2
      exit 2 ;;
  esac
done
# Use the feature Python entry point directly for custom grids, and the readout
# entry point directly for a custom ridge grid or feature expansion.
if [[ "$mode" == plan ]]; then
  exec "$python_bin" "$script_dir/src/run_gqc_mpsk_resume.py" --repo "$repo" --ep-table "$ep_table" --output "$output" --dry-run "$@"
fi
if [[ "$mode" == all || "$mode" == features ]]; then
  "$python_bin" -u "$script_dir/src/run_gqc_mpsk_resume.py" --repo "$repo" --ep-table "$ep_table" --output "$output" "$@"
fi
if [[ "$mode" == all || "$mode" == readout ]]; then
  resolved_output="$("$python_bin" -c 'import pathlib,sys; print((pathlib.Path(sys.argv[1])/sys.argv[2]).resolve())' "$repo" "$output")"
  "$python_bin" -u "$script_dir/src/fit_gqc_mpsk_readout.py" \
    --features "$resolved_output" --output "${resolved_output}_readout"
fi
