#!/usr/bin/env bash
# Run from the reservoir_computing repository root.
set -euo pipefail

ep_table=results/pass_gqc_r2_extended/ep_table.csv

python -m src.run_pass_gqc_mpsk --ep-table "$ep_table" \
  --output results/pass_gqc/applications_parallel2/mpsk

python -m src.run_pass_gqc_pam4 --ep-table "$ep_table" \
  --output results/pass_gqc/applications_parallel2/pam4

python -m src.run_pass_gqc_waveform_classification --ep-table "$ep_table" \
  --output results/pass_gqc/applications_parallel2/waveform_classification