# Passive Graphene Quantum Capacitor Reservoir Computing

The `passive-gqc` branch implements reservoir computing with passive graphene quantum capacitor (GQC) circuits. Each dimer contains two RLC nodes. The input is injected as a current at $V_1$, and the readout uses time samples of $V_1$ and $V_2$. The branch supports single- and parallel-dimer simulations, exceptional-point (EP) design sweeps, 11 reservoir properties, seven benchmarks, and three applications covering MPSK, PAM4, and waveform classification.

## Environment and EP Table

Run the commands from the repository root. Use an existing Python environment or create one with NumPy and SciPy.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy
```

A design index is the **zero-based position of a data row in the CSV**. If you use a different table or reorder its rows, verify the design indices against the device geometry and bias. The runner accepts rows marked `status=candidate` and checks the EP and the small-signal stability of each operating point.

## Generate and Select an EP Table

Run the device sweep to generate both CSVs.

```bash
python -m src.run_passive_gqc_device_ep \
  --output results/passive_gqc/device_ep.csv
```

`device_ep.csv` records all attempted designs and their statuses. The generator also writes `device_ep_candidates.csv`, which contains the accepted designs sorted by `fr`. Set `ep_table` to that candidate file for the commands below.

```bash
ep_table=results/passive_gqc/device_ep_candidates.csv
```

To reproduce the previous extended sweep with its exact input table, use the copy archived by that run.

```bash
ep_table=results/pass_gqc_r2_extended/ep_table.csv
```

Every experiment runner saves a copy of the CSV it received as `ep_table.csv` in its output directory. The archived extended-sweep copy and the candidate file used for the current experiments are byte-for-byte identical. A newly generated candidate file may have a different row order or candidate set if the sweep settings change.

## Screening EP Candidates

The following command evaluates every row in the selected candidate table at its own EP, with both circuit modes and three seeds.

```bash
python -m src.run_properties_pass_gqc \
  --ep-table "$ep_table" \
  --xi2 1 --modes graphene matched_linear --seeds 42 43 44 \
  --n-dimers 1 --cycles-per-symbol 0.5 --smoke \
  --output results/pass_gqc/candidate_screen
```

The aggregate properties runner selects all rows when you omit `--design-indices`. The task-specific entry points default to Designs 37 and 5. This screen uses the shorter `--smoke` settings used in the original 112-candidate run. Use a fresh output directory for each run.

Review `summary.csv` and `sweep_summary.csv` to shortlist candidates. Keep the same candidate CSV when passing the shortlisted row indices to a follow-up benchmark or application sweep. The Design 37 and Design 5 values shown below refer to the EP table used in the current experiments.

## Quick Check

This smoke test runs delay-1 XOR at the EP of Design 37. Set `ep_table` in the same shell before running it.

```bash
python -m src.run_pass_gqc_xor --ep-table "$ep_table" \
  --design-indices 37 --xi2 1 --seeds 44 --modes graphene \
  --delays 1 --cycles-per-symbol 1 --smoke \
  --output results/pass_gqc/xor_smoke
```

## Operating Points and Architectures

$$
\xi_2 = \frac{R_{2,\mathrm{EP}}}{R_2},
\qquad
R_2 = \frac{R_{2,\mathrm{EP}}}{\xi_2}.
$$

$\xi_2=1$ is the EP reference for each design. Increasing $\xi_2$ decreases $R_2$ and increases the branch conductance. Each design has its own resistance for a given $\xi_2$.

| Design | `fr` | $R_{2,\mathrm{EP}}$ |
|---|---:|---:|
| 37 | 0.999753 | 152.921 Ω |
| 5 | 0.995292 | 1783.464 Ω |

- `--modes graphene matched_linear` compares the voltage-dependent GQC model with a constant capacitance and equivalent series resistance (ESR) matched at the bias point.
- `--n-dimers 1` uses a single dimer. Use `--n-dimers 2 --routing broadcast` to send the same input to both dimers.
- `--n-dimers 2 --routing interleaved --ordering age` routes the input to each dimer in turn. The other dimers continue to evolve between injections. The readout orders features by how recently each dimer received an input. Use `--ordering physical` to keep a fixed dimer order.
- `--cycles-per-symbol` sets the number of reference cycles per symbol or per interleaving slot within a waveform trial. To compare architectures, use the same EP table, seeds, timing, input amplitude, and data splits.

## Properties and Benchmarks

Run the full single-dimer sweep with the batch script.

```bash
EP_TABLE="$ep_table" bash run_pass_gqc_single.sh results/pass_gqc/single_sweep_new
```

The script sweeps Design 37 and Design 5 in both modes with seeds 42, 43, and 44 and $\xi_2$ values `0.5 1 2 4 8 12`. By default, XOR and parity use one reference cycle per symbol. The other tasks use half a cycle. The script accepts the environment variables `EP_TABLE`, `PYTHON_BIN`, `PROPERTY_CYCLES`, `BENCHMARK_CYCLES`, and `XOR_PARITY_CYCLES`.

You can also run individual tasks with `python -m src.run_pass_gqc_<task>`.

| Category | Task names |
|---|---|
| Properties | `memory_capacity`, `memory_capacity_null`, `nonlinear_memory_capacity`, `nonlinear_memory_capacity_null`, `cross_memory_capacity`, `cross_memory_capacity_null`, `kernel_rank`, `generalization_rank`, `effective_dimension`, `voltage_decay`, `fading_memory_convergence` |
| Benchmarks | `xor`, `parity`, `sine_cube`, `narma10`, `henon`, `rossler`, `mackey_glass` |

The following command runs delay-1 XOR with two interleaved dimers.

```bash
python -m src.run_pass_gqc_xor --ep-table "$ep_table" \
  --design-indices 37 5 --xi2 0.5 1 2 4 8 12 --seeds 42 43 44 \
  --delays 1 --cycles-per-symbol 1 \
  --n-dimers 2 --routing interleaved --ordering age \
  --output results/pass_gqc/xor_d1_parallel_new
```

The benchmark entry points default to an 80% training, 10% validation, and 10% test split after washout. They select operating points on the validation set and evaluate the selected points on the held-out test set. Each property reports its metric across operating points.

## Applications

The application batch script runs all three applications in single-dimer mode. It uses the archived table at `results/pass_gqc_r2_extended/ep_table.csv` and writes to `results/pass_gqc/applications_single/`. If a task directory already contains results, use its individual entry point with a new output path.

```bash
bash run_pass_gqc_applications.sh
```

| Entry point | Tasks and options |
|---|---|
| `src.run_pass_gqc_mpsk` | 4-, 8-, 16-, and 32-PSK. Use `--psk-orders` and `--conditions clean awgn isi` to choose the orders and channel conditions. AWGN defaults to 20 dB and can be changed with `--snr-db`. The ISI model includes a pulse from one preceding random symbol and defaults to `--isi-tap 0.5 --isi-spacing-cycles 2`. |
| `src.run_pass_gqc_pam4` | Use `--pam4-channels linear nonlinear` to select the channel. Both modes apply linear ISI. The `nonlinear` mode adds nonlinear distortion. Adjust the settings with `--distortion-strength` and `--pam4-snr-db`. |
| `src.run_pass_gqc_waveform_classification` | Classifies sine, square, triangle, and sawtooth waves. Use `--experiments A B C D E` to select combinations of phase, amplitude, frequency, and noise. |

All three entry points accept the same design, $\xi_2$, mode, seed, and routing options. The following command runs single-dimer QPSK.

```bash
python -m src.run_pass_gqc_mpsk --ep-table "$ep_table" \
  --psk-orders 4 --conditions clean awgn isi \
  --output results/pass_gqc/qpsk_single
```

MPSK and waveform classification reset the circuit state for each trial and use separate random streams for training, validation, and testing. In clean MPSK, repeated trials of the same class use the same fixed waveform. PAM4 uses a continuous symbol sequence and excludes symbols affected by ISI across split boundaries. Applications report both reservoir and input-only readout results.

## Output Files

| File | Contents |
|---|---|
| `config.json`, `ep_table.csv` | Run parameters, source hashes, and a copy of the EP table used for the run |
| `summary.csv` | Results for each seed and operating point, including the actual $R_2$, circuit mode, and run status |
| `sweep_summary.csv` | Means and standard deviations across seeds for each operating point |
| `selected_test.csv` | Validation-selected settings and their held-out test results for benchmarks and applications |
| Case subdirectories | Features, predictions, confusion matrices, or detailed property measurements, depending on the task |

Use a separate output directory for each full sweep to retain previous results. The single-dimer batch writes logs under `logs/` in its output directory. The property, benchmark, and application entry points support `--help` and `--smoke`.

## ADS Waveform Comparison

```bash
python -m src.run_passive_gqc_ads_basic
```

The tool defaults to the matched-linear EP setting for Design 37. It exports the piecewise-linear input current and raw node voltages to `results/ads_compare/basic/`. The columns in `input.csv` are `time_s,i_in_a`. The columns in `output.csv` are `time_s,v1_v,v2_v`. The default input and output sampling intervals are 2 ps and 20 ps, respectively. To export at 100 ps, set `output_dt_ps` to 100 in `src/run_passive_gqc_ads_basic.py`.

## Source Layout

- `src/circuits/passive_gqc.py` and `src/circuits/passive_gqc_parallel.py` implement the single-dimer dynamics and parallel input routing.
- `src/pass_gqc/` handles case loading, simulation, sweeps, evaluation, and output.
- `src/tasks/` provides task data generators and metrics.
- Planned multi-dimer tasks include phase- and amplitude-encoded integration and the Fourier transform.
