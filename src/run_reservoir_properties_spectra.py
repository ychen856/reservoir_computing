"""Physical PT reservoir property experiments.

Run from project root: python3 -m src.run_reservoir_properties
Existing src.run_property_validation is synthetic-only and remains unchanged.
"""
import csv
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.circuits.pt_single import PTSingleReservoir
from src.simulation.runner import simulate_continuous_task, validate_solution
from src.tasks.properties.nonlinear_memory import evaluate_nonlinear_memory
from src.tasks.properties.cross_memory import evaluate_cross_memory
from src.tasks.properties.kernel_rank import evaluate_kernel_rank
from src.tasks.properties.generalization_rank import evaluate_generalization_rank
from src.tasks.properties.effective_dimension import evaluate_effective_dimension
from src.tasks.properties.voltage_decay import evaluate_voltage_decay

# -------- Settings (small first pass; scale only after it succeeds) --------
FC = 48e6
TC = 1 / FC
TS_RATIO = 0.5
TS = TC * TS_RATIO
CONFIGS = [(0.0, 0.0), (1.0, 0.0), (1.0, 8.33)]
N_TRAIN = 1600
N_TEST = 1600
WASHOUT = 200
POINTS_PER_SYMBOL = 16
K = 8
INPUT_SCALE = 1e-2
RIDGE = 1e-6
MEMORY_DELAYS = tuple(range(1, 21))
CROSS_MAX_DELAY = 10
N_TRIALS = 24
TRIAL_SYMBOLS = 48
PERTURBATION_STD = 0.02
RANK_TOLERANCE = 1e-3
SEED = 42
OUTPUT = Path('results/reservoir_properties')


def simulate_symbols(reservoir, u):
    """Independent call => zero initial state; ZOH exact for alpha=0."""
    u = np.asarray(u, dtype=float)
    if u.ndim != 1 or len(u) < 2:
        raise ValueError('Expected 1D sequence with >=2 symbols')
    task = SimpleNamespace(u=u, y=np.zeros(len(u)))
    solver = 'exact_zoh' if reservoir.alpha == 0 else 'ivp'
    result = simulate_continuous_task(
        reservoir=reservoir, task_data=task, symbol_period=TS,
        points_per_symbol=POINTS_PER_SYMBOL, k=K,
        amplitude_scale=INPUT_SCALE, encoding='zoh', solver=solver,
    )
    if result.X.shape != (len(u), 2*K):
        raise RuntimeError(f'Unexpected feature shape: {result.X.shape}')
    return result


def trial_matrices(reservoir, rng):
    """One feature vector per independent, zero-state input-history trial.

    KR: independent histories. GR: one common history with small perturbations.
    Both use identical length and feature sampling, but different inputs.
    """
    base = rng.uniform(-1, 1, size=TRIAL_SYMBOLS)
    kr, gr = [], []
    for i in range(N_TRIALS):
        independent = rng.uniform(-1, 1, size=TRIAL_SYMBOLS)
        perturbed = np.clip(
            base + rng.normal(0, PERTURBATION_STD, size=TRIAL_SYMBOLS), -1, 1
        )
        kr.append(simulate_symbols(reservoir, independent).X[-1])
        gr.append(simulate_symbols(reservoir, perturbed).X[-1])
    return np.asarray(kr), np.asarray(gr)


def pulse_decay(reservoir):
    """Short, smooth current pulse followed by unforced relaxation.

    Uses IVP for a continuous pulse (exact_zoh is not applicable).
    Hilbert exponential fit is diagnostic only: multi-mode and growing
    responses may not have a meaningful single positive decay constant.
    """
    t_end = 1.5e-6
    n = 6000
    t_eval = np.linspace(0.0, t_end, n, endpoint=False)
    center, sigma = 0.12e-6, 0.035e-6
    def current(t):
        return INPUT_SCALE * np.exp(-0.5*((t-center)/sigma)**2) * np.cos(2*np.pi*FC*(t-center))
    sol = reservoir.simulate(
        input_func=current, t_span=(0.0, t_end), t_eval=t_eval,
        solver='ivp', symbol_period=None,
    )
    validate_solution(sol)
    rows = []
    for channel in (0, 1):
        try:
            res = evaluate_voltage_decay(
                sol.t, sol.y[channel],
                fit_start=0.3e-6, fit_end=1.25e-6,
            )
            rows.append((channel, res.tau_seconds, res.r_squared, res.slope))
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            print(f'  Decay fit V{channel+1} unavailable: {exc}')
            rows.append((channel, float('nan'), float('nan'), float('nan')))
    return rows


def append_rows(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Overwrite deliberately: each run is one reproducible complete experiment.
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    rng_train = np.random.default_rng(SEED)
    rng_test = np.random.default_rng(SEED + 1)
    rng_trials = np.random.default_rng(SEED + 2)
    all_rows = []
    detail_rows = []
    rank_rows = []
    for eta, alpha in CONFIGS:
        print(f'\n=== eta={eta:g} alpha={alpha:g} Ts/Tc={TS_RATIO:g} ===', flush=True)
        reservoir = PTSingleReservoir(eta=eta, alpha=alpha)
        # Independent streams, no train/test trajectory leakage.
        u_train = rng_train.uniform(-1, 1, N_TRAIN)
        u_test = rng_test.uniform(-1, 1, N_TEST)
        train = simulate_symbols(reservoir, u_train)
        test = simulate_symbols(reservoir, u_test)
        nm = evaluate_nonlinear_memory(
            train.X, u_train, test.X, u_test,
            delays=MEMORY_DELAYS, degrees=(2,3),
            washout=WASHOUT, ridge=RIDGE,
        )
        cm = evaluate_cross_memory(
            train.X, u_train, test.X, u_test,
            max_delay=CROSS_MAX_DELAY,
            washout=WASHOUT, ridge=RIDGE,
        )
        ed = evaluate_effective_dimension(train.X[WASHOUT:], standardize=False)
        print(f'  Nonlinear MC={nm.total_test:.5f}; Cross MC={cm.total_test:.5f}; d_eff={ed.dimension:.3f}',flush=True)
        for family, result in [('nonlinear', nm), ('cross', cm)]:
            for label, train_score, test_score in zip(result.labels,result.train_scores,result.test_scores):
                detail_rows.append(dict(eta=eta, alpha=alpha, ts_ratio=TS_RATIO,
                                        property=family, target=label,
                                        train_score=float(train_score),test_score=float(test_score)))
        kr_X, gr_X = trial_matrices(reservoir, rng_trials)
        kr = evaluate_kernel_rank(kr_X, relative_tolerance=RANK_TOLERANCE)
        gr = evaluate_generalization_rank(gr_X, relative_tolerance=RANK_TOLERANCE)
        print(f'  KR={kr.rank}; GR={gr.rank} (rank_tol={RANK_TOLERANCE})',flush=True)
        # Save the full singular-value spectrum, not only thresholded ranks.
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for family, result in [('KR', kr), ('GR', gr)]:
            singular = np.asarray(result.singular_values, dtype=float)
            scale = singular[0] if len(singular) and singular[0] > 0 else 1.0
            normalized = singular / scale
            for i, (sv, sv_norm) in enumerate(zip(singular, normalized), start=1):
                rank_rows.append(dict(
                    eta=eta, alpha=alpha, ts_ratio=TS_RATIO,
                    property=family, component=i,
                    singular_value=float(sv), normalized_singular_value=float(sv_norm),
                    rank_tolerance=RANK_TOLERANCE,
                    above_tolerance=bool(sv_norm > RANK_TOLERANCE),
                ))
            ax.semilogy(np.arange(1, len(normalized)+1),
                        np.maximum(normalized, 1e-16), marker='o', label=family)
            print(f'  {family} normalized singular values: '
                  + np.array2string(normalized, precision=3, suppress_small=False),
                  flush=True)
        ax.axhline(RANK_TOLERANCE, linestyle='--', color='gray',
                   label=f'Rank tolerance = {RANK_TOLERANCE:g}')
        ax.set_xlabel('Singular-value index')
        ax.set_ylabel('Normalized singular value')
        ax.set_title(f'KR/GR spectra: eta={eta:g}, alpha={alpha:g}')
        ax.set_xticks(np.arange(1, 2*K+1))
        ax.grid(True, which='both', alpha=0.3)
        ax.legend()
        fig.tight_layout()
        OUTPUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT / f'rank_spectrum_eta{eta:g}_alpha{alpha:g}.png', dpi=200)
        plt.close(fig)
        decays = pulse_decay(reservoir)
        for channel, tau, r2, slope in decays:
            print(f'  V{channel+1} tau={tau:.4g} s, fit R²={r2:.3f}',flush=True)
            all_rows.append(dict(eta=eta,alpha=alpha,ts_ratio=TS_RATIO,
                                 n_train=N_TRAIN,n_test=N_TEST,feature_dim=2*K,
                                 nonlinear_mc=nm.total_test,cross_mc=cm.total_test,
                                 effective_dimension=ed.dimension,kr=kr.rank,gr=gr.rank,
                                 rank_tol=RANK_TOLERANCE,perturbation_std=PERTURBATION_STD,
                                 decay_channel=channel,decay_tau_s=tau,decay_fit_r2=r2,
                                 decay_slope_per_s=slope))
    append_rows(OUTPUT/'property_summary.csv',list(all_rows[0]),all_rows)
    append_rows(OUTPUT/'memory_target_scores.csv',list(detail_rows[0]),detail_rows)
    append_rows(OUTPUT/'rank_singular_values.csv',list(rank_rows[0]),rank_rows)
    print(f'\nSaved {OUTPUT / "property_summary.csv"} and {OUTPUT / "memory_target_scores.csv"}')
    print(f'Saved {OUTPUT / "rank_singular_values.csv"} and rank-spectrum PNGs')
    print('Note: KR/GR are protocol-dependent; decay tau only interpretable for good decaying fits.')

if __name__ == '__main__':
    main()
