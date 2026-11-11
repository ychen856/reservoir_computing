"""Passive-GQC dimer and array properties, reusing the existing metric implementations."""
import numpy as np

from src.pass_gqc.common import parser, validate, load_cases, simulate, symbol_period, Results, write_csv
from src.tasks.properties.suite import TASKS, memory
from src.tasks.properties.kernel_rank import evaluate_kernel_rank
from src.tasks.properties.generalization_rank import evaluate_generalization_rank
from src.tasks.properties.effective_dimension import evaluate_effective_dimension
from src.tasks.properties.voltage_decay import evaluate_voltage_decay
from src.utils.encoding import sample_hold_input


def rank(case, name, a, out):
    rng = np.random.default_rng(case.seed+2)
    base = rng.uniform(-1, 1, a.trial_symbols)
    features = []
    for _ in range(a.trials):
        independent = rng.uniform(-1, 1, a.trial_symbols)
        perturbed = np.clip(base+rng.normal(0, a.perturbation_std, a.trial_symbols), -1, 1)
        u = independent if name == "kernel_rank" else perturbed
        features.append(simulate(case, u, a).X[-1])
    X = np.asarray(features)
    evaluator = evaluate_kernel_rank if name == "kernel_rank" else evaluate_generalization_rank
    result = evaluator(X, relative_tolerance=a.rank_tolerance)
    np.savez_compressed(out.directory(case, name)/"spectrum.npz",
                        X=X, singular_values=result.singular_values)
    out.add(case, name, status="ok", rank=result.rank, feature_dim=X.shape[1],
            rank_ceiling=min(a.trials-1, X.shape[1]))


def convergence(case, a, out):
    reservoir = case.reservoir()
    ts = symbol_period(case, a)
    t = np.arange(a.convergence_symbols*a.points_per_symbol)*ts/a.points_per_symbol
    span = (0, a.convergence_symbols*ts)
    u = np.random.default_rng(case.seed).uniform(-1, 1, a.convergence_symbols)
    for forcing in ("zero", "driven"):
        drive = sample_hold_input(u if forcing == "driven" else np.zeros(len(u)), ts, a.amplitude)
        base = reservoir.simulate(drive, span, t, solver="exact_zoh", symbol_period=ts)
        for perturbation in ("common", "differential", "inductor", "internal"):
            initial = np.zeros(reservoir.state_size)
            if perturbation == "common":
                initial[:2] = [0.1, 0.1]
            elif perturbation == "differential":
                initial[:2] = [0.1, -0.1]
                initial[4] = 0.2
            elif perturbation == "inductor":
                initial[2:4] = [1e-3, -1e-3]
            else:
                initial[4] = 0.1
            initial = np.tile(initial[:5], case.n_dimers)
            solution = reservoir.simulate(drive, span, t, x0=initial,
                                           solver="exact_zoh", symbol_period=ts)
            difference = solution.y-base.y
            distance = np.linalg.norm(difference[list(reservoir.voltage_state_indices)], axis=0)
            # Dimensionless state norm uses voltage 1 V, current 1/Z A.
            z = np.sqrt(case.l_nh*1e-9/(case.c1_pf*1e-12))
            scales = np.tile([1, 1, 1/z, 1/z, 1], case.n_dimers)
            state_distance = np.linalg.norm(difference/scales[:, None], axis=0)
            tail = slice(int(0.9*len(t)), None)
            np.savez_compressed(out.directory(case, "fading_memory_convergence")/
                                f"{forcing}_{perturbation}.npz", t=t, states_A=base.y,
                                states_B=solution.y, voltage_distance=distance,
                                state_distance=state_distance)
            out.add(case, "fading_memory_convergence", status="ok", forcing=forcing,
                    perturbation=perturbation, final_voltage_distance=float(distance[-1]),
                    tail_rms_voltage=float(np.sqrt(np.mean(distance[tail]**2))),
                    final_state_distance=float(state_distance[-1]))


def decay(case, a, out):
    reservoir = case.reservoir()
    ts = symbol_period(case, a)
    # A finite ZOH pulse followed by an exactly zero-input ringdown.
    u = np.r_[np.ones(a.decay_drive_symbols), np.zeros(a.decay_symbols)]
    drive = sample_hold_input(u, ts, a.amplitude)
    t = np.arange(len(u)*a.points_per_symbol)*ts/a.points_per_symbol
    solution = reservoir.simulate(drive, (0, len(u)*ts), t,
                                   solver="exact_zoh", symbol_period=ts)
    cutoff = a.decay_drive_symbols*ts
    fit_start, fit_end = cutoff+ts, t[-1]
    np.savez_compressed(out.directory(case, "voltage_decay")/"waveforms.npz",
                        t=t, states=solution.y, fit_start=fit_start, fit_end=fit_end)
    for channel, state_index in enumerate(reservoir.voltage_state_indices, 1):
        result = evaluate_voltage_decay(t, solution.y[state_index], fit_start, fit_end)
        out.add(case, "voltage_decay", status="ok", channel=channel,
                tau_s=result.tau_seconds, slope=result.slope,
                fit_r_squared=result.r_squared, n_fit=result.n_fit)


def main(argv=None, *, task=None):
    p = parser("properties", TASKS, task)
    p.add_argument("--max-delay", type=int, default=50)
    p.add_argument("--nonlinear-max-delay", type=int, default=20)
    p.add_argument("--cross-max-delay", type=int, default=20)
    p.add_argument("--permutations", type=int, default=100)
    p.add_argument("--null-seed", type=int, default=2026)
    p.add_argument("--trials", type=int, default=80)
    p.add_argument("--trial-symbols", type=int, default=48)
    p.add_argument("--rank-tolerance", type=float, default=1e-3)
    p.add_argument("--perturbation-std", type=float, default=0.02)
    p.add_argument("--convergence-symbols", type=int, default=480)
    p.add_argument("--decay-drive-symbols", type=int, default=8)
    p.add_argument("--decay-symbols", type=int, default=480)
    a = p.parse_args(argv)
    validate(a)
    if a.smoke:
        a.max_delay, a.nonlinear_max_delay, a.cross_max_delay = 5, 5, 4
        a.permutations, a.trials, a.trial_symbols = 2, 8, 12
        a.convergence_symbols, a.decay_symbols = 48, 96
    counts = [a.max_delay, a.nonlinear_max_delay, a.permutations, a.trial_symbols,
              a.convergence_symbols, a.decay_drive_symbols, a.decay_symbols]
    if (min(counts) < 1 or a.cross_max_delay < 2 or a.trials < 2
            or not np.isfinite([a.rank_tolerance, a.perturbation_std]).all()
            or a.rank_tolerance <= 0 or a.perturbation_std < 0):
        p.error("Invalid property settings")
    maximum = max(a.max_delay, a.nonlinear_max_delay, a.cross_max_delay)
    if (a.symbols-max(a.washout, maximum))*(1-a.train_ratio) <= maximum+3:
        p.error("Insufficient held-out samples after delay purge")
    cases = load_cases(a)
    out = Results(a)
    for case in cases:
        cache = {}
        for name in a.tasks:
            try:
                if name in ("kernel_rank", "generalization_rank"):
                    rank(case, name, a, out)
                elif name == "fading_memory_convergence":
                    convergence(case, a, out)
                elif name == "voltage_decay":
                    decay(case, a, out)
                else:
                    distribution = "uniform" if name.startswith("nonlinear") or name == "effective_dimension" else "bipolar"
                    if distribution not in cache:
                        rng = np.random.default_rng(case.seed)
                        u = (rng.uniform(-1, 1, a.symbols) if distribution == "uniform"
                             else rng.choice([-1.0, 1.0], a.symbols))
                        X = simulate(case, u, a).X
                        cache[distribution] = (u, X)
                        np.savez_compressed(out.directory(case, "features")/f"{distribution}.npz", u=u, X=X)
                    u, X = cache[distribution]
                    if name == "effective_dimension":
                        result = evaluate_effective_dimension(X[a.washout:], standardize=False)
                        np.savez_compressed(out.directory(case, name)/"spectrum.npz", eigenvalues=result.eigenvalues)
                        out.add(case, name, status="ok", feature_dim=X.shape[1], dimension=result.dimension)
                    else:
                        memory(case, name, a, out, (u, X))
            except Exception as error:
                out.fail(case, name, error)
    out.finish()


if __name__ == "__main__":
    main()
