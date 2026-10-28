"""Controlled waveform classification experiments A-E.

Run from repository root: python3 -m src.run_waveform_classification
For quick smoke test: python3 -m src.run_waveform_classification --experiments A --n-per-class 8
"""
import argparse
import csv
from pathlib import Path
import numpy as np

from src.tasks.applications.waveform_classification import (
    EXPERIMENTS, LABELS, generate_waveform_classification,
)
from src.tasks.base import TaskData
from src.task_evaluation import evaluate, stratified_indices
from src._remaining_runner import CONFIGS, INPUT_SCALE, K, RIDGE
from src.circuits.pt_single import PTSingleReservoir
from src.simulation.runner import simulate_independent_trials

OUT = Path('results/remaining_tasks/waveform_ablation')


def _write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _plot_inputs(task, experiment):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(9, 5), sharex=True)
    t = task.metadata['t_eval']*1e6
    for c, ax in enumerate(axes.flat):
        examples = np.flatnonzero(task.y == c)[:3]
        for i in examples:
            ax.plot(t, task.u[i], linewidth=1, alpha=0.75)
        ax.set_title(LABELS[c]); ax.grid(alpha=0.3)
        ax.set_xlabel('Time (us)'); ax.set_ylabel('Normalized amplitude')
    fig.suptitle(f'Experiment {experiment}: input waveforms')
    fig.tight_layout()
    fig.savefig(OUT/f'{experiment}_input_examples.png', dpi=180)
    plt.close(fig)


def _features_input(task):
    # Fair low-dimensional temporal baseline: K samples, same as the number
    # of temporal nodes per reservoir voltage (reservoir has 2*K features).
    ix = np.linspace(0, task.u.shape[1]-1, K, dtype=int)
    # Full input waveform baseline diagnoses loss from temporal downsampling.
    # FFT magnitude is phase-insensitive, diagnostic only (nonlinear transform).
    return {
        f'Input {K} temporal samples': task.u[:, ix],
        'Input full temporal samples': task.u,
        'Input FFT magnitude (diagnostic)': np.abs(np.fft.rfft(task.u, axis=1))[:, 1:],
    }


def _reservoir_representations(sim, k):
    """Reuse the SAME ODE trajectories for all readout representations.

    Every trial starts from zero initial conditions (independent trials).
    Voltage order is [V1(t), V2(t)] at each time point.
    """
    trajectories = []
    for sol in sim.solutions:
        if not getattr(sol, 'success', False):
            raise RuntimeError('Reservoir simulation failed')
        v = np.asarray(sol.y[:2], dtype=float).T
        if v.ndim != 2 or v.shape[1] != 2 or not np.isfinite(v).all():
            raise ValueError('Invalid reservoir voltage trajectory')
        trajectories.append(v)
    V = np.stack(trajectories, axis=0)  # [trials, time, 2]
    # FFT magnitude per voltage channel, excluding DC; no nonlinear readout.
    spectra = np.abs(np.fft.rfft(V, axis=1))[:, 1:, :]
    return {
        'PT reservoir': sim.X,
        'PT full temporal samples': V.reshape(len(V), -1),
        'PT FFT magnitude (diagnostic)': spectra.reshape(len(V), -1),
    }


def run(experiments, n_per_class=40, seed=42, n_points=128):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for exp in experiments:
        opts = EXPERIMENTS[exp]
        task = generate_waveform_classification(
            n_per_class=n_per_class, n_points=n_points, seed=seed, **opts)
        train, test = stratified_indices(task.y, task.train_ratio, seed=123)
        _plot_inputs(task, exp)
        methods = _features_input(task)
        # Input baselines are evaluated once per experiment, not repeated for
        # each reservoir configuration.
        for method, X in methods.items():
            result = evaluate(X, task.y, train, test, 'classification', RIDGE)
            rows.append(dict(experiment=exp, eta='', alpha='', method=method,
                             train=result['train'], test=result['test'],
                             n_train=len(train), n_test=len(test)))
            np.savetxt(OUT/f'{exp}_{method.replace(" ", "_").replace("/", "_")}_confusion.csv',
                       result['confusion'], delimiter=',', fmt='%d')
            print(f'{exp} {method:34s} train={result["train"]:.4f} test={result["test"]:.4f}')

        for eta, alpha in CONFIGS:
            reservoir = PTSingleReservoir(eta=eta, alpha=alpha)
            scaled = TaskData(task.name, task.u*INPUT_SCALE, task.y,
                              task.task_type, train_ratio=task.train_ratio,
                              metadata=task.metadata)
            sim = simulate_independent_trials(reservoir, scaled, k=K)
            for method, X in _reservoir_representations(sim, K).items():
                result = evaluate(X, task.y, train, test, 'classification', RIDGE)
                rows.append(dict(experiment=exp, eta=eta, alpha=alpha,
                                 method=method, train=result['train'],
                                 test=result['test'], n_train=len(train), n_test=len(test)))
                safe_method = method.replace(' ', '_').replace('/', '_')
                np.savetxt(OUT/f'{exp}_PT_eta{eta:g}_alpha{alpha:g}_{safe_method}_confusion.csv',
                           result['confusion'], delimiter=',', fmt='%d')
                print(f'{exp} PT eta={eta:g} alpha={alpha:g} {method:34s} '
                      f'train={result["train"]:.4f} test={result["test"]:.4f}')
        _write_csv(OUT/f'{exp}_summary.csv', [r for r in rows if r['experiment'] == exp])
    _write_csv(OUT/'all_experiments_summary.csv', rows)
    print('Saved', OUT/'all_experiments_summary.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', nargs='+', choices=list(EXPERIMENTS),
                        default=list(EXPERIMENTS))
    parser.add_argument('--n-per-class', type=int, default=40)
    parser.add_argument('--n-points', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run(args.experiments, args.n_per_class, args.seed, args.n_points)
