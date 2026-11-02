"""PAM4 PT-reservoir benchmark: full, uniform, and greedy temporal readouts.

Run from the project root: python run_pam4_clean.py
Requires the existing src package. No extra ODE runs for feature comparisons.
"""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

from src.circuits.pt_single import PTSingleReservoir
from src.tasks.applications.pam4 import generate_pam4_task, build_channel_window, quantize_pam4_class
from src.utils.encoding import sample_hold_input
from src.utils.features import extract_temporal_features
from src.readout.ridge import train_ridge_regression, predict_ridge_regression
from src.readout.preprocessing import fit_standardizer, transform_standardizer
from src.readout.metrics import nrmse, accuracy, symbol_error_rate

# ----------------------------- Configuration -----------------------------
N_SYMBOLS = 3000
TRAIN_RATIO = 0.8
WASHOUT = 200
SEED = 42
FC = 48e6
N_CYCLES_PER_SYMBOL = 0.5
T_SYMBOL = N_CYCLES_PER_SYMBOL / FC
I_SCALE = 1e-4
K = 8
POINTS_PER_SYMBOL = 16
RIDGE = 1e-6
ALPHA = 8.33
ETA = 1.0
L_LIST = [20] #(0, 1, 2, 4, 8, 12, 16, 20)
GREEDY_L_LIST = (8, 12, 16, 20)
N_SELECTED_NODES = 32
SELECT_TRAIN_FRACTION = 0.8  # chronological inner split, training only
OUTPUT_DIR = Path('pam4_results')


def evaluate(X, y, train_idx, test_idx, name=''):
    """Scalar ridge + nearest-PAM4-level decoding; train-only standardization."""
    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = y[train_idx], y[test_idx]
    mu, std = fit_standardizer(Xtr)
    Xtr = transform_standardizer(Xtr, mu, std)
    Xte = transform_standardizer(Xte, mu, std)
    w = train_ridge_regression(Xtr, ytr, ridge=RIDGE)
    ptr = predict_ridge_regression(Xtr, w)
    pte = predict_ridge_regression(Xte, w)
    htr = quantize_pam4_class(ptr)
    hte = quantize_pam4_class(pte)
    return dict(name=name, feature_dim=X.shape[1],
                train_accuracy=float(accuracy(ytr, htr)),
                test_accuracy=float(accuracy(yte, hte)),
                train_nrmse=float(nrmse(ytr, ptr)),
                test_nrmse=float(nrmse(yte, pte)),
                train_ser=float(symbol_error_rate(ytr, htr)),
                test_ser=float(symbol_error_rate(yte, hte)))


def full_window(X, L):
    """Rows n contain the K-node states from symbols n through n+L."""
    N, D = X.shape
    out = np.full((N, D * (L + 1)), np.nan)
    for j in range(L + 1):
        out[:N - L, j * D:(j + 1) * D] = X[j:N - L + j]
    return out


def uniform_nodes(Xfull, count=N_SELECTED_NODES):
    """Select evenly spaced V1/V2 pairs, keeping both voltages at each node."""
    total = Xfull.shape[1] // 2
    nodes = np.rint(np.linspace(0, total - 1, count)).astype(int)
    return Xfull[:, node_columns(nodes)], nodes


def node_columns(nodes):
    nodes = np.asarray(nodes, dtype=int)
    return (2 * nodes[:, None] + np.array([0, 1])[None, :]).ravel()


def validation_mse(X, y, fit_idx, val_idx):
    """Use inner-training fit only; select nodes by inner-validation MSE."""
    mu, std = fit_standardizer(X[fit_idx])
    a = transform_standardizer(X[fit_idx], mu, std)
    b = transform_standardizer(X[val_idx], mu, std)
    w = train_ridge_regression(a, y[fit_idx], ridge=RIDGE)
    pred = predict_ridge_regression(b, w)
    return float(np.mean((pred - y[val_idx]) ** 2))


def greedy_nodes(Xfull, y, fit_idx, val_idx, count=N_SELECTED_NODES):
    """Forward select V1/V2 node pairs using *only* inner train/validation.

    Never uses held-out test targets. Selection objective: validation MSE.
    """
    n_nodes = Xfull.shape[1] // 2
    chosen = []
    remaining = set(range(n_nodes))
    for step in range(min(count, n_nodes)):
        best_node, best_loss = None, np.inf
        for node in sorted(remaining):
            columns = node_columns(chosen + [node])
            loss = validation_mse(Xfull[:, columns], y, fit_idx, val_idx)
            if loss < best_loss:
                best_node, best_loss = node, loss
        chosen.append(best_node)
        remaining.remove(best_node)
        print(f'    selected {step + 1}/{count}: node={best_node:3d}, '
              f'validation MSE={best_loss:.6f}')
    return np.array(chosen, dtype=int)


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Saved:', path)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    task = generate_pam4_task(n_symbols=N_SYMBOLS, seed=SEED,
                              washout=WASHOUT, train_ratio=TRAIN_RATIO)
    u, y = np.asarray(task.u), np.asarray(task.y)
    N = len(u)
    radius = int(task.metadata['channel_radius'])
    dt = T_SYMBOL / POINTS_PER_SYMBOL
    # Integer-grid sampling avoids np.arange floating point endpoint drift.
    t_eval = np.arange(N * POINTS_PER_SYMBOL) * dt
    print(f'N={N}, T_symbol={T_SYMBOL * 1e9:.4f} ns, '
          f'dt={dt * 1e12:.4f} ps, K={K}, alpha={ALPHA}')
    inp = sample_hold_input(u, symbol_period=T_SYMBOL,
                            amplitude_scale=I_SCALE)
    reservoir = PTSingleReservoir(eta=ETA, alpha=ALPHA)
    sol = reservoir.simulate(input_func=inp,
                             t_span=(0., N * T_SYMBOL), t_eval=t_eval)
    if not sol.success or not np.isfinite(sol.y).all():
        raise RuntimeError(f'Invalid simulation: {sol.message}')
    X = extract_temporal_features(sol, n_symbols=N,
                                  symbol_period=T_SYMBOL, k=K)
    if X.shape != (N, 2 * K):
        raise ValueError(f'Unexpected reservoir feature shape {X.shape}')
    print('X shape:', X.shape,
          'V1 range:', (sol.y[0].min(), sol.y[0].max()),
          'V2 range:', (sol.y[1].min(), sol.y[1].max()))

    valid = np.arange(max(task.washout, radius), N - radius)
    split = int(task.train_ratio * len(valid))
    train_idx, test_idx = valid[:split], valid[split:]
    boundary = int(test_idx[0])
    max_l = max(L_LIST)
    # Identical target samples for all methods; no future-state windows
    # from training targets are allowed to cross the test boundary.
    common_train = train_idx[train_idx + max_l < boundary]
    common_test = test_idx[test_idx + max_l < N - radius]
    if len(common_train) < 50 or len(common_test) < 50:
        raise ValueError('Too few samples after boundary protection')
    print('Common train/test samples:', len(common_train), len(common_test))

    # Baselines: identical train/test target indices.
    baseline_rows = []
    for name, Z in [('PT single-symbol', X),
                    ('Input only', u.reshape(-1, 1)),
                    ('5-tap channel window', build_channel_window(u, radius=radius))]:
        r = evaluate(Z, y, common_train, common_test, name)
        baseline_rows.append(r)
        print(f"{name:25s} dim={r['feature_dim']:3d} "
              f"test={r['test_accuracy']:.2%} NRMSE={r['test_nrmse']:.4f}")
    write_csv(OUTPUT_DIR / 'baselines.csv', baseline_rows)

    # Sweep: full-window versus uniform 8-node pair selection.
    sweep_rows = []
    for L in L_LIST:
        Z = full_window(X, L)
        Zfixed, nodes = uniform_nodes(Z)
        for method, features in [('full', Z), ('uniform_fixed', Zfixed)]:
            r = evaluate(features, y, common_train, common_test, method)
            row = dict(method=method, L=L, latency_ns=L * T_SYMBOL * 1e9,
                       selected_nodes=';'.join(map(str, nodes)) if method == 'uniform_fixed' else '',
                       **r)
            sweep_rows.append(row)
            print(f"L={L:2d} {method:14s} dim={r['feature_dim']:3d} "
                  f"test={r['test_accuracy']:.2%} NRMSE={r['test_nrmse']:.4f}")
    write_csv(OUTPUT_DIR / 'temporal_sweep.csv', sweep_rows)

    # Greedy node selection: use only the chronological inner training split.
    inner_split = int(len(common_train) * SELECT_TRAIN_FRACTION)
    fit_idx = common_train[:inner_split]
    val_idx = common_train[inner_split:]
    greedy_rows = []
    for L in GREEDY_L_LIST: 
        print(f'\nGreedy selection L={L} ...')
        Z = full_window(X, L)
        nodes = greedy_nodes(Z, y, fit_idx, val_idx)
        selected = Z[:, node_columns(nodes)]
        # Refit readout on the entire outer-training set, then test once.
        r = evaluate(selected, y, common_train, common_test, f'greedy L={L}')
        greedy_rows.append(dict(method='greedy_fixed', L=L,
                                latency_ns=L * T_SYMBOL * 1e9,
                                selected_nodes=';'.join(map(str, nodes)), **r))
        print(f"L={L:2d} greedy_fixed   dim={r['feature_dim']:3d} "
              f"test={r['test_accuracy']:.2%} NRMSE={r['test_nrmse']:.4f}")
    write_csv(OUTPUT_DIR / 'greedy_node_sweep.csv', greedy_rows)
    print('\nNote: select final L/hyperparameters by validation, not test accuracy.')


if __name__ == '__main__':
    main()
