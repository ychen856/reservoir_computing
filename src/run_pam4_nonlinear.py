"""Nonlinear PAM4: single-symbol and full temporal-window ridge baselines.

Place this file in src/run_pam4_nonlinear.py and run:
    python3 -m src.run_pam4_nonlinear

The reservoir is simulated ONCE. All methods use the same target indices,
with future observation windows constrained to stay within each data split.
"""
from pathlib import Path
import csv
import numpy as np

from src.circuits.pt_single import PTSingleReservoir
from src.tasks.applications.pam4_nonlinear import generate_pam4_nonlinear_task
from src.simulation.runner import simulate_continuous_task

N_SYMBOLS = 3000
WASHOUT = 200
TRAIN_RATIO = 0.8
FC = 48e6
N_CYCLES_PER_SYMBOL = 1 / 2
T_SYMBOL = N_CYCLES_PER_SYMBOL / FC
I_SCALE = 1e-4
K = 8
POINTS_PER_SYMBOL = 16
RIDGE = 1e-6
ALPHA = 8.33
SEED = 42
ADD_AWGN = False
SNR_DB = 40.0
NOISE_SEED = 123
NL_QUAD = 0.036
NL_CUBIC = -0.011
CHANNEL_RADIUS = 2
L_LIST = (0, 4, 8, 12, 16, 20)
OUTPUT_CSV = Path('pam4_nonlinear_temporal_sweep.csv')


def build_future_window(X, L):
    """Row n = [X[n], X[n+1], ..., X[n+L]]; invalid tail is NaN."""
    N, D = X.shape
    out = np.full((N, D * (L + 1)), np.nan, dtype=float)
    for offset in range(L + 1):
        out[:N - L, offset * D:(offset + 1) * D] = X[offset:N - L + offset]
    return out


def evaluate(X, y, train_idx, test_idx, ridge=RIDGE):
    """Scalar ridge with intercept, train-only z-score, and PAM4 quantization."""
    a, b = np.asarray(X[train_idx], float), np.asarray(X[test_idx], float)
    yt, yv = np.asarray(y[train_idx], float), np.asarray(y[test_idx], float)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError('Nonfinite features: check window bounds')
    mu, std = a.mean(axis=0), a.std(axis=0)
    std[std < 1e-12] = 1.0
    a, b = (a - mu) / std, (b - mu) / std
    # Intercept is unregularized; centered y keeps ridge well conditioned.
    ymean = yt.mean()
    gram = a.T @ a
    coef = np.linalg.solve(gram + ridge * np.eye(a.shape[1]), a.T @ (yt - ymean))
    pred_train, pred_test = a @ coef + ymean, b @ coef + ymean

    def score(truth, pred):
        labels = np.clip(np.rint(pred), 0, 3).astype(int)
        acc = float(np.mean(labels == truth))
        denom = np.std(truth)
        err = float(np.sqrt(np.mean((truth - pred) ** 2)) / denom) if denom > 0 else np.nan
        return acc, err

    train_acc, train_nrmse = score(yt, pred_train)
    test_acc, test_nrmse = score(yv, pred_test)
    return dict(feature_dim=X.shape[1], n_train=len(train_idx), n_test=len(test_idx),
                train_accuracy=train_acc, test_accuracy=test_acc,
                train_nrmse=train_nrmse, test_nrmse=test_nrmse,
                train_ser=1-train_acc, test_ser=1-test_acc)


def main():
    task = generate_pam4_nonlinear_task(
        n_symbols=N_SYMBOLS, seed=SEED, washout=WASHOUT, train_ratio=TRAIN_RATIO,
        quadratic=NL_QUAD, cubic=NL_CUBIC, add_awgn=ADD_AWGN,
        snr_db=SNR_DB, noise_seed=NOISE_SEED, channel_radius=CHANNEL_RADIUS,
    )
    u = np.asarray(task.u, float)
    reservoir = PTSingleReservoir(eta=1.0, alpha=ALPHA)
    sim = simulate_continuous_task(
        reservoir=reservoir, task_data=task, symbol_period=T_SYMBOL,
        points_per_symbol=POINTS_PER_SYMBOL, k=K,
        amplitude_scale=I_SCALE, encoding='zoh',
    )
    X = np.asarray(sim.X, float)
    y = np.asarray(sim.y, int)
    if X.shape != (len(y), 2*K):
        raise ValueError(f'Unexpected feature shape: {X.shape}; expected {(len(y),2*K)}')

    # The same split and samples are used for every row in the table.
    # Future windows must not cross the train/test boundary or the final channel edge.
    valid_start = max(WASHOUT, CHANNEL_RADIUS)
    valid_stop = len(y) - CHANNEL_RADIUS
    all_indices = np.arange(valid_start, valid_stop)
    boundary_pos = int(len(all_indices) * TRAIN_RATIO)
    train_boundary = int(all_indices[boundary_pos])
    max_L = max(L_LIST)
    train_idx = all_indices[:boundary_pos]
    train_idx = train_idx[train_idx + max_L < train_boundary]
    test_idx = all_indices[boundary_pos:]
    test_idx = test_idx[test_idx + max_L < valid_stop]
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError('Not enough samples for the selected maximum window')

    print(f'T_SYMBOL={T_SYMBOL*1e9:.4f} ns | dt={T_SYMBOL/POINTS_PER_SYMBOL*1e12:.3f} ps')
    print(f'K={K}, X={X.shape}, AWGN={ADD_AWGN}, nonlinear=q+{NL_QUAD}q²{NL_CUBIC:+}q³')
    print(f'Common train/test sizes: {len(train_idx)}/{len(test_idx)}')
    print(f'Boundary index: {train_boundary}; maximum future window: {max_L}')

    # Centered 5-tap equalizer. All methods predict the same symbol y[n].
    X_5tap = np.full((len(y), 5), np.nan)
    for offset in range(-2, 3):
        valid_n = np.arange(CHANNEL_RADIUS, len(y)-CHANNEL_RADIUS)
        X_5tap[valid_n, offset+2] = u[valid_n+offset]

    experiments = [('input_only', None, u.reshape(-1, 1)),
                   ('linear_5tap', None, X_5tap)]
    for L in L_LIST:
        experiments.append(('pt_full_window', L, build_future_window(X, L)))

    rows = []
    print(f"{'Method':<19} {'L':>3} {'Dim':>5} {'Train acc':>11} {'Test acc':>10} {'Test NRMSE':>11}")
    print('-'*69)
    for method, L, features in experiments:
        metrics = evaluate(features, y, train_idx, test_idx)
        row = dict(method=method, L='' if L is None else L,
                   latency_ns=0.0 if L is None else L*T_SYMBOL*1e9, **metrics)
        rows.append(row)
        print(f"{method:<19} {str(row['L']):>3} {metrics['feature_dim']:>5} "
              f"{metrics['train_accuracy']:>10.2%} {metrics['test_accuracy']:>9.2%} "
              f"{metrics['test_nrmse']:>11.6f}")

    with OUTPUT_CSV.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nSaved: {OUTPUT_CSV.resolve()}')
    print('Note: latency_ns is additional look-ahead relative to the current symbol.')


if __name__ == '__main__':
    main()
