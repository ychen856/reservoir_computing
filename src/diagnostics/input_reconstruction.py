"""Delay-wise linear input reconstruction, with a shared chronological split."""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


def evaluate_input_reconstruction(X, u, washout, train_ratio=0.8,
                                  max_delay=20, ridge=1e-6):
    X = np.asarray(X, dtype=float)
    u = np.asarray(u, dtype=float).reshape(-1)
    if X.ndim != 2 or len(X) != len(u):
        raise ValueError('X must be (n_symbols, n_features), u must be (n_symbols,)')
    if not (0 < train_ratio < 1) or max_delay < 0 or washout < 0:
        raise ValueError('Invalid train_ratio, max_delay, or washout')
    start = max(int(washout), int(max_delay))
    n = len(u)
    indices = np.arange(start, n)
    n_train = int(len(indices) * train_ratio)
    if n_train < 2 or len(indices) - n_train < 2:
        raise ValueError('Not enough train/test samples')
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    scaler = StandardScaler().fit(X[train_idx])
    X_train = scaler.transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])
    # Baseline is a linear regression on the current u_n only.
    current_scaler = StandardScaler().fit(u[train_idx, None])
    current_train = current_scaler.transform(u[train_idx, None])
    current_test = current_scaler.transform(u[test_idx, None])

    rows = []
    for d in range(max_delay + 1):
        y_train = u[train_idx - d]
        y_test = u[test_idx - d]
        model = Ridge(alpha=ridge, fit_intercept=True).fit(X_train, y_train)
        baseline = Ridge(alpha=ridge, fit_intercept=True).fit(current_train, y_train)
        rows.append({
            'delay': d,
            'reservoir_train_r2': float(r2_score(y_train, model.predict(X_train))),
            'reservoir_test_r2': float(r2_score(y_test, model.predict(X_test))),
            'current_input_train_r2': float(r2_score(y_train, baseline.predict(current_train))),
            'current_input_test_r2': float(r2_score(y_test, baseline.predict(current_test))),
        })
    return rows, {'n_train': len(train_idx), 'n_test': len(test_idx),
                  'first_index': int(start), 'first_test_index': int(test_idx[0])}


def save_diagnostic(rows, out_dir, title):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / 'reconstruction_scores.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    d = [r['delay'] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(d, [r['reservoir_train_r2'] for r in rows], 'o-', ms=3, label='PT train')
    ax.plot(d, [r['reservoir_test_r2'] for r in rows], 's-', ms=3, label='PT test')
    ax.plot(d, [r['current_input_test_r2'] for r in rows], '--', label='Current-input baseline test')
    ax.axhline(0, color='gray', lw=0.8)
    ax.set(xlabel='Input delay (symbols)', ylabel=r'$R^2$', title=title)
    ax.set_xticks(range(0, max(d) + 1, 2))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'reconstruction_vs_delay.png', dpi=220)
    plt.close(fig)
