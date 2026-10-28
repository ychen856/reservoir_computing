import numpy as np
from src.tasks.base import TaskData


def generate_parity(n_samples=3000, window=5, washout=200, train_ratio=0.8, seed=42):
    """Parity of the most recent `window` bits including current bit."""
    if window < 2 or n_samples <= washout + window + 10:
        raise ValueError('Invalid window or sample count')
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, n_samples, dtype=np.int64)
    y = np.zeros(n_samples, dtype=np.int64)
    for d in range(window):
        y[d:] ^= bits[:n_samples-d]
    return TaskData('Parity', 2 * bits.astype(float) - 1, y, 'classification',
                    washout=max(washout, window - 1), train_ratio=train_ratio,
                    metadata={'window': window, 'seed': seed})
