import numpy as np
from src.tasks.base import TaskData


def generate_xor(n_samples=3000, delay=1, washout=200, train_ratio=0.8, seed=42):
    """y[k] = bit[k] XOR bit[k-delay]; input is bipolar {-1,+1}."""
    if delay < 1 or n_samples <= washout + delay + 10:
        raise ValueError('Invalid delay or sample count')
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, n_samples, dtype=np.int64)
    y = np.zeros(n_samples, dtype=np.int64)
    y[delay:] = np.bitwise_xor(bits[delay:], bits[:-delay])
    return TaskData('XOR', 2 * bits.astype(float) - 1, y, 'classification',
                    washout=max(washout, delay), train_ratio=train_ratio,
                    metadata={'delay': delay, 'seed': seed})
