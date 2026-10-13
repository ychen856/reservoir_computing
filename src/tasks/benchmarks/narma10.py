import numpy as np
from src.tasks.base import TaskData


import numpy as np

from src.tasks.base import TaskData


def generate_narma10(
    n_samples=3000,
    washout=200,
    train_ratio=0.8,
    seed=42,
):
    """
    Generate NARMA-10 benchmark.

    Input:
        u[n] ~ Uniform(0, 0.5)

    Target:
        reservoir state at symbol n -> y[n+1]
    """

    if n_samples <= washout + 2:
        raise ValueError("n_samples must exceed washout + 2")

    rng = np.random.default_rng(seed)

    u = rng.uniform(
        0.0,
        0.5,
        size=n_samples,
    )

    y = np.zeros(n_samples + 1)

    for n in range(9, n_samples):

        y[n + 1] = (
            0.3 * y[n]
            + 0.05 * y[n] * np.sum(y[n-9:n+1])
            + 1.5 * u[n-9] * u[n]
            + 0.1
        )

    return TaskData(
        name="narma10",
        u=u,
        y=y[1:],
        task_type="regression",
        washout=washout,
        train_ratio=train_ratio,
        target_offset=1,
        metadata={
            "seed": seed,
            "order": 10,
            "simulation_mode": "continuous_sequence",
        },
    )