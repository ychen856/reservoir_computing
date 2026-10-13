import numpy as np
from src.tasks.base import TaskData


def generate_sine_cube(
    n_samples=2000,
    washout=100,
    train_ratio=0.8,
):

    t = np.linspace(0, 20 * np.pi, n_samples)

    u = np.sin(t)
    y = u ** 3

    return TaskData(
        name="sine_cube",
        u=u,
        y=y,
        task_type="regression",
        washout=washout,
        train_ratio=train_ratio,
        target_offset=0,
    )