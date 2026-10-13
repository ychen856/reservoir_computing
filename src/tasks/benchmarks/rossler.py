import numpy as np
from scipy.integrate import solve_ivp

from src.tasks.base import TaskData


def generate_rossler(
    n_samples=3000,
    dt=0.1,
    transient=1000,
    a=0.2,
    b=0.2,
    c=5.7,
    x0=(1.0, 1.0, 1.0),
    washout=200,
    train_ratio=0.8,
):
    """
    Generate Rössler observer benchmark.

    Dynamics:
        dx/dt = -y - z
        dy/dt = x + a*y
        dz/dt = b + z*(x-c)

    Task:
        input  = x[n]
        target = [y[n], z[n]]

    Only the physical reservoir input is normalized.
    Targets remain in their original scales.
    """

    n_total = transient + n_samples

    t_eval = np.arange(
        n_total,
        dtype=float,
    ) * dt

    def rhs(t, s):
        x, y, z = s

        return [
            -y - z,
            x + a * y,
            b + z * (x - c),
        ]

    sol = solve_ivp(
        rhs,
        (t_eval[0], t_eval[-1]),
        x0,
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-11,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    xyz = sol.y[:, transient:].T

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    # Normalize physical reservoir input only.
    input_scale = np.max(np.abs(x))

    if input_scale == 0:
        raise ValueError("Rössler input has zero amplitude")

    u = x / input_scale

    Y = np.column_stack([y, z])

    return TaskData(
        name="rossler",
        u=u,
        y=Y,
        task_type="regression",
        washout=washout,
        train_ratio=train_ratio,
        target_offset=0,
        metadata={
            "a": a,
            "b": b,
            "c": c,
            "dt": dt,
            "transient": transient,
            "input_scale": input_scale,
            "raw_x": x,
            "simulation_mode": "continuous_sequence",
        },
    )