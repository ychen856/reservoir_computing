import numpy as np
from scipy.integrate import solve_ivp


def generate_rossler(n_samples=3000, dt=0.1, transient=1000,
                     a=0.2, b=0.2, c=5.7,
                     x0=(1.0, 1.0, 1.0)):
    """Generate uniformly sampled Rossler trajectory after discarding transient."""
    n_total = transient + n_samples
    t_eval = np.arange(n_total, dtype=float) * dt

    def rhs(t, s):
        x, y, z = s
        return [-y - z, x + a*y, b + z*(x - c)]

    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), x0,
                    t_eval=t_eval, rtol=1e-9, atol=1e-11)
    if not sol.success:
        raise RuntimeError(sol.message)

    xyz = sol.y[:, transient:].T
    t = t_eval[transient:] - t_eval[transient]
    return t, xyz[:, 0], xyz[:, 1], xyz[:, 2]
