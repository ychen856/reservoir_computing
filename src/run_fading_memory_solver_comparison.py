"""Compare nonlinear PT solver paths under identical piecewise-constant input.

Place in src/run_fading_memory_solver_comparison.py and run:
    python3 -m src.run_fading_memory_solver_comparison

NOTE: For alpha != 0, 'exact_zoh' is piecewise DOP853, NOT an exact matrix exponential.
The reference uses tighter-tolerance piecewise DOP853; this checks numerical
convergence, not agreement of independent integration algorithms.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from src.circuits.pt_single import PTSingleReservoir

ETAS = (0.0, 1.0, 1.05)
ALPHA = 8.33
FC = 48e6
TS = 0.5 / FC
N_SYMBOLS = 480
POINTS_PER_SYMBOL = 8
INPUT_SCALES = (0.0, 1e-6, 1e-5, 2e-5, 3e-5, 5e-5, 7e-5, 1e-4, 1e-3)
SEED = 42
X0_A = np.zeros(4)
X0_B = np.array([0.1, -0.1, 0.0, 0.0])
STATE_SCALES = np.array([1.0, 1.0, 1e-3, 1e-3])
OUTPUT = Path('results/fading_memory_solver_comparison')


def piecewise_ivp_reference(tank, values, t_eval, x0, rtol=1e-11, atol=1e-14):
    """Independent per-symbol integration with tight tolerance and fixed ZOH input."""
    out = np.empty((len(t_eval), 4))
    x = np.asarray(x0, dtype=float).copy()
    for k, value in enumerate(values):
        a, b = k * TS, (k + 1) * TS
        i0, i1 = k * POINTS_PER_SYMBOL, (k + 1) * POINTS_PER_SYMBOL
        local_t = t_eval[i0:i1]
        def rhs(t, state):
            return tank.tank.nonlinear_rhs(t, state, tank.eta, tank.alpha,
                                           lambda _: float(value))
        sol = solve_ivp(rhs, (a, b), x, method='DOP853', dense_output=True,
                        rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f'Reference IVP failed at symbol {k}: {sol.message}')
        out[i0:i1] = sol.sol(local_t).T
        x = sol.y[:, -1].copy()
    return out


def distance(a, b):
    d = a - b
    return np.linalg.norm(d[:, :2], axis=1), np.linalg.norm(d / STATE_SCALES, axis=1)


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    t_eval = np.arange(N_SYMBOLS * POINTS_PER_SYMBOL) * TS / POINTS_PER_SYMBOL
    base = np.random.default_rng(SEED).uniform(-1.0, 1.0, N_SYMBOLS)
    rows = []
    for eta in ETAS:
        tank = PTSingleReservoir(eta=eta, alpha=ALPHA)
        for scale in INPUT_SCALES:
            values = scale * base
            def current(t):
                k = np.clip(int(np.floor(t / TS)), 0, N_SYMBOLS - 1)
                return float(values[k])
            trajectories = {}
            for label, x0 in (('A', X0_A), ('B', X0_B)):
                zoh = tank.simulate(current, (0.0, N_SYMBOLS * TS), t_eval,
                                    x0=x0.copy(), solver='exact_zoh', symbol_period=TS)
                trajectories[f'zoh_{label}'] = np.asarray(zoh.y).T
                trajectories[f'ref_{label}'] = piecewise_ivp_reference(tank, values, t_eval, x0)
            dz, sz = distance(trajectories['zoh_A'], trajectories['zoh_B'])
            dr, sr = distance(trajectories['ref_A'], trajectories['ref_B'])
            ea, _ = distance(trajectories['zoh_A'], trajectories['ref_A'])
            eb, _ = distance(trajectories['zoh_B'], trajectories['ref_B'])
            tail = slice(int(0.9 * len(t_eval)), None)
            row = dict(eta=eta, alpha=ALPHA, input_scale_A=scale,
                       zoh_final_DV=dz[-1], reference_final_DV=dr[-1],
                       zoh_tail_rms_DV=rms(dz[tail]), reference_tail_rms_DV=rms(dr[tail]),
                       zoh_tail_rms_Dstate=rms(sz[tail]), reference_tail_rms_Dstate=rms(sr[tail]),
                       solver_voltage_rmse_A=rms(ea), solver_voltage_rmse_B=rms(eb),
                       solver_voltage_max_A=float(ea.max()), solver_voltage_max_B=float(eb.max()))
            rows.append(row)
            print(f"eta={eta:.2f} scale={scale:.1e} | "
                  f"tail ZOH={row['zoh_tail_rms_DV']:.4e} "
                  f"ref={row['reference_tail_rms_DV']:.4e} | "
                  f"solver RMSE A={row['solver_voltage_rmse_A']:.3e} "
                  f"B={row['solver_voltage_rmse_B']:.3e}")
            np.savez_compressed(OUTPUT / f'trajectories_eta_{eta:.2f}_scale_{scale:.0e}.npz',
                                t=t_eval, input_values=values, **trajectories,
                                zoh_voltage_distance=dz, reference_voltage_distance=dr,
                                zoh_scaled_state_distance=sz, reference_scaled_state_distance=sr)
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.semilogy(t_eval * 1e6, np.maximum(dz, 1e-15), label='existing exact_zoh (piecewise DOP853)')
            ax.semilogy(t_eval * 1e6, np.maximum(dr, 1e-15), '--', label='tight-tolerance piecewise IVP')
            ax.set(xlabel='Time [µs]', ylabel='Voltage distance [V]',
                   title=f'Initial-state convergence: η={eta}, input={scale:.1e} A')
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(OUTPUT / f'comparison_eta_{eta:.2f}_scale_{scale:.0e}.png', dpi=180)
            plt.close(fig)
    with (OUTPUT / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nSaved to {OUTPUT}')
    print('Finite-time convergence only; not proof of ESP or fading memory.')


if __name__ == '__main__':
    main()
