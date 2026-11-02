"""Initial-condition convergence test for the PT single-tank reservoir.

Place at src/run_fading_memory_convergence.py and run:
    python3 -m src.run_fading_memory_convergence

This is a finite-time, two-initial-condition test, not a proof of the ESP.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.circuits.pt_single import PTSingleReservoir

ETAS = (0.0, 1.0, 1.05, 1.1)
ALPHA = 8.33
FC = 48e6
TS = 0.5 / FC
N_SYMBOLS = 480  # 5 us
POINTS_PER_SYMBOL = 8
INPUT_SCALE = 1e-4
SEED = 42
X0_A = np.zeros(4)
X0_B = np.array([0.1, -0.1, 0.0, 0.0])  # [V1,V2,IL1,IL2]
STATE_SCALES = np.array([1.0, 1.0, 1e-3, 1e-3])  # V, V, A, A
OUTPUT = Path('results/fading_memory_convergence')


def run_one(eta, mode, symbols):
    tank = PTSingleReservoir(eta=eta, alpha=ALPHA)
    t_end = N_SYMBOLS * TS
    t_eval = np.arange(N_SYMBOLS * POINTS_PER_SYMBOL) * (TS / POINTS_PER_SYMBOL)
    if mode == 'zero':
        def current(t):
            return 0.0
        solver = 'ivp'
    else:
        def current(t):
            k = min(max(int(np.floor(t / TS)), 0), N_SYMBOLS - 1)
            return INPUT_SCALE * symbols[k]
        solver = 'exact_zoh'  # nonlinear piecewise DOP853, not linear matrix exp

    kwargs = dict(input_func=current, t_span=(0.0, t_end), t_eval=t_eval,
                  solver=solver, symbol_period=TS)
    a = tank.simulate(x0=X0_A.copy(), **kwargs)
    b = tank.simulate(x0=X0_B.copy(), **kwargs)
    if not a.success or not b.success:
        raise RuntimeError(f'Simulation failed: eta={eta}, mode={mode}')
    ya = np.asarray(a.y).T
    yb = np.asarray(b.y).T
    if not (np.all(np.isfinite(ya)) and np.all(np.isfinite(yb))):
        raise RuntimeError(f'Nonfinite state: eta={eta}, mode={mode}')
    diff = ya - yb
    dv = np.linalg.norm(diff[:, :2], axis=1)
    dr = np.linalg.norm(diff / STATE_SCALES, axis=1)
    return t_eval, ya, yb, dv, dr


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    symbols = np.random.default_rng(SEED).uniform(-1.0, 1.0, N_SYMBOLS)
    rows = []
    all_results = {}
    for mode in ('zero', 'driven'):
        for eta in ETAS:
            print(f'\nRunning {mode}, eta={eta}')
            t, a, b, dv, dr = run_one(eta, mode, symbols)
            all_results[(mode, eta)] = (t, a, b, dv, dr)
            tail = slice(int(0.9 * len(t)), None)
            rows.append(dict(mode=mode, eta=eta, alpha=ALPHA,
                             duration_us=t[-1]*1e6, initial_DV_V=dv[0],
                             final_DV_V=dv[-1], tail_rms_DV_V=np.sqrt(np.mean(dv[tail]**2)),
                             initial_Dstate=dr[0], final_Dstate=dr[-1],
                             tail_rms_Dstate=np.sqrt(np.mean(dr[tail]**2))))
            np.savez(OUTPUT / f'trajectories_{mode}_eta_{eta:.2f}.npz',
                     t=t, states_A=a, states_B=b, voltage_distance=dv,
                     normalized_state_distance=dr)
            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            for ch in range(2):
                axes[ch].plot(t*1e6, a[:, ch], lw=0.7, label='Initial A')
                axes[ch].plot(t*1e6, b[:, ch], lw=0.7, alpha=0.75, label='Initial B')
                axes[ch].set_ylabel(f'V{ch+1} [V]')
                axes[ch].grid(alpha=0.3)
                axes[ch].legend()
            axes[2].semilogy(t*1e6, np.maximum(dv, 1e-15), label='Voltage distance [V]')
            axes[2].semilogy(t*1e6, np.maximum(dr, 1e-15), label='Scaled full-state distance')
            axes[2].set_ylabel('Distance (log scale)')
            axes[2].set_xlabel('Time [us]')
            axes[2].grid(alpha=0.3)
            axes[2].legend()
            fig.suptitle(f'PT initial-state convergence: {mode}, eta={eta}, alpha={ALPHA}')
            fig.tight_layout()
            fig.savefig(OUTPUT / f'convergence_{mode}_eta_{eta:.2f}.png', dpi=180)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        for eta in ETAS:
            t, _, _, dv, _ = all_results[(mode, eta)]
            ax.semilogy(t*1e6, np.maximum(dv/dv[0], 1e-14), label=f'eta={eta}')
        ax.set(xlabel='Time [us]', ylabel='D_V(t) / D_V(0)',
               title=f'Normalized initial-state convergence ({mode})')
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT / f'comparison_{mode}.png', dpi=180)
        plt.close(fig)

    with (OUTPUT / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('\n=== Convergence summary ===')
    for row in rows:
        print(f"{row['mode']:6s} eta={row['eta']:.2f} "
              f"D_V(0)={row['initial_DV_V']:.4g} "
              f"D_V(end)={row['final_DV_V']:.4g} "
              f"tail_RMS={row['tail_rms_DV_V']:.4g}")
    print(f'Saved to: {OUTPUT}')
    print('Finite-time initial-condition convergence only; not a proof of fading memory/ESP.')


if __name__ == '__main__':
    main()
