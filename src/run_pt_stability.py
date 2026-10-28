"""Fixed-circuit nonlinear PT reservoir characterization.

Run from project root:
    python3 run_pt_stability.py

Uses src.circuits.pt_single.PTSingleReservoir, with its existing
solver='ivp' and solver='exact_zoh' interfaces. No benchmark tuning.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.circuits.pt_single import PTSingleReservoir


def zoh_signal(values: np.ndarray, period: float):
    """Piecewise-constant input; midpoint sampling avoids boundary ambiguity."""
    values = np.asarray(values, dtype=float)

    def u(t):
        idx = int(np.floor(max(float(t), 0.0) / period))
        return float(values[idx]) if idx < len(values) else 0.0

    return u


def simulate_zoh(reservoir, values, ts, pps, x0=None):
    duration = len(values) * ts
    t = np.arange(len(values) * pps, dtype=float) * (ts / pps)
    sol = reservoir.simulate(
        input_func=zoh_signal(values, ts),
        t_span=(0.0, duration),
        t_eval=t,
        x0=x0,
        solver='exact_zoh',
        symbol_period=ts,
    )
    if not sol.success or not np.isfinite(sol.y).all():
        raise RuntimeError('Nonfinite or failed ZOH solution')
    return t, sol.y


def simulate_free(reservoir, duration, dt, x0):
    t = np.arange(int(np.ceil(duration / dt)) + 1) * dt
    t = t[t <= duration]
    sol = reservoir.simulate(
        input_func=lambda t: 0.0,
        t_span=(0.0, duration),
        t_eval=t,
        x0=x0,
        solver='ivp',
    )
    if not sol.success or not np.isfinite(sol.y).all():
        raise RuntimeError('Nonfinite or failed free-decay solution')
    return t, sol.y


def metrics(label, t, states, drive_end=None):
    v = states[:2]
    row = {'test': label,
           'peak_v1_V': float(np.max(np.abs(v[0]))),
           'peak_v2_V': float(np.max(np.abs(v[1]))),
           'rms_v1_V': float(np.sqrt(np.mean(v[0] ** 2))),
           'rms_v2_V': float(np.sqrt(np.mean(v[1] ** 2))),
           'finite': bool(np.isfinite(states).all())}
    if drive_end is not None:
        # Check whether the post-drive voltage envelope has decreased.
        mask = t >= drive_end
        if np.any(mask):
            post = np.max(np.abs(v[:, mask]), axis=0)
            m = max(1, len(post) // 5)
            row['post_early_peak_V'] = float(np.max(post[:m]))
            row['post_late_peak_V'] = float(np.max(post[-m:]))
            row['post_end_norm_V'] = float(np.linalg.norm(v[:, -1]))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eta', type=float, default=1.0)
    ap.add_argument('--alpha', type=float, default=8.33)
    ap.add_argument('--fc', type=float, default=48e6)
    ap.add_argument('--ts-cycles', type=float, default=0.5,
                    help='ZOH symbol period as fraction of carrier period')
    ap.add_argument('--pps', type=int, default=16)
    ap.add_argument('--amplitude', type=float, default=1e-2,
                    help='Input current magnitude [A]')
    ap.add_argument('--drive-symbols', type=int, default=500)
    ap.add_argument('--tail-symbols', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--initial-v', type=float, default=0.1,
                    help='Free-decay initial V1 [V]')
    ap.add_argument('--out', type=Path, default=Path('results/pt_stability'))
    args = ap.parse_args()
    if args.ts_cycles <= 0 or args.pps < 2 or args.drive_symbols < 2 or args.tail_symbols < 2:
        ap.error('Invalid timing or symbol counts')
    args.out.mkdir(parents=True, exist_ok=True)
    ts = args.ts_cycles / args.fc
    r = PTSingleReservoir(eta=args.eta, alpha=args.alpha)

    # 1) Free decay from a small voltage perturbation.
    t_free, x_free = simulate_free(r, (args.drive_symbols + args.tail_symbols) * ts,
                                  ts / args.pps,
                                  np.array([args.initial_v, 0., 0., 0.]))

    # 2) Single current pulse followed by zero input.
    pulse = np.zeros(args.drive_symbols + args.tail_symbols)
    pulse[0] = args.amplitude
    t_pulse, x_pulse = simulate_zoh(r, pulse, ts, args.pps)

    # 3) Fixed-seed bipolar excitation, followed by input-off decay.
    rng = np.random.default_rng(args.seed)
    random_input = np.r_[args.amplitude * rng.choice([-1., 1.], args.drive_symbols),
                         np.zeros(args.tail_symbols)]
    t_random, x_random = simulate_zoh(r, random_input, ts, args.pps)
    drive_end = args.drive_symbols * ts

    records = [metrics('free_decay', t_free, x_free, drive_end=0.0),
               metrics('single_pulse', t_pulse, x_pulse, drive_end=ts),
               metrics('random_zoh', t_random, x_random, drive_end=drive_end)]

    fields = sorted(set().union(*(row.keys() for row in records)))
    with (args.out / 'stability_metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    for name, t, x, cutoff in [
        ('free_decay', t_free, x_free, None),
        ('single_pulse', t_pulse, x_pulse, ts),
        ('random_zoh', t_random, x_random, drive_end),
    ]:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(t * 1e6, x[0], lw=1.0, label='V1')
        ax.plot(t * 1e6, x[1], lw=1.0, label='V2')
        if cutoff is not None:
            ax.axvline(cutoff * 1e6, ls='--', color='black', lw=1,
                       label='Input off')
        ax.set(xlabel='Time [µs]', ylabel='Voltage [V]',
               title=f'{name}  |  η={args.eta}, α={args.alpha}')
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out / f'{name}.png', dpi=200)
        plt.close(fig)

    print(f'Fixed PT configuration: eta={args.eta}, alpha={args.alpha}')
    print(f'Ts={ts * 1e9:.5f} ns, amplitude={args.amplitude:.4g} A, seed={args.seed}')
    for row in records:
        print(f"{row['test']:>14}: max|V1|={row['peak_v1_V']:.6g} V, "
              f"max|V2|={row['peak_v2_V']:.6g} V, finite={row['finite']}")
        if 'post_late_peak_V' in row:
            print(f"  post-drive early/late peak: {row['post_early_peak_V']:.6g} / "
                  f"{row['post_late_peak_V']:.6g} V")
    print(f'Results: {args.out.resolve()}')
    print('Note: finite trajectories and decreasing tail peaks are diagnostics, '
          'not a proof of global nonlinear stability.')


if __name__ == '__main__':
    main()
