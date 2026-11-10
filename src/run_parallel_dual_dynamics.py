"""Compare damping diversity within interleaved PT lanes on sliding DFT."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .run_parallel_fourier import complex_nrmse, sliding_dft
from .run_parallel_integrals import PTParallelReservoir, fit_predict, rf_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lanes', type=int, default=64)
    parser.add_argument('--delta-b', type=float, nargs='+', default=[0.10, 0.30],
                        help='Damping ratios G0/G_EP for the second branch')
    parser.add_argument('--windows', type=int, nargs='+',
                        default=[192, 224, 232, 240, 248, 256, 288, 320])
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--symbols', type=int, default=3000)
    parser.add_argument('--washout', type=int, default=200)
    parser.add_argument('--seed', type=int, default=44)
    parser.add_argument('--output', type=Path, default=Path('results/parallel_dual_dynamics'))
    args = parser.parse_args()
    if min([args.lanes, args.k, *args.windows]) < 1 or args.symbols < 50 or args.washout < 0:
        parser.error('Use positive lanes, k and windows, symbols >= 50 and washout >= 0.')
    if any(not np.isfinite(d) or d <= 0 for d in args.delta_b):
        parser.error('Damping ratios must be positive and finite.')

    reference = PTParallelReservoir(1, 0, 8.33).branches[0].tank
    delta_a = reference.delta
    fc = reference.f_EP
    ts = 1 / fc
    gap = max(args.windows)
    start = max(gap, args.washout)
    train = slice(start, start + int(.6 * args.symbols))
    val = slice(train.stop + gap, train.stop + gap + int(.2 * args.symbols))
    test = slice(val.stop + gap, start + args.symbols + 2 * gap)
    values = np.random.default_rng(args.seed).uniform(-1, 1, test.stop)
    phases = np.zeros_like(values)
    amplitude = 1e-4 * (1 + .8 * values)
    ridge_grid = [1e-8, 1e-6, 1e-4, 1e-2, 1, 100]
    args.output.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args), output=str(args.output), eta=0, alpha=8.33,
                  delta_a=delta_a, carrier_hz=fc, symbol_period=ts,
                  revisit_period=args.lanes * ts, current_a=1e-4, modulation_depth=.8,
                  steps_per_cycle=16, rtol=1e-8, ridge_candidates=ridge_grid,
                  feature_order='lane age, then branch A/B, then within-symbol voltage samples',
                  splits={name: [s.start, s.stop] for name, s in
                          [('train', train), ('validation', val), ('test', test)]})
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    np.save(args.output / 'amplitude_values.npy', values)

    def simulate(delta):
        print(f'Simulating {args.lanes} lanes at delta={delta:g}', flush=True)
        reservoir = PTParallelReservoir(args.lanes, 0, 8.33, mode='interleaved')
        for branch in reservoir.branches:
            tank = branch.tank
            tank.delta = delta
            tank.G0 = delta * tank.G_EP
            tank.R0 = 1 / tank.G0
        x, _ = rf_features(reservoir, values, phases, ts, fc, amplitude, args.k, 16, 1e-8)
        return x.reshape(len(values), args.lanes, 2 * args.k)

    a = simulate(delta_a)
    targets = {length: sliding_dft(values, length) for length in args.windows}
    summary, ages = [], []
    for delta_b in dict.fromkeys([None, delta_a, *args.delta_b]):
        label = 'single' if delta_b is None else 'identical' if delta_b == delta_a else f'diverse_{delta_b!r}'
        if delta_b is None:
            x = a.reshape(len(values), -1)
        else:
            b = a if delta_b == delta_a else simulate(delta_b)
            x = np.concatenate((a, b), axis=2).reshape(len(values), -1)
        print(f'Fitting {label}', flush=True)
        for length, target in targets.items():
            imag = slice(1, -1 if length % 2 == 0 else None)
            y = np.column_stack((target.real, target[:, imag].imag))
            pred, ridge, _, model = fit_predict(x, y, train, val, test, ridge_grid)
            width = target.shape[1]
            spectrum = pred[test, :width].astype(complex)
            spectrum[:, imag] += 1j * pred[test, width:]
            scores, overall = complex_nrmse(target[test], spectrum, length)
            error = np.fft.irfft(spectrum - target[test], n=length, axis=1) * length
            age_mse = np.mean(error[:, ::-1] ** 2, axis=0)
            frequencies = np.fft.rfftfreq(length, d=ts)
            np.savez_compressed(args.output / f'{label}_L{length}.npz',
                                target=target[test], prediction=spectrum, bin_nrmse=scores,
                                frequency_hz=frequencies, age_mse=age_mse,
                                symbol_index=np.arange(test.start, test.stop), **model)
            summary.append(dict(configuration=label, lanes=args.lanes,
                                tanks=args.lanes * (1 if delta_b is None else 2),
                                delta_a=delta_a, delta_b=delta_b, features=x.shape[1],
                                window=length, window_us=length * ts * 1e6, ridge=ridge,
                                test_complex_nrmse=overall, worst_bin_nrmse=float(np.max(scores)),
                                passing_bins=int(np.sum(scores <= .1)), total_bins=width,
                                all_bins_pass=bool(np.all(scores <= .1))))
            ages.extend(dict(configuration=label, window=length, age=age, mse=mse)
                        for age, mse in enumerate(age_mse))
        for name, rows in [('summary.csv', summary), ('age_errors.csv', ages)]:
            with (args.output / name).open('w', newline='') as out:
                writer = csv.DictWriter(out, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


if __name__ == '__main__':
    main()
