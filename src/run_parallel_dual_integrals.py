"""Dual-dynamics sliding first/second integrals; current reservoir features only."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .run_parallel_integrals import PTParallelReservoir, fit_predict, rf_features, targets, modulation_values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delta-a', type=float, default=0.20)
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--alpha', type=float, default=8.33)
    parser.add_argument('--encodings', nargs='+', choices=['amplitude', 'phase'], default=['amplitude'])
    parser.add_argument('--levels', type=int, default=8)
    parser.add_argument('--cycles-per-symbol', type=float, default=1.0)
    parser.add_argument('--lanes', type=int, default=64)
    parser.add_argument('--delta-b', type=float, nargs='+', default=[0.10],
                        help='Damping ratios G0/G_EP for the second branch')
    parser.add_argument('--windows', type=int, nargs='+',
                        default=[128, 192, 224, 256, 288, 320, 328, 336, 352, 384, 416])
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--symbols', type=int, default=3000)
    parser.add_argument('--washout', type=int, default=200)
    parser.add_argument('--seed', type=int, default=44)
    parser.add_argument('--output', type=Path, default=Path('results/parallel_dual_integrals'))
    args = parser.parse_args()
    if min([args.lanes, args.k, *args.windows]) < 1 or args.symbols < 50 or args.washout < 0:
        parser.error('Use positive lanes, k and windows, symbols >= 50 and washout >= 0.')
    if any(not np.isfinite(d) or d <= 0 for d in [args.delta_a, *args.delta_b]):
        parser.error('Damping ratios must be positive and finite.')

    if (args.levels < 2 or not np.isfinite(args.cycles_per_symbol) or args.cycles_per_symbol <= 0
            or not np.isfinite(args.eta) or not np.isfinite(args.alpha) or args.alpha < 0):
        parser.error('Invalid encoding, timing, eta or alpha.')

    reference = PTParallelReservoir(1, 0, 8.33).branches[0].tank
    delta_a = args.delta_a
    fc = reference.f_EP
    ts = args.cycles_per_symbol / fc
    gap = max(args.windows)
    start = max(gap, args.washout)
    train = slice(start, start + int(.6 * args.symbols))
    val = slice(train.stop + gap, train.stop + gap + int(.2 * args.symbols))
    test = slice(val.stop + gap, start + args.symbols + 2 * gap)
    values = np.random.default_rng(args.seed).uniform(-1, 1, test.stop)
    ridge_grid = [1e-8, 1e-6, 1e-4, 1e-2, 1, 100]
    args.output.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args), output=str(args.output), eta=args.eta, alpha=args.alpha,
                  delta_a=delta_a, carrier_hz=fc, symbol_period=ts,
                  revisit_period=args.lanes * ts, current_a=1e-4, modulation_depth=.8,
                  steps_per_cycle=16, rtol=1e-8, ridge_candidates=ridge_grid,
                  timestamp='row n at (n+1)Ts, includes value[n]',
                  second_order='same-window zero-initial double integral, not cascaded sliding sums',
                  feature_order='lane age, then branch A/B, then within-symbol voltage samples',
                  splits={name: [s.start, s.stop] for name, s in
                          [('train', train), ('validation', val), ('test', test)]})
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    np.save(args.output / 'input_values.npy', values)

    def simulate(delta, values, phases, amplitude):
        print(f'Simulating {args.lanes} lanes at delta={delta:g}', flush=True)
        reservoir = PTParallelReservoir(args.lanes, args.eta, args.alpha, mode='interleaved')
        for branch in reservoir.branches:
            tank = branch.tank
            tank.delta = delta
            tank.G0 = delta * tank.G_EP
            tank.R0 = 1 / tank.G0
        x, _ = rf_features(reservoir, values, phases, ts, fc, amplitude, args.k, 16, 1e-8)
        return x.reshape(len(values), args.lanes, 2 * args.k)

    summary = []
    for encoding in args.encodings:
        q, phases, amplitude = modulation_values(values, encoding, args.levels,
                                                'continuous', 1e-4, .8)
        np.savez_compressed(args.output / f'{encoding}_stimulus.npz',
                            values=q, phases=phases, peak_current=amplitude)
        a = simulate(delta_a, q, phases, amplitude)
        for delta_b in dict.fromkeys([None, delta_a, *args.delta_b]):
            label = ('single' if delta_b is None else 'identical' if delta_b == delta_a
                     else f'diverse_{delta_b!r}')
            if delta_b is None:
                x = a.reshape(len(values), -1)
            else:
                b = a if delta_b == delta_a else simulate(delta_b, q, phases, amplitude)
                x = np.concatenate((a, b), axis=2).reshape(len(values), -1)
            print(f'Fitting {encoding} {label}', flush=True)
            for length in args.windows:
                y = targets(q, length, ts)
                pred, ridge, metrics, model = fit_predict(x, y, train, val, test, ridge_grid)
                name = f'{encoding}_{label}_L{length}'
                np.savez_compressed(args.output / (name + '.npz'),
                                    target=y[test], prediction=pred[test],
                                    symbol_index=np.arange(test.start, test.stop), **model)
                np.savetxt(args.output / (name + '_test.csv'),
                           np.column_stack(((np.arange(test.start, test.stop)+1)*ts,
                                            q[test], y[test], pred[test])), delimiter=',',
                           header='time_s,value,target_I1,target_I2,prediction_I1,prediction_I2',
                           comments='')
                errors = np.sqrt(metrics['test_nmse'])
                row = dict(encoding=encoding, configuration=label, lanes=args.lanes,
                           tanks=args.lanes*(1 if delta_b is None else 2),
                           eta=args.eta, alpha=args.alpha, delta_a=delta_a, delta_b=delta_b,
                           features=x.shape[1], window=length, window_us=length*ts*1e6,
                           ridge=ridge, test_nrmse_order1=errors[0], test_nrmse_order2=errors[1],
                           both_orders_pass=bool(np.all(errors <= .1)))
                for split, scores in metrics.items():
                    row[split+'_order1'], row[split+'_order2'] = scores
                summary.append(row)
            with (args.output / 'summary.csv').open('w', newline='') as out:
                writer = csv.DictWriter(out, fieldnames=list(summary[0]))
                writer.writeheader()
                writer.writerows(summary)
    print(f'Completed: {args.output / "summary.csv"}', flush=True)


if __name__ == '__main__':
    main()
