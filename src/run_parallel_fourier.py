"""Sliding amplitude-domain DFT from PT tank states.

Copy into src/ alongside run_parallel_integrals.py and run
python -m src.run_parallel_fourier --n-tanks 1 16 32 64 --windows 16 32 64 128
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .run_parallel_integrals import PTParallelReservoir, fit_predict, rf_features


def sliding_dft(values, length):
    """Row n contains the rectangular window ending at symbol n."""
    history = sliding_window_view(np.pad(values, (length - 1, 0)), length)
    return np.fft.rfft(history, axis=1) / length


def complex_nrmse(target, prediction, length):
    """Return per-bin error and Parseval-weighted full-spectrum error."""
    error = np.mean(np.abs(prediction - target) ** 2, axis=0)
    variance = np.mean(np.abs(target - target.mean(axis=0)) ** 2, axis=0)
    ratio = np.full_like(error, np.nan)
    np.divide(error, variance, out=ratio, where=variance > 0)
    weights = np.full(error.shape, 2.0)
    weights[0] = 1
    if length % 2 == 0:
        weights[-1] = 1
    denominator = weights @ variance
    total = np.sqrt((weights @ error) / denominator) if denominator > 0 else np.nan
    return np.sqrt(ratio), total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-tanks', type=int, nargs='+', default=[1, 16, 32, 64])
    parser.add_argument('--windows', type=int, nargs='+', default=[16, 32, 64, 128])
    parser.add_argument('--eta', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=8.33)
    parser.add_argument('--k', type=int, default=8, help='Samples per voltage node per symbol')
    parser.add_argument('--symbols', type=int, default=3000)
    parser.add_argument('--washout', type=int, default=200)
    parser.add_argument('--seed', type=int, default=44)
    parser.add_argument('--output', type=Path, default=Path('results/parallel_fourier'))
    args = parser.parse_args()
    if min(args.n_tanks + args.windows + [args.k]) < 1 or args.symbols < 50 or args.washout < 0:
        parser.error('Positive tank counts, windows and k, symbols >= 50 and washout >= 0 required.')

    fc = PTParallelReservoir(1, args.eta, args.alpha).branches[0].tank.f_EP
    ts = 1 / fc
    gap = max(args.windows)
    start = max(gap, args.washout)
    train = slice(start, start + int(0.6 * args.symbols))
    val = slice(train.stop + gap, train.stop + gap + int(0.2 * args.symbols))
    test = slice(val.stop + gap, start + args.symbols + 2 * gap)
    values = np.random.default_rng(args.seed).uniform(-1, 1, test.stop)
    phases = np.zeros_like(values)
    amplitude = 1e-4 * (1 + 0.8 * values)
    lambdas = [1e-8, 1e-6, 1e-4, 1e-2, 1, 100]
    args.output.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args), output=str(args.output), carrier_hz=fc, symbol_period=ts,
                  current_a=1e-4, modulation_depth=0.8, steps_per_cycle=16, rtol=1e-8,
                  feature_order='age', ridge_candidates=lambdas,
                  splits={name: [s.start, s.stop] for name, s in
                          [('train', train), ('validation', val), ('test', test)]})
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    np.save(args.output / 'amplitude_values.npy', values)
    targets = {length: sliding_dft(values, length) for length in args.windows}
    summary, bins = [], []

    for index, count in enumerate(args.n_tanks):
        print(f'Simulating {count} tanks', flush=True)
        reservoir = PTParallelReservoir(count, args.eta, args.alpha, mode='interleaved')
        features, raw = rf_features(reservoir, values, phases, ts, fc, amplitude, args.k, 16, 1e-8)
        sources = {'reservoir': features}
        if index == 0:
            sources['input_only'] = raw
        for source, x in sources.items():
            for length, target in targets.items():
                # DC and, for even L, Nyquist have no imaginary degree of freedom.
                imaginary = slice(1, -1 if length % 2 == 0 else None)
                y = np.column_stack((target.real, target[:, imaginary].imag))
                prediction, ridge, _, model = fit_predict(x, y, train, val, test, lambdas)
                width = target.shape[1]
                spectrum = prediction[:, :width].astype(complex)
                spectrum[:, imaginary] += 1j * prediction[:, width:]
                per_bin, overall = complex_nrmse(target[test], spectrum[test], length)
                tanks = count if source == 'reservoir' else 0
                frequencies = np.fft.rfftfreq(length, d=ts)
                name = f'{source}_tanks{tanks}_L{length}'
                np.savez_compressed(args.output / f'{name}.npz', target=target[test],
                                    prediction=spectrum[test], frequency_hz=frequencies,
                                    symbol_index=np.arange(test.start, test.stop), **model)
                row = dict(source=source, n_tanks=tanks, window=length, features=x.shape[1],
                           ridge=ridge, test_complex_nrmse=overall, dc_nrmse=per_bin[0])
                summary.append(row)
                for k, (frequency, error) in enumerate(zip(frequencies, per_bin)):
                    bins.append(dict(source=source, n_tanks=tanks, window=length, bin=k,
                                     frequency_hz=frequency, test_complex_nrmse=error))
        for filename, rows in [('summary.csv', summary), ('bins.csv', bins)]:
            with (args.output / filename).open('w', newline='') as output:
                writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


if __name__ == '__main__':
    main()
