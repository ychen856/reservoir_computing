"""Run GQC sliding integrals and Fourier windows with PT-compatible targets."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from src.pass_gqc.common import load_cases, parser, validate, write_csv
from src.pass_gqc.windowed import (
    complex_nrmse, fit_predict, modulation_values, rf_features, sliding_dft, targets,
)


RIDGES = [1e-8, 1e-6, 1e-4, 1e-2, 1, 100]


def run_case(case, args, directory):
    directory.mkdir(parents=True, exist_ok=True)
    ts = args.cycles_per_symbol / case.frequency
    gap = max(args.windows)
    start = max(gap, args.washout)
    train = slice(start, start + int(.6*args.symbols))
    validation = slice(train.stop + gap, train.stop + gap + int(.2*args.symbols))
    test = slice(validation.stop + gap, start + args.symbols + 2*gap)
    stimulus = np.random.default_rng(case.seed).uniform(-1, 1, test.stop)
    metadata = dict(vars(args), **asdict(case))
    metadata.update(carrier_hz=case.frequency, symbol_period_s=ts,
                    splits=dict(train=[train.start, train.stop],
                                validation=[validation.start, validation.stop],
                                test=[test.start, test.stop]), ridge_candidates=RIDGES)
    (directory/'config.json').write_text(json.dumps(metadata, indent=2, default=str))
    np.save(directory/'input.npy', stimulus)
    rows = []
    for encoding in args.encodings:
        integral = 'integral' in args.tasks
        fourier = 'fourier' in args.tasks and encoding == 'amplitude'
        if not (integral or fourier):
            continue
        values, phases, current = modulation_values(stimulus, encoding, args.levels,
                                                     'continuous', args.amplitude, .8)
        print(f'{directory.name}: simulating {encoding} RF input', flush=True)
        reservoir, raw = rf_features(case, phases, current, ts, args.k, args.points_per_symbol)
        for source, features in [('reservoir', reservoir), ('input_only', raw)]:
            for window in args.windows:
                base = dict(asdict(case), encoding=encoding, source=source, window=window,
                            window_seconds=window*ts, circuit_type=('open' if case.coupling_l_nh is None else 'coupled'))
                if integral:
                    expected = targets(values, window, ts)
                    for readout in args.readouts:
                        design = features if readout == 'linear' else np.column_stack((features, features**2))
                        pred, ridge, metrics, model = fit_predict(
                            design, expected, train, validation, test, RIDGES)
                        for order in (1, 2):
                            row = dict(base, task=f'integral_order{order}', readout=readout,
                                       features=design.shape[1], ridge=ridge, metric='NMSE')
                            row.update({name: score[order-1] for name, score in metrics.items()})
                            rows.append(row)
                        np.savez_compressed(directory/f'integral_{encoding}_{source}_{readout}_L{window}.npz',
                                            target=expected[test], prediction=pred[test],
                                            symbol_index=np.arange(test.start, test.stop), **model)
                if fourier:
                    expected = sliding_dft(values, window)
                    imaginary = slice(1, -1 if window % 2 == 0 else None)
                    target = np.column_stack((expected.real, expected[:, imaginary].imag))
                    pred, ridge, _, model = fit_predict(
                        features, target, train, validation, test, RIDGES)
                    width = expected.shape[1]
                    spectrum = pred[:, :width].astype(complex)
                    spectrum[:, imaginary] += 1j*pred[:, width:]
                    per_bin, test_error = complex_nrmse(expected[test], spectrum[test], window)
                    _, val_error = complex_nrmse(expected[validation], spectrum[validation], window)
                    rows.append(dict(base, task='fourier', readout='linear', features=features.shape[1],
                                     ridge=ridge, metric='complex_NRMSE',
                                     validation_complex_nrmse=val_error, test_complex_nrmse=test_error))
                    np.savez_compressed(directory/f'fourier_{source}_L{window}.npz',
                                        target=expected[test], prediction=spectrum[test], per_bin_nrmse=per_bin,
                                        frequency_hz=np.fft.rfftfreq(window, d=ts),
                                        symbol_index=np.arange(test.start, test.stop), **model)
    write_csv(directory/'summary.csv', rows)


def main():
    p = parser('benchmarks', ('integral', 'fourier'))
    p.description = __doc__
    p.set_defaults(design_indices=[5, 37], modes=['graphene'], seeds=[44],
                   routing='interleaved', ordering='age', cycles_per_symbol=1.,
                   amplitude=1e-4, washout=200, train_ratio=.6, output=Path('results/pass_gqc/windowed'))
    p.add_argument('--counts', nargs='+', type=int, default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument('--circuit-types', nargs='+', choices=['open', 'coupled'], default=['open', 'coupled'])
    p.add_argument('--windows', nargs='+', type=int, default=[4, 8, 16, 32, 64, 128])
    p.add_argument('--encodings', nargs='+', choices=['phase', 'amplitude'], default=['phase', 'amplitude'])
    p.add_argument('--readouts', nargs='+', choices=['linear', 'square'], default=['linear', 'square'])
    p.add_argument('--levels', type=int, default=8)
    args = p.parse_args()
    if (min(args.counts+args.windows)<1 or args.levels<2 or args.routing!='interleaved'
            or args.ordering!='age' or args.n_dimers!=1 or args.train_ratio!=.6):
        p.error('Use positive counts/windows, levels >= 2, the default 60/20/20 split, and interleaved age routing. Set dimer counts with --counts.')
    if 'fourier' in args.tasks and 'amplitude' not in args.encodings:
        p.error('Fourier requires amplitude encoding')
    if 'coupled' in args.circuit_types and any(n>1 and n%2 for n in args.counts):
        p.error('Coupled counts must be even')
    if args.circuit_types==['coupled'] and set(args.counts)=={1}:
        p.error('A coupled experiment requires at least two dimers')
    inductance = args.coupling_l_nh
    for count in dict.fromkeys(args.counts):
        for circuit in dict.fromkeys(args.circuit_types):
            if count==1 and circuit=='coupled':
                continue
            args.n_dimers = count
            args.coupling_l_nh = (inductance if inductance is not None else (2 if count==2 else 8)) if circuit=='coupled' else None
            validate(args)
            suffix = f'{args.coupling_l_nh:g}nH_{args.coupling_r_ohm:g}ohm' if circuit=='coupled' else 'unlinked'
            for case in load_cases(args):
                destination = args.output/circuit/f'{count}dimers'/suffix/case.key
                run_case(case, args, destination)


if __name__ == '__main__':
    main()
