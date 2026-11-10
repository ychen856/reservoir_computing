"""RF-encoded sliding integrals. Run: python -m src.run_parallel_integrals --help."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from .circuits.pt_parallel import PTParallelReservoir


def encode(a, levels):
    """Uniform bins over [-1,1], bin-centre reconstruction; no duplicated 2pi."""
    indices = np.minimum(((np.clip(a, -1, 1)+1)*levels/2).astype(int), levels-1)
    return -1+(indices+0.5)*2/levels, indices*2*np.pi/levels


def modulation_values(a, encoding, levels, amplitude_values, current, depth):
    """Return encoded numerical values, RF phase and peak current."""
    q, phases = encode(a, levels)
    if encoding == 'phase':
        return q, phases, np.full(len(a), current)
    values = a.copy() if amplitude_values == 'continuous' else q
    return values, np.zeros(len(a)), current*(1+depth*values)


def targets(q, window, ts):
    """Row n is read at (n+1)Ts and includes q[n]. ZOH-exact integrals.

    I2 = integral_a^b (b-t)q(t)dt, a=max(0,b-window*Ts).
    This is NOT two moving sums (which would have support 2L-1).
    """
    first = np.convolve(q, np.ones(window), mode='full')[:len(q)]*ts
    second = np.convolve(q, np.arange(window)+0.5, mode='full')[:len(q)]*ts**2
    return np.column_stack((first, second))


def rf_features(reservoir, q, phases, ts, fc, amplitude, k, steps, rtol):
    """Continuous sinusoidal forcing; integrate separately at every switch edge.

    No state reset between symbols, no history buffer, no carrier ZOH shortcut.
    Use each symbol's interior samples only; output timestamp is its right edge.
    """
    count = len(q)
    blocks = np.empty((count, reservoir.n_tanks, 2*k))
    fractions = np.arange(1, k+1)/(k+1)
    raw = np.empty((count, k))
    for n in range(count):
        raw[n] = amplitude[n]*np.cos(2*np.pi*fc*(n+fractions)*ts+phases[n])
    for j, branch in enumerate(reservoir.branches):
        state = np.zeros(4)
        for n in range(count):
            active = reservoir.mode == 'broadcast' or n % reservoir.n_tanks == j
            def drive(local):
                # Symbol index is captured explicitly; no boundary rounding ambiguity.
                return (amplitude[n]*np.cos(2*np.pi*fc*(n*ts+local)+phases[n])
                        if active else 0.0)
            def rhs(local, x):
                return branch.tank.nonlinear_rhs(local, x, branch.eta, branch.alpha, drive)
            sol = solve_ivp(rhs, (0, ts), state, method='DOP853',
                            t_eval=np.r_[fractions*ts, ts], rtol=rtol,
                            atol=rtol*1e-3, max_step=min(ts/8, 1/(fc*steps)))
            if not sol.success or not np.isfinite(sol.y).all():
                raise RuntimeError(f'RF solve failed: branch={j}, symbol={n}: {sol.message}')
            state = sol.y[:, -1].copy()
            blocks[n, j] = sol.y[:2, :-1].T.reshape(-1)
    ages = (np.arange(count)[:, None]-np.arange(reservoir.n_tanks)) % reservoir.n_tanks
    if reservoir.mode == 'interleaved':
        blocks = np.take_along_axis(blocks, ages[:, :, None], axis=1)
    return blocks.reshape(count, -1), raw


def fit_predict(X, Y, train, val, test, lambdas):
    """Training-only scaling; validation-only ridge selection; frozen test readout."""
    mean, scale = X[train].mean(0), X[train].std(0)
    scale = np.where(scale > 1e-14, scale, 1)
    Z = (X-mean)/scale
    ym, ys = Y[train].mean(0), Y[train].std(0)
    ys = np.where(ys > 0, ys, 1)
    target = (Y-ym)/ys
    zm = Z[train].mean(0)
    A, B = Z[train]-zm, target[train]-target[train].mean(0)
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    candidates = []
    for lam in lambdas:
        w = (Vh.T*(s/(s*s+lam))) @ (U.T@B)
        bias = target[train].mean(0)-zm@w
        prediction = (Z@w+bias)*ys+ym
        score = np.mean(((prediction[val]-Y[val])/ys)**2)
        candidates.append((score, lam, w, bias, prediction))
    _, lam, w, bias, pred = min(candidates, key=lambda item: item[0])
    metrics = {}
    for label, sl in [('train', train), ('validation', val), ('test', test)]:
        mse = np.mean((pred[sl]-Y[sl])**2, axis=0)
        variance = np.var(Y[sl], axis=0)
        metrics[label+'_nmse'] = (mse/np.maximum(variance, 1e-300)).tolist()
    model = dict(mean=mean, scale=scale, y_mean=ym, y_scale=ys, w=w, bias=bias)
    return pred, lam, metrics, model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n-tanks', nargs='+', type=int, default=[1, 4])
    p.add_argument('--encodings', nargs='+', choices=['phase', 'amplitude'], default=['phase', 'amplitude'])
    p.add_argument('--windows', nargs='+', type=int, default=[4, 8, 16, 32])
    p.add_argument('--levels', type=int, default=8)
    p.add_argument('--eta', type=float, default=0.5)
    p.add_argument('--alpha', type=float, default=8.33)
    p.add_argument('--fc', type=float, default=None, help='Hz; default: loaded tank f_EP')
    timing = p.add_mutually_exclusive_group()
    timing.add_argument('--cycles-per-symbol', type=float, default=None)
    timing.add_argument('--symbol-period', type=float, default=None, help='envelope sample interval in seconds; independent of fc')
    p.add_argument('--amplitude-values', choices=['continuous', 'quantized'], default='continuous')
    p.add_argument('--curve', choices=['iid', 'smooth'], default='iid', help='iid for memory evaluation; smooth for demonstration')
    p.add_argument('--input-current', type=float, default=1e-4, help='carrier peak in amperes')
    p.add_argument('--modulation-depth', type=float, default=0.8)
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--steps-per-cycle', type=int, default=16)
    p.add_argument('--rtol', type=float, default=1e-8)
    p.add_argument('--symbols', type=int, default=3000, help='scored train/val/test sequence length')
    p.add_argument('--washout', type=int, default=200)
    p.add_argument('--seed', type=int, default=44)
    p.add_argument('--readouts', nargs='+', choices=['linear', 'square'], default=['linear', 'square'])
    p.add_argument('--output', type=Path, default=Path('results/parallel_integrals'))
    args = p.parse_args()
    if (min(args.n_tanks+args.windows) < 1 or args.levels < 2 or args.k < 1
            or args.symbols < 50 or args.washout < 0
            or (args.cycles_per_symbol is not None and args.cycles_per_symbol <= 0)
            or (args.symbol_period is not None and args.symbol_period <= 0)
            or args.input_current <= 0 or not 0 <= args.modulation_depth < 1
            or args.steps_per_cycle < 4 or args.rtol <= 0 or (args.fc is not None and args.fc <= 0)):
        p.error('invalid dimensions, modulation depth, frequency, or solver settings')
    args.output.mkdir(parents=True, exist_ok=True)
    fc = args.fc or PTParallelReservoir(n_tanks=1).branches[0].tank.f_EP
    ts = args.symbol_period if args.symbol_period is not None else (args.cycles_per_symbol or 1)/fc
    gap = max(args.windows)
    # Guard gaps ensure target windows in adjacent splits do not share symbols.
    start = max(args.washout, gap)
    total = start+args.symbols+2*gap
    a = np.random.default_rng(args.seed).uniform(-1, 1, total)
    if args.curve == 'smooth':
        # Fixed period in physical time, independent of sample interval.
        t = np.arange(total)*ts
        a = (0.5*np.sin(2*np.pi*0.2e6*t+0.3)
             +0.3*np.sin(2*np.pi*0.73e6*t+1.1)
             +0.2*np.cos(2*np.pi*1.37e6*t))
    q, phase = encode(a, args.levels)
    nt, nv = int(args.symbols*.6), int(args.symbols*.2)
    train = slice(start, start+nt)
    val = slice(train.stop+gap, train.stop+gap+nv)
    test = slice(val.stop+gap, total)
    metadata = vars(args).copy()
    metadata.update(output=str(args.output), fc=fc, symbol_period=ts,
                    feature_order='age', second_order='same-window zero-initial-condition double integral',
                    timestamp='row n at (n+1)Ts; target includes encoded numerical value[n]',
                    splits={key:[sl.start,sl.stop] for key,sl in [('train',train),('validation',val),('test',test)]})
    (args.output/'config.json').write_text(json.dumps(metadata, indent=2))
    np.savez_compressed(args.output/'input.npz', a=a, q=q, phase=phase, time=(np.arange(total)+1)*ts)
    rows = []
    for encoding in args.encodings:
        values, phases, amplitudes = modulation_values(
            a, encoding, args.levels, args.amplitude_values,
            args.input_current, args.modulation_depth)
        np.savez_compressed(args.output/(encoding+'_stimulus.npz'),
                            values=values, phase=phases, peak_current=amplitudes,
                            time=(np.arange(total)+1)*ts)
        for tanks in args.n_tanks:
            print(f'Simulating {encoding}, tanks={tanks}, fc={fc:g} Hz', flush=True)
            reservoir = PTParallelReservoir(tanks, args.eta, args.alpha, mode='interleaved')
            X, raw = rf_features(reservoir, values, phases, ts, fc, amplitudes,
                                 args.k, args.steps_per_cycle, args.rtol)
            for source, features in [('reservoir', X), ('input_only', raw)]:
                for readout in args.readouts:
                    F = features if readout == 'linear' else np.column_stack((features, features**2))
                    for window in args.windows:
                        Y = targets(values, window, ts)
                        pred, lam, metrics, model = fit_predict(F, Y, train, val, test, [1e-8,1e-6,1e-4,1e-2,1,100])
                        name = f'{encoding}_tanks{tanks}_{source}_{readout}_L{window}'
                        np.savez_compressed(args.output/(name+'.npz'), target=Y, prediction=pred, **model)
                        if source == "reservoir":
                            sl = test
                            table = np.column_stack(((np.arange(total)[sl]+1)*ts, values[sl], Y[sl], pred[sl]))
                            np.savetxt(args.output/(name+"_test.csv"), table, delimiter=",",
                                       header="time,value,target_I1,target_I2,prediction_I1,prediction_I2", comments="")
                        row = dict(encoding=encoding, n_tanks=tanks, source=source, readout=readout,
                                   amplitude_values=args.amplitude_values if encoding == 'amplitude' else 'quantized',
                                   window=window, window_seconds=window*ts, features=F.shape[1], ridge=lam)
                        for split, errors in metrics.items():
                            row[split+'_order1'], row[split+'_order2'] = errors
                        rows.append(row)
            with (args.output/'summary.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
    print(f'Completed: {args.output / "summary.csv"}', flush=True)


if __name__ == '__main__':
    main()
