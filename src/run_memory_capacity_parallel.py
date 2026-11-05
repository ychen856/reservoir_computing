"""Compare single and parallel reservoirs with identical input and MC evaluation.

python -m src.run_memory_capacity_parallel
python -m src.run_memory_capacity_parallel --symbols 400 --washout 80 --max-delay 20
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from src.tasks.base import TaskData
from src.circuits.pt_single import PTSingleReservoir
from src.circuits.pt_parallel import PTParallelReservoir
from src.simulation.runner import simulate_continuous_task, extract_continuous_features
from src.tasks.properties.memory_capacity import evaluate_linear_memory_capacity
from src.utils.features import reorder_features_by_age

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbols', type=int, default=3000)
    parser.add_argument('--washout', type=int, default=200)
    parser.add_argument('--max-delay', type=int, default=50)
    parser.add_argument('--eta', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=8.33)
    parser.add_argument('--tanks', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--modes', nargs='+', choices=['broadcast', 'interleaved'],
                        default=['broadcast', 'interleaved'])
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--points-per-symbol', type=int, default=16)
    parser.add_argument('--cycles-per-symbol', type=float, default=0.5)
    parser.add_argument('--amplitude', type=float, default=1e-4)
    parser.add_argument('--ridge', type=float, default=1e-6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--output', type=Path, default=Path('results/parallel_memory_capacity'))
    args = parser.parse_args()
    if (args.symbols <= max(args.washout, args.max_delay)+10
            or args.washout < 0 or args.max_delay < 1 or args.k < 1
            or args.points_per_symbol < args.k+1 or args.cycles_per_symbol <= 0
            or any(m < 1 for m in args.tanks) or args.ridge < 0):
        parser.error('invalid sample, delay, tank, timing or feature configuration')
    u = np.random.default_rng(args.seed).choice([-1., 1.], args.symbols)
    task = TaskData('linear_memory_capacity', u, u, 'regression',
                    washout=args.washout, train_ratio=args.train_ratio)
    ts = args.cycles_per_symbol/48e6
    options = dict(symbol_period=ts, points_per_symbol=args.points_per_symbol,
                   k=args.k, amplitude_scale=args.amplitude, encoding='zoh', solver='exact_zoh')
    def simulate(reservoir):
        return simulate_continuous_task(reservoir, task, **options)
    def evaluate(sim):
        return evaluate_linear_memory_capacity(sim.X, u, max_delay=args.max_delay,
            washout=args.washout, train_ratio=args.train_ratio, ridge=args.ridge, normalize=True)
    single = simulate(PTSingleReservoir(args.eta, args.alpha))
    reference = evaluate(single)
    rows, curves = [], []
    def record(label, sim, ev):
        print(f'{label:24s} D={ev.feature_dim:3d} train MC={ev.total_train_capacity:.6f} '
              f'test MC={ev.total_test_capacity:.6f}')
        rows.append(dict(architecture=label, feature_dim=ev.feature_dim,
            n_train=ev.n_train, n_test=ev.n_test,
            train_mc=ev.total_train_capacity, test_mc=ev.total_test_capacity))
        for d, train, test in zip(ev.delays, ev.train_capacity, ev.test_capacity):
            curves.append(dict(architecture=label, delay=int(d), train_mc=train, test_mc=test))
    record('single', single, reference)
    for count in args.tanks:
        for mode in args.modes:
            p = PTParallelReservoir(count, eta=args.eta, alpha=args.alpha, mode=mode)
            sim = simulate(p)
            if sim.X.shape != (args.symbols, 2*count*args.k):
                raise AssertionError('parallel feature dimension mismatch')
            # Re-extraction must preserve every branch and feature ordering.
            np.testing.assert_array_equal(sim.X, extract_continuous_features(
                sim, args.symbols, ts, args.k))
            ev = evaluate(sim)
            if count == 1:
                np.testing.assert_allclose(sim.X, single.X, rtol=1e-6, atol=1e-8)
                np.testing.assert_allclose(ev.test_capacity, reference.test_capacity,
                                            rtol=1e-4, atol=1e-5)
                print('  PASS: n_tank=1 features and MC reproduce single')
            if mode == 'broadcast':
                # Identical branches should duplicate features, not add dynamics.
                blocks = sim.X.reshape(args.symbols, args.k, count, 2)
                for j in range(1, count):
                    np.testing.assert_allclose(blocks[:, :, 0], blocks[:, :, j], atol=1e-10)
            #record(f'{mode}_n_tank{count}', sim, ev)
            record(
                f"{mode}_pairs{count}_physical",
                sim,
                ev,
            )

            if mode == "interleaved":
                X_age = reorder_features_by_age(
                    X=sim.X,
                    n_pairs=count,
                    k=args.k,
                    symbol_indices=np.arange(len(sim.X)),
                )

                age_ev = evaluate_linear_memory_capacity(
                    X=X_age,
                    u=u,
                    max_delay=args.max_delay,
                    washout=args.washout,
                    train_ratio=args.train_ratio,
                    ridge=args.ridge,
                    normalize=True,
                )

                record(
                    f"{mode}_pairs{count}_age",
                    sim,
                    age_ev,
                )

                # With one pair, ordering must have no effect.
                if count == 1:
                    np.testing.assert_array_equal(X_age, sim.X)

                    np.testing.assert_allclose(
                        age_ev.test_capacity,
                        ev.test_capacity,
                    )


    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in [('summary.csv', rows), ('by_delay.csv', curves)]:
        with (args.output/name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    settings = vars(args).copy()
    settings['output'] = str(args.output)
    settings['feature_order'] = 'time-major, then branch-major V1/V2'
    settings['note'] = 'Raw squared-correlation MC; no permutation-null correction. Fixed ridge.'
    (args.output/'config.json').write_text(json.dumps(settings, indent=2))
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
