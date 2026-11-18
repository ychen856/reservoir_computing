"""Compare parallel A/B dynamics against same-size controls."""
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.pass_gqc.common import load_cases, parser, simulate, validate, write_csv
from src.pass_gqc.windowed import (
    complex_nrmse, fit_predict, modulation_values, rf_features, sliding_dft, targets,
)
from src.run_pass_gqc_windowed import RIDGES
from src.tasks.benchmarks.suite import evaluate, tasks
from src.tasks.properties.effective_dimension import evaluate_effective_dimension
from src.tasks.properties.memory_capacity import evaluate_linear_memory_capacity


def configurations(a, b, full):
    return {
        "matched_AA": np.repeat(a, 2, axis=1),
        "matched_BB": np.repeat(b, 2, axis=1),
        "dual_AB": np.stack((a, b), axis=2).reshape(len(a), -1),
        "homogeneous_A": full,
    }


def windowed(args, a, b, full, rows):
    ts = args.cycles_per_symbol / a.frequency
    gap = max(args.windows)
    start = max(gap, args.washout)
    train = slice(start, start + int(.6 * args.symbols))
    val = slice(train.stop + gap, train.stop + gap + int(.2 * args.symbols))
    test = slice(val.stop + gap, start + args.symbols + 2 * gap)
    stimulus = np.random.default_rng(a.seed).uniform(-1, 1, test.stop)
    for encoding in args.encodings:
        values, phases, current = modulation_values(stimulus, encoding, 8, "continuous", args.amplitude, .8)

        def features(case):
            print(f"{encoding}: {case.n_dimers} dimers at xi2={case.xi2:g}", flush=True)
            return rf_features(case, phases, current, ts, args.k, args.points_per_symbol)[0]

        xa, xb = features(a), features(b)
        if args.ab_only:
            states = {"dual_AB": np.stack((xa, xb), axis=2).reshape(len(xa), -1)}
        else:
            states = configurations(xa, xb, features(full))
        for name, x in states.items():
            for w in args.windows:
                y = targets(values, w, ts)
                pred, ridge, scores, _ = fit_predict(x, y, train, val, test, RIDGES)
                details = dict(configuration=name, encoding=encoding, window=w, features=x.shape[1])
                for index in (0, 1):
                    rows.append(dict(details, task=f"integral_order{index + 1}", metric="NMSE",
                                     validation=scores["validation_nmse"][index],
                                     test=scores["test_nmse"][index], ridge=ridge))
                np.savez_compressed(args.output/f"{name}_{encoding}_integral_{w}.npz",
                                    target=y[test], prediction=pred[test])
                if encoding == "amplitude":
                    spectrum = sliding_dft(values, w)
                    imaginary = slice(1, -1 if w % 2 == 0 else None)
                    target = np.column_stack((spectrum.real, spectrum[:, imaginary].imag))
                    fitted, ridge, _, _ = fit_predict(x, target, train, val, test, RIDGES)
                    width = spectrum.shape[1]
                    prediction = fitted[:, :width].astype(complex)
                    prediction[:, imaginary] += 1j * fitted[:, width:]
                    _, validation = complex_nrmse(spectrum[val], prediction[val], w)
                    _, score = complex_nrmse(spectrum[test], prediction[test], w)
                    rows.append(dict(details, task="fourier", metric="complex_NRMSE",
                                     validation=validation, test=score, ridge=ridge))
                    np.savez_compressed(args.output/f"{name}_fourier_{w}.npz",
                                        target=spectrum[test], prediction=prediction[test])
        write_csv(args.output/"summary.csv", rows)


def zoh(args, a, b, full, rows):
    def features(u):
        return configurations(*(simulate(case, u, args).X for case in (a, b, full)))

    print("ZOH: linear memory and effective dimension", flush=True)
    u = np.random.default_rng(a.seed).uniform(-1, 1, args.symbols)
    for name, x in features(u).items():
        memory = evaluate_linear_memory_capacity(x, u, max_delay=32, washout=args.washout,
                                                 train_ratio=args.train_ratio, ridge=args.ridge,
                                                 normalize=True)
        dimension = evaluate_effective_dimension(x[args.washout:], standardize=False)
        rows.append(dict(configuration=name, task="linear_memory_32", metric="capacity",
                         test=float(np.sum(memory.test_capacity)), features=x.shape[1]))
        rows.append(dict(configuration=name, task="effective_dimension", metric="dimension",
                         test=dimension.dimension, features=x.shape[1]))
    write_csv(args.output/"summary.csv", rows)
    for family in ("narma10", "xor"):
        print(f"ZOH: {family}", flush=True)
        for name, task in tasks(family, args, a.seed):
            for label, x in features(task.u).items():
                scores, _ = evaluate(x, task, args)
                rows.append(dict(configuration=label, task=name, **scores))
        write_csv(args.output/"summary.csv", rows)


def main():
    p = parser("benchmarks", ("windowed", "zoh"))
    p.set_defaults(design_indices=[5], modes=["graphene"], seeds=[44], xi2=[.5, 2],
                   n_dimers=4, routing="interleaved", ordering="age", symbols=600,
                   washout=100, cycles_per_symbol=1., amplitude=1e-4, train_ratio=.8)
    p.add_argument("--windows", type=int, nargs="+", default=[16, 32, 64])
    p.add_argument("--encodings", nargs="+", choices=["phase", "amplitude"],
                   default=["amplitude", "phase"])
    p.add_argument("--delays", type=int, nargs="+", default=[4])
    p.add_argument("--ab-only", action="store_true", help="Run only the windowed A/B configuration")
    args = p.parse_args()
    if args.ab_only and set(args.tasks) != {"windowed"}:
        p.error("--ab-only requires --tasks windowed")
    if len(args.design_indices) != 1 or len(args.seeds) != 1 or len(args.modes) != 1:
        p.error("Use one design, one seed and one mode per run")
    if len(args.xi2) != 2 or args.xi2[0] == args.xi2[1]:
        p.error("Provide two distinct xi2 values, A then B")
    if min(args.windows) < 2 or args.routing != "interleaved" or args.ordering != "age":
        p.error("Require window >= 2 and interleaved age ordering")
    validate(args)
    cases = {case.xi2: case for case in load_cases(args)}
    a, b = (cases[value] for value in args.xi2)
    full = replace(a, n_dimers=2*a.n_dimers)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"config.txt").write_text(
        repr(vars(args)) + f"\nTotal dimers={full.n_dimers}; bank size={a.n_dimers}\n"
        "matched_AA/BB: duplicated deterministic identical-bank features. "
        "homogeneous_A: one round-robin bank over the full budget. "
        "dual_AB: separate interleaved banks exposed to the same input.\n"
    )
    rows = []
    if "windowed" in args.tasks:
        windowed(args, a, b, full, rows)
    if "zoh" in args.tasks:
        zoh(args, a, b, full, rows)


if __name__ == "__main__":
    main()
