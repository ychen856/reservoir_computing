"""Passive-GQC benchmarks with the existing generators and evaluators."""
from collections import defaultdict

import numpy as np

from src.pass_gqc.common import parser, validate, load_cases, simulate, Results, write_csv
from src.tasks.benchmarks.suite import TASKS, tasks, evaluate
from src.task_evaluation import history_features


def split_indices(task, a, gap):
    valid = np.arange(task.washout, len(task.u))
    train_end = int(len(valid) * task.train_ratio)
    validation_end = train_end + int(len(valid) * a.validation_ratio)
    return valid[:train_end], valid[train_end+gap:validation_end], valid[validation_end+gap:]


def score_selected(candidates, a, out):
    rows = []
    for (mode, name), points in candidates.items():
        eligible = [(point, runs) for point, runs in points.items()
                    if len(runs) == len(set(a.seeds))
                    and all(np.isfinite(run[2]["validation"]) for run in runs)]
        if not eligible:
            continue
        def merit(entry):
            _, runs = entry
            scores = np.mean([run[2]["validation"] for run in runs])
            return scores if runs[0][2]["metric"] == "accuracy" else -scores

        _, runs = max(eligible, key=merit)
        for case, family, values in runs:
            task = dict(tasks(family, a, case.seed))[name]
            gap = task.target_offset if family == "mackey_glass" else 0
            train, _, test = split_indices(task, a, gap)
            directory = out.directory(case, name)
            with np.load(directory/"features_targets.npz") as data:
                X = data["X"]
            for method, features in [("reservoir", X), *(
                    (f"input_history{taps}", history_features(task.u, taps))
                    for taps in a.history_taps)]:
                test_values, arrays = evaluate(features, task, a, gap, split=(train, test))
                np.savez_compressed(directory/f"{method}_test.npz", **arrays)
                rows.append(dict(design_index=case.design_index, xi2=case.xi2,
                                 r2_ohm=case.r2_ohm, r2_ep_ohm=case.r2_ep_ohm,
                                 n_dimers=case.n_dimers, routing=case.routing,
                                 ordering=case.ordering, mode=mode, task=name,
                                 seed=case.seed, method=method, **test_values))
    write_csv(out.root/"selected_test.csv", rows)


def main(argv=None, *, task=None):
    p = parser("benchmarks", TASKS, task)
    p.add_argument("--validation-ratio", type=float, default=0.1 if task else 0.0,
                   help="Fraction of post-washout symbols used for validation; 0 enables the legacy split")
    p.add_argument("--windows", type=int, nargs="+", default=[2, 3, 5, 10])
    p.add_argument("--delays", type=int, nargs="+", default=[1, 2, 5, 10])
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 20])
    p.add_argument("--history-taps", type=int, nargs="+", default=[1, 10, 16])
    a = p.parse_args(argv)
    validate(a)
    if a.smoke and "mackey_glass" in a.tasks and a.horizons == [1, 5, 10, 20]:
        a.horizons = [min(a.horizons)]
    if not 0 <= a.validation_ratio < 1-a.train_ratio:
        p.error("train-ratio plus validation-ratio must be below 1")
    if task and not a.validation_ratio:
        p.error("Task-specific entry points require a validation split")
    if min(a.windows) < 2 or min(a.delays+a.horizons+a.history_taps) < 1:
        p.error("Invalid window, delay, horizon or history count")
    if max(a.history_taps)-1 > a.washout:
        p.error("washout must cover history taps")
    cases = load_cases(a)
    out = Results(a)
    candidates = defaultdict(lambda: defaultdict(list))
    for case in cases:
        for family in a.tasks:
            try:
                variants = list(tasks(family, a, case.seed))
                cached_u, cached_X = None, None
                for name, task in variants:
                    if cached_u is None or not np.array_equal(task.u, cached_u):
                        cached_u = task.u.copy()
                        cached_X = simulate(case, task.u, a).X
                    gap = task.target_offset if family == "mackey_glass" else 0
                    split = None
                    if a.validation_ratio:
                        train, validation, test = split_indices(task, a, gap)
                        if min(len(train), len(validation), len(test)) < 3:
                            raise ValueError("Not enough samples after purge for train/validation/test")
                        split = (train, validation)
                    values, arrays = evaluate(cached_X, task, a, gap, split=split)
                    if split is not None:
                        values["validation"] = values.pop("test")
                        values["n_validation"] = values.pop("n_test")
                    directory = out.directory(case, name)
                    np.savez_compressed(directory/"predictions.npz", **arrays)
                    np.savez_compressed(directory/"features_targets.npz", X=cached_X, u=task.u, y=task.y)
                    out.add(case, name, status="ok", method="reservoir", **values)
                    for taps in a.history_taps:
                        X = history_features(task.u, taps)
                        baseline, baseline_arrays = evaluate(X, task, a, gap, split=split)
                        if split is not None:
                            baseline["validation"] = baseline.pop("test")
                            baseline["n_validation"] = baseline.pop("n_test")
                        np.savez_compressed(directory/f"history{taps}.npz", **baseline_arrays)
                        out.add(case, name, status="ok", method=f"input_history{taps}", **baseline)
                    if split is not None:
                        candidates[case.mode, name][case.design_index, case.xi2].append(
                            (case, family, values))
            except Exception as error:
                out.fail(case, family, error)
    out.finish()
    if a.validation_ratio:
        score_selected(candidates, a, out)


if __name__ == "__main__":
    main()
