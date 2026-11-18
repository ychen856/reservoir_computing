#!/usr/bin/env python3
"""Train M-PSK readouts from cached features, with validation-only selection."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys

import numpy as np
from scipy.linalg import svd

VERSION = 1
DEFAULT_RIDGES = [0., 1e-10, 1e-8, 1e-6, 1e-4, .01, .1, 1., 10., 100., 1000., 10000., 100000., 1000000.]


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_features(path, source):
    prefix = "X" if source == "reservoir" else "raw"
    with np.load(path, allow_pickle=False) as archive:
        modern = f"{prefix}_train" in archive
        keys = {"train": (f"{prefix}_train", "y_train") if modern else (prefix, "y"),
                "validation": (f"{prefix}_validation", "y_validation"),
                "test": (f"{prefix}_test", "y_test")}
        data = {}
        for split, (xkey, ykey) in keys.items():
            if split == "test" and (xkey not in archive or ykey not in archive):
                continue
            if xkey not in archive or ykey not in archive:
                raise ValueError(f"{path}: missing {xkey} or {ykey}")
            x = np.asarray(archive[xkey], dtype=float)
            labels = archive[ykey]
            if (x.ndim != 2 or labels.ndim != 1 or len(x) != len(labels) or not len(x)
                    or not np.isfinite(x).all() or not np.issubdtype(labels.dtype, np.integer)
                    or (labels < 0).any()):
                raise ValueError(f"Invalid {split} features or labels in {path}")
            data[split] = (x, labels.astype(int))
    if len({x.shape[1] for x, _ in data.values()}) != 1:
        raise ValueError(f"Feature dimensions disagree in {path}")
    classes = np.unique(data["train"][1])
    if not np.array_equal(classes, np.arange(len(classes))) or len(classes) < 2:
        raise ValueError("Train labels must contain every class numbered 0 through M-1")
    if any((y >= len(classes)).any() for _, y in data.values()):
        raise ValueError("Validation/test contains an unseen class")
    return data, len(classes)


def metadata(path):
    sidecar = path.parent / "metadata.json"
    if sidecar.exists():
        return json.loads(sidecar.read_text())
    # Legacy archives use the original config.json above their case/task directory.
    for directory in [path.parent, *path.parents]:
        config_path = directory / "config.json"
        if config_path.exists():
            c = json.loads(config_path.read_text())
            if any(len(c.get(key, [])) != 1 for key in ("seeds", "xi2", "design_indices", "psk_orders", "conditions")):
                raise ValueError(f"Ambiguous legacy metadata in {config_path}. Supply a metadata.json beside features.npz.")
            return dict(seed=c["seeds"][0], xi2=c["xi2"][0], design_index=c["design_indices"][0],
                        order=c["psk_orders"][0], condition=c["conditions"][0], n_dimers=c["n_dimers"],
                        topology="coupled" if c.get("coupling_l_nh") is not None else "open")
    # Single stand-alone archives can be fitted without circuit metadata.
    return dict(seed=0, xi2=1., design_index="unknown", order="unknown", condition="unspecified", n_dimers="unknown", topology="unknown")


def feature_map(x, kind):
    if kind == "linear":
        return x
    if kind == "square":
        return np.column_stack((x, x * x))
    first, second = np.triu_indices(x.shape[1])
    return np.column_stack((x, x[:, first] * x[:, second]))


def standardizer(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    return mean, np.where(std < 1e-12, 1., std)


def prepare_map(train, kind):
    base_mean, base_std = standardizer(train)
    mapped = feature_map((train - base_mean) / base_std, kind)
    # Preserve the existing original linear feature convention.
    map_mean, map_std = (np.zeros(mapped.shape[1]), np.ones(mapped.shape[1])) if kind == "linear" else standardizer(mapped)
    return (mapped - map_mean) / map_std, dict(base_mean=base_mean, base_std=base_std,
                                             map_mean=map_mean, map_std=map_std)


def transform(x, kind, state):
    return (feature_map((x - state["base_mean"]) / state["base_std"], kind) - state["map_mean"]) / state["map_std"]


def predictions(x, model):
    kind = str(model["feature_map"].item())
    return transform(x, kind, model) @ model["W"][1:] + model["W"][0]


def fit_maps(data, classes, ridges, original_ridge, maps):
    train, y = data["train"]
    validation, yv = data["validation"]
    target = np.eye(classes)[y]
    ymean = target.mean(axis=0)
    candidates, best_models, best_scores = [], {}, {}
    for kind in dict.fromkeys(["linear", *maps]):
        x, state = prepare_map(train, kind)
        xv = transform(validation, kind, state)
        xmean = x.mean(axis=0)
        u, singular, vt = svd(x - xmean, full_matrices=False, check_finite=False, lapack_driver="gesdd")
        projection = u.T @ (target - ymean)
        threshold = np.finfo(float).eps * max(x.shape) * (singular[0] if len(singular) else 0.)
        lambdas = sorted(set([*ridges, original_ridge] if kind == "linear" else ridges))
        for ridge in lambdas:
            factors = np.zeros_like(singular)
            if ridge == 0:
                np.divide(1., singular, out=factors, where=singular > threshold)
            else:
                factors = singular / (singular ** 2 + ridge)
            weights = (vt.T * factors) @ projection
            bias = ymean - xmean @ weights
            scores = xv @ weights + bias
            accuracy = float(np.mean(scores.argmax(axis=1) == yv))
            candidates.append(dict(feature_map=kind, ridge=ridge, validation_accuracy=accuracy))
            families = []
            if kind == "linear" and ridge == original_ridge:
                families.append("original_linear")
            if kind == "linear" and ridge in ridges:
                families.append("tuned_linear")
            if kind in maps and kind != "linear" and ridge in ridges:
                families.append("best_poly")
            # Ties prefer fewer features, then stronger regularization.
            merit = (accuracy, -x.shape[1], ridge)
            for family in families:
                if family not in best_scores or merit > best_scores[family]:
                    best_scores[family] = merit
                    best_models[family] = dict(**state, W=np.vstack((bias, weights)), feature_map=np.array(kind),
                                               ridge=np.array(ridge), classes=np.array(classes),
                                               validation_accuracy=np.array(accuracy))
    return candidates, best_models


def evaluate_test(feature_path, source, model_path, output):
    data, classes = load_features(feature_path, source)
    if "test" not in data:
        return dict(test_status="missing_test_features", test_accuracy="", ser="", n_test=0)
    with np.load(model_path, allow_pickle=False) as saved:
        model = dict(saved)
    x, y = data["test"]
    scores = predictions(x, model)
    prediction = scores.argmax(axis=1)
    confusion = np.zeros((classes, classes), dtype=int)
    np.add.at(confusion, (y, prediction), 1)
    atomic_npz(output, y=y, prediction=prediction, scores=scores, confusion=confusion)
    accuracy = float(np.mean(prediction == y))
    return dict(test_status="ok", test_accuracy=accuracy, ser=1 - accuracy, n_test=len(y))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, required=True, help="One NPZ or root containing features.npz files")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sources", nargs="+", choices=["reservoir", "input_only"], default=["reservoir", "input_only"])
    p.add_argument("--maps", nargs="+", choices=["square", "quadratic"], default=["square", "quadratic"])
    p.add_argument("--ridges", nargs="+", type=float, default=DEFAULT_RIDGES)
    p.add_argument("--original-ridge", type=float, default=1e-6)
    p.add_argument("--expected-seeds", nargs="+", type=int)
    p.add_argument("--expected-xi2", nargs="+", type=float)
    p.add_argument("--validation-only", action="store_true", help="Fit and select without reading test features")
    a = p.parse_args()
    if not np.isfinite([*a.ridges, a.original_ridge]).all() or min([*a.ridges, a.original_ridge]) < 0:
        p.error("Ridges must be finite and nonnegative")
    paths = [a.features] if a.features.is_file() else sorted(a.features.rglob("features.npz"))
    if not paths:
        p.error("No features.npz found")
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output / ".readout.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another readout process is using this output directory")
        records = []
        expected_grids = {}
        signature = dict(version=VERSION, script_sha256=digest(Path(__file__)), ridges=a.ridges,
                         original_ridge=a.original_ridge, maps=a.maps)
        for path in paths:
            meta = metadata(path)
            identity = {key: meta[key] for key in ("design_index", "topology", "n_dimers", "order", "condition", "xi2", "seed")}
            identity.update(snr_db=meta.get("snr_db", "unspecified"), isi_tap=meta.get("isi_tap", "unspecified"),
                            isi_spacing_cycles=meta.get("isi_spacing_cycles", "unspecified"))
            expected_grids[str(path.resolve())] = (meta.get("grid_seeds", [meta["seed"]]),
                                                       meta.get("grid_xi2", [meta["xi2"]]))
            for source in dict.fromkeys(a.sources):
                key = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
                out = a.output / "fits" / key / source
                out.mkdir(parents=True, exist_ok=True)
                fingerprint = dict(**signature, feature_sha256=digest(path), source=source)
                marker = out / "complete.json"
                cached = json.loads(marker.read_text()) if marker.exists() else {}
                families = ("original_linear", "tuned_linear", "best_poly")
                good = cached.get("fingerprint") == fingerprint
                if good:
                    try:
                        for family in families:
                            model_path = out / f"{family}.npz"
                            if digest(model_path) != cached["model_sha256"][family]:
                                good = False
                    except (OSError, KeyError):
                        good = False
                if not good:
                    data, classes = load_features(path, source)
                    if meta["order"] != "unknown" and classes != int(meta["order"]):
                        raise ValueError(f"Class count disagrees with metadata in {path}")
                    candidates, models = fit_maps(data, classes, a.ridges, a.original_ridge, a.maps)
                    write_csv(out / "validation_grid.csv", candidates)
                    scores = {}
                    for family, model in models.items():
                        atomic_npz(out / f"{family}.npz", **model)
                        scores[family] = {"validation_accuracy": float(model["validation_accuracy"]),
                                          "ridge": float(model["ridge"]), "feature_map": str(model["feature_map"].item())}
                    cached = dict(fingerprint=fingerprint, scores=scores,
                                  model_sha256={family: digest(out / f"{family}.npz") for family in models})
                    atomic_json(marker, cached)
                    print(f"Fit {path} [{source}]", flush=True)
                else:
                    print(f"Reuse {path} [{source}]", flush=True)
                for family, scores in cached["scores"].items():
                    records.append(dict(**identity, source=source, readout=family, **scores,
                                        features_path=str(path.resolve()), model_path=str((out / f"{family}.npz").resolve())))
        write_csv(a.output / "validation_all.csv", records)
        groups = {}
        group_keys = ("design_index", "topology", "n_dimers", "order", "condition", "snr_db", "isi_tap", "isi_spacing_cycles", "source", "readout")
        for row in records:
            groups.setdefault(tuple(row[key] for key in group_keys), []).append(row)
        selected, aggregates, incomplete = [], [], 0
        for group, rows in groups.items():
            seeds = set(a.expected_seeds) if a.expected_seeds else {seed for r in rows for seed in expected_grids[r["features_path"]][0]}
            xi_values = set(a.expected_xi2) if a.expected_xi2 else {xi for r in rows for xi in expected_grids[r["features_path"]][1]}
            points = {xi: [r for r in rows if r["xi2"] == xi] for xi in xi_values}
            if any(len(runs) != len(seeds) or {r["seed"] for r in runs} != seeds for runs in points.values()):
                print(f"Incomplete grid; selection pending: {group}", file=sys.stderr)
                incomplete += 1
                continue
            # Mean validation accuracy across seeds chooses one physical xi2 per family.
            best_xi = max(sorted(points), key=lambda xi: np.mean([r["validation_accuracy"] for r in points[xi]]))
            chosen = points[best_xi]
            validation_mean = float(np.mean([r["validation_accuracy"] for r in chosen]))
            evaluated = []
            for row in chosen:
                result = dict(row, selected_xi2=best_xi, validation_mean=validation_mean)
                if not a.validation_only:
                    model_path = Path(row["model_path"])
                    result.update(evaluate_test(Path(row["features_path"]), row["source"], model_path,
                                                model_path.with_name(model_path.stem + "_test.npz")))
                selected.append(result)
                if result.get("test_status") == "ok":
                    evaluated.append(result["test_accuracy"])
            aggregate = dict(zip(group_keys, group), selected_xi2=best_xi, validation_mean=validation_mean,
                             n_seeds=len(chosen), test_seeds=len(evaluated))
            if len(evaluated) == len(chosen):
                aggregate.update(test_accuracy_mean=float(np.mean(evaluated)),
                                 test_accuracy_std=float(np.std(evaluated, ddof=1)) if len(evaluated) > 1 else 0.,
                                 ser_mean=1 - float(np.mean(evaluated)))
            aggregates.append(aggregate)
        write_csv(a.output / "selected_results.csv", selected)
        write_csv(a.output / "summary.csv", aggregates)
        print(f"Selected {len(aggregates)} groups; {incomplete} incomplete groups. Output: {a.output}")
        if incomplete:
            sys.exit(2)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted. Completed readout fits are retained.", file=sys.stderr)
        sys.exit(130)
