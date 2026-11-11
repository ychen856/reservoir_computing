"""Select application operating points by validation and score their test sets."""

from collections import defaultdict

import numpy as np

from src.pass_gqc.application_waveforms import EXPERIMENTS, mpsk_trials, shape_trials, waveform_features
from src.pass_gqc.common import Results, load_cases, parser, simulate, validate, write_csv
from src.readout.preprocessing import fit_standardizer, transform_standardizer
from src.readout.ridge import predict_ridge_regression, train_ridge_regression
from src.tasks.applications.pam4 import quantize_pam4_class
from src.tasks.applications.pam4_nonlinear import generate_pam4_nonlinear_task


def variants(a, task):
    if task == "mpsk":
        for order in dict.fromkeys(a.psk_orders):
            for condition in dict.fromkeys(a.conditions):
                detail = f"_snr{a.snr_db:g}" if condition == "awgn" else ""
                yield f"{order}psk_{condition}{detail}", (order, condition)
    elif task == "pam4":
        for channel in dict.fromkeys(a.pam4_channels):
            yield f"pam4_{channel}", channel
    else:
        for experiment in dict.fromkeys(a.experiments):
            yield f"waveform_{experiment}", experiment


def fit(train, labels, classes, scalar, ridge):
    mean, std = fit_standardizer(train)
    target = labels if scalar else np.eye(classes)[labels]
    weight = train_ridge_regression(transform_standardizer(train, mean, std), target, ridge)
    return weight, mean, std


def predict(features, model, scalar):
    weight, mean, std = model
    scores = predict_ridge_regression(transform_standardizer(features, mean, std), weight)
    return quantize_pam4_class(scores) if scalar else np.argmax(scores, axis=1)


def pam4_features(case, channel, a):
    nonlinear = channel == "nonlinear"
    task = generate_pam4_nonlinear_task(
        n_symbols=a.symbols, seed=case.seed, washout=a.washout,
        quadratic=0.036*a.distortion_strength if nonlinear else 0.0,
        cubic=-0.011*a.distortion_strength if nonlinear else 0.0,
        add_awgn=a.pam4_snr_db is not None,
        snr_db=a.pam4_snr_db if a.pam4_snr_db is not None else 40.0,
        noise_seed=np.random.SeedSequence([case.seed, 3]),
    )
    radius = 2
    ids = np.arange(max(a.washout, radius), len(task.u)-radius)
    train_end = int(len(ids)*a.train_ratio)
    validation_end = train_end + int(len(ids)*a.validation_ratio)
    # Adjacent partitions use disjoint transmitted symbols in the ISI channel.
    train = ids[:train_end-radius]
    validation = ids[train_end+radius:validation_end-radius]
    test = ids[validation_end+radius:]
    if min(len(train), len(validation), len(test)) < 3:
        raise ValueError("Too few PAM4 symbols after channel purge")
    # The task generator returns normalized data. Rescale using only training.
    received = task.u / np.max(np.abs(task.u[train]))
    features = simulate(case, received, a).X
    return dict(X=features, raw=received[:, None], y=task.y,
                train=train, validation=validation, test=test)


def trial_features(case, task, variant, a, split):
    seed = np.random.SeedSequence([case.seed, split])
    if task == "mpsk":
        waves, y, t, observation = mpsk_trials(a, *variant, case.frequency, seed)
    else:
        waves, y, t, observation = shape_trials(a, variant, case.frequency, seed)
    X, raw, times = waveform_features(case, waves, t, observation, a)
    return X, raw, y, times


def evaluate(case, task, variant, name, a, out):
    path = out.directory(case, name)
    scalar = task == "pam4"
    if scalar:
        data = pam4_features(case, variant, a)
        np.savez_compressed(path/"features.npz", **data)
        X, raw, y = (data[k] for k in ("X", "raw", "y"))
        train = X[data["train"]], raw[data["train"]], y[data["train"]]
        validation = X[data["validation"]], raw[data["validation"]], y[data["validation"]]
    else:
        X, raw, y, times = trial_features(case, task, variant, a, 0)
        Xv, rawv, yv, _ = trial_features(case, task, variant, a, 1)
        train, validation = (X, raw, y), (Xv, rawv, yv)
        np.savez_compressed(path/"features.npz", X=X, raw=raw, y=y,
                            X_validation=Xv, raw_validation=rawv, y_validation=yv, sample_times_s=times)
    classes = variant[0] if task == "mpsk" else 4
    models = {}
    accuracies = {}
    for i, method in enumerate(("reservoir", "input_only")):
        model = fit(train[i], train[2], classes, scalar, a.ridge)
        train_prediction = predict(train[i], model, scalar)
        prediction = predict(validation[i], model, scalar)
        accuracy = float(np.mean(prediction == validation[2]))
        out.add(case, name, status="ok", method=method, metric="accuracy",
                train=float(np.mean(train_prediction == train[2])), validation=accuracy,
                feature_dim=train[i].shape[1], n_train=len(train[2]), n_validation=len(validation[2]))
        weight, mean, std = model
        np.savez_compressed(path/f"{method}_validation.npz", W=weight, mean=mean, std=std,
                            y=validation[2], prediction=prediction)
        models[method] = model
        accuracies[method] = accuracy
    return accuracies["reservoir"], models


def score_selected(groups, task, a, out):
    rows = []
    for (_, name), points in groups.items():
        point = max(points.values(), key=lambda runs: np.mean([run[2] for run in runs]))
        validation_mean = float(np.mean([run[2] for run in point]))
        for case, variant, _, models in point:
            path = out.directory(case, name)
            scalar = task == "pam4"
            if scalar:
                with np.load(path/"features.npz") as data:
                    ids = data["test"]
                    X, raw, y = (data[k][ids] for k in ("X", "raw", "y"))
            else:
                X, raw, y, _ = trial_features(case, task, variant, a, 2)
            for method, features in (("reservoir", X), ("input_only", raw)):
                prediction = predict(features, models[method], scalar)
                classes = variant[0] if task == "mpsk" else 4
                confusion = np.zeros((classes, classes), dtype=int)
                np.add.at(confusion, (y, prediction), 1)
                accuracy = float(np.mean(prediction == y))
                np.savez_compressed(path/f"{method}_test.npz", y=y, prediction=prediction, confusion=confusion)
                rows.append(dict(task=name, mode=case.mode, design_index=case.design_index,
                                 xi2=case.xi2, r2_ohm=case.r2_ohm, r2_ep_ohm=case.r2_ep_ohm,
                                 seed=case.seed, n_dimers=case.n_dimers, routing=case.routing,
                                 ordering=case.ordering, method=method,
                                 reservoir_validation_mean=validation_mean,
                                 test=accuracy, ser=1-accuracy, n_test=len(y)))
    write_csv(out.root/"selected_test.csv", rows)


def main(task):
    p = parser("applications", (task,), task)
    p.set_defaults(xi2=[0.5, 1, 2, 4, 8, 12])
    if task == "pam4":
        p.add_argument("--validation-ratio", type=float, default=0.1)
        p.add_argument("--pam4-channels", nargs="+", choices=("linear", "nonlinear"),
                       default=["linear", "nonlinear"])
        p.add_argument("--distortion-strength", type=float, default=1.0)
        p.add_argument("--pam4-snr-db", type=float)
    else:
        p.add_argument("--repeats", type=int, default=20, help="Trials per class per split")
        p.add_argument("--samples-per-cycle", type=int, default=16)
        if task == "mpsk":
            p.add_argument("--psk-orders", type=int, nargs="+", default=[4, 8, 16, 32])
            p.add_argument("--conditions", nargs="+", choices=("clean", "awgn", "isi"),
                           default=["clean", "awgn", "isi"])
            p.add_argument("--snr-db", type=float, default=20.0)
            p.add_argument("--isi-tap", type=float, default=0.5)
            p.add_argument("--isi-spacing-cycles", type=float, default=2.0)
        else:
            p.add_argument("--experiments", nargs="+", choices=tuple(EXPERIMENTS),
                           default=list(EXPERIMENTS))
    a = p.parse_args()
    validate(a)
    if task == "pam4":
        if not 0 < a.validation_ratio < 1-a.train_ratio:
            p.error("train-ratio plus validation-ratio must be below 1")
        if not np.isfinite(a.distortion_strength) or a.distortion_strength < 0:
            p.error("distortion-strength must be finite and nonnegative")
        if a.pam4_snr_db is not None and not np.isfinite(a.pam4_snr_db):
            p.error("pam4-snr-db must be finite")
    else:
        if a.smoke:
            a.repeats = 5 if task == "waveform_classification" else 2
            if task == "mpsk" and a.psk_orders == [4, 8, 16, 32]:
                a.psk_orders = [4]
        if a.repeats < (5 if task == "waveform_classification" else 1) or a.samples_per_cycle < 4:
            p.error("Too few trials or waveform samples")
        if task == "mpsk" and (min(a.psk_orders) < 2 or not np.isfinite([a.snr_db, a.isi_tap]).all()
                               or not 0 < a.isi_spacing_cycles < 12):
            p.error("Invalid MPSK order, noise or ISI settings")
    cases = load_cases(a)
    a.protocol = ("PAM4 ZOH and chronological purged splits" if task == "pam4"
                  else "Piecewise-linear trial inputs; independent train/validation/test seeds; trial state reset")
    out = Results(a)
    groups = defaultdict(lambda: defaultdict(list))
    for case in cases:
        for name, variant in variants(a, task):
            try:
                accuracy, models = evaluate(case, task, variant, name, a, out)
                groups[case.mode, name][case.design_index, case.xi2].append(
                    (case, variant, accuracy, models))
            except Exception as error:
                out.fail(case, name, error)
    out.finish()
    score_selected(groups, task, a, out)
