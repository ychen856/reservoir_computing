#!/usr/bin/env python3
"""Checkpoint Design 5 open M-PSK features using the repository's waveform code."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import zipfile

import numpy as np

SCHEMA = 1
SPLITS = ("train", "validation", "test")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_chunk(path, labels, width, k, times):
    try:
        with np.load(path, allow_pickle=False) as data:
            return (data["X"].shape == (len(labels), width)
                    and data["raw"].shape == (len(labels), k)
                    and np.array_equal(data["y"], labels)
                    and np.array_equal(data["times"], times)
                    and np.isfinite(data["X"]).all() and np.isfinite(data["raw"]).all())
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path.cwd())
    p.add_argument("--ep-table", type=Path, default=Path("results/pass_gqc_r2_extended/ep_table.csv"))
    p.add_argument("--output", type=Path, default=Path("results/pass_gqc/mpsk_d5_open_20reps"))
    p.add_argument("--dimers", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--orders", type=int, nargs="+", default=[4, 8, 16, 32])
    p.add_argument("--conditions", nargs="+", choices=["clean", "awgn", "isi"], default=["clean", "awgn", "isi"])
    p.add_argument("--xi2", type=float, nargs="+", default=[.5, 2, 12]) #default=[.5, 1, 2, 4, 8, 12])
    p.add_argument("--seeds", type=int, nargs="+", default=[42]) #default=[42, 43, 44])
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--chunk-size", type=int, default=16, help="Trials committed per checkpoint")
    p.add_argument("--amplitude", type=float, default=.001)
    p.add_argument("--cycles-per-symbol", type=float, default=.5)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--samples-per-cycle", type=int, default=16)
    p.add_argument("--snr-db", type=float, default=20.)
    p.add_argument("--isi-tap", type=float, default=.5)
    p.add_argument("--isi-spacing-cycles", type=float, default=2.)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    numeric = [*a.xi2, a.amplitude, a.cycles_per_symbol, a.snr_db, a.isi_tap, a.isi_spacing_cycles]
    if (not np.isfinite(numeric).all() or min(a.dimers) < 1 or min(a.orders) < 2
            or min(a.xi2) <= 0 or min(a.seeds) < 0 or min(a.repeats, a.chunk_size, a.k) < 1
            or a.samples_per_cycle < 4 or min(a.amplitude, a.cycles_per_symbol) <= 0
            or not 0 < a.isi_spacing_cycles < 12):
        p.error("Invalid dimensions, repeats, circuit timing, noise or ISI parameters")
    a.repo = a.repo.resolve()
    a.ep_table = (a.repo / a.ep_table).resolve()
    a.output = (a.repo / a.output).resolve()
    sys.path.insert(0, str(a.repo))
    required = [a.repo / "src/pass_gqc/common.py", a.repo / "src/pass_gqc/application_waveforms.py", a.ep_table]
    if any(not path.is_file() for path in required):
        p.error("Missing repository source or EP table: " + ", ".join(str(x) for x in required if not x.is_file()))
    from src.pass_gqc.common import parser, load_cases, validate
    from src.pass_gqc.application_waveforms import mpsk_trials, waveform_features

    # Reuse the installed repository parser so new required Case fields retain defaults.
    base = parser("applications", ("mpsk",), "mpsk").parse_args([
        "--ep-table", str(a.ep_table), "--design-indices", "5", "--modes", "graphene",
        "--seeds", *map(str, dict.fromkeys(a.seeds)), "--xi2", *map(str, dict.fromkeys(a.xi2)),
        "--routing", "interleaved", "--ordering", "age", "--output", str(a.output)])
    vars(base).update(amplitude=a.amplitude, cycles_per_symbol=a.cycles_per_symbol,
                      k=a.k, samples_per_cycle=a.samples_per_cycle, repeats=a.repeats,
                      snr_db=a.snr_db, isi_tap=a.isi_tap, isi_spacing_cycles=a.isi_spacing_cycles)
    # Hash source dependencies. Readout scripts are deliberately outside this fingerprint.
    source_paths = sorted({*a.repo.joinpath("src/pass_gqc").glob("*.py"),
                           *a.repo.joinpath("src/circuits").glob("*.py"),
                           *a.repo.joinpath("src/simulation").glob("*.py"),
                           *a.repo.joinpath("src/utils").glob("*.py"),
                           *a.repo.joinpath("src/tasks/applications").glob("*.py")})
    source_hash = {str(x.relative_to(a.repo)): digest(x) for x in source_paths}
    jobs = []
    for count in dict.fromkeys(a.dimers):
        base.n_dimers = count
        validate(base)
        for case in load_cases(base):
            if case.design_index != 5 or case.mode != "graphene":
                raise ValueError("Expected Design 5 graphene cases")
            # Guard against an installed parser accidentally enabling inter-dimer links.
            if getattr(base, "coupling_l_nh", None) is not None:
                raise ValueError("Open run requires coupling_l_nh=None")
            for order in dict.fromkeys(a.orders):
                for condition in dict.fromkeys(a.conditions):
                    jobs.append((case, order, condition))
    print(f"{len(jobs)} cases; {a.repeats} trials/class/split; splits=train,validation,test", flush=True)
    if a.dry_run:
        for case, order, condition in jobs:
            print(f"d{case.n_dimers} {order}PSK {condition} xi2={case.xi2:g} seed={case.seed}")
        return
    a.output.mkdir(parents=True, exist_ok=True)
    # Advisory locks are released by the OS after Ctrl-C, process failure or reboot.
    with (a.output / ".features.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another feature process is using this output directory")
        for case, order, condition in jobs:
            config = dict(schema=SCHEMA, design_index=5, mode="graphene", topology="open",
                          n_dimers=case.n_dimers, order=order, condition=condition, xi2=case.xi2,
                          seed=case.seed, repeats=a.repeats, chunk_size=a.chunk_size,
                          routing="interleaved", ordering="age", amplitude=a.amplitude,
                          cycles_per_symbol=a.cycles_per_symbol, k=a.k, samples_per_cycle=a.samples_per_cycle,
                          snr_db=a.snr_db, isi_tap=a.isi_tap, isi_spacing_cycles=a.isi_spacing_cycles,
                          ep_table_sha256=digest(a.ep_table), source_sha256=source_hash,
                          feature_protocol="repository mpsk_trials/waveform_features; reset each trial",
                          runner_schema=SCHEMA, runner_sha256=digest(Path(__file__)),
                          grid_seeds=list(dict.fromkeys(a.seeds)), grid_xi2=list(dict.fromkeys(a.xi2)))
            fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
            directory = a.output / f"{order}psk_{condition}" / f"{case.n_dimers}dimers" / f"xi2_{case.xi2:g}" / f"seed_{case.seed}"
            directory.mkdir(parents=True, exist_ok=True)
            meta = directory / "metadata.json"
            if meta.exists() and json.loads(meta.read_text()) != config:
                raise ValueError(f"Configuration/source changed in {directory}. Use another --output to preserve this run.")
            atomic_json(meta, config)
            print(str(directory.relative_to(a.output)), flush=True)
            arrays = {}
            for split_id, split in enumerate(SPLITS):
                # Generate the complete split before chunking to preserve the original RNG sequence.
                waves, labels, t, observation = mpsk_trials(base, order, condition, case.frequency,
                                                          np.random.SeedSequence([case.seed, split_id]))
                times = np.linspace(*observation, a.k)
                chunks = directory / "checkpoints" / split
                chunks.mkdir(parents=True, exist_ok=True)
                features, raw = [], []
                for start in range(0, len(labels), a.chunk_size):
                    end = min(start + a.chunk_size, len(labels))
                    path = chunks / f"{start:06d}_{end:06d}.npz"
                    expected = labels[start:end]
                    if not check_chunk(path, expected, 2 * a.k * case.n_dimers, a.k, times):
                        X, received, sample_times = waveform_features(case, waves[start:end], t, observation, base)
                        atomic_npz(path, X=X, raw=received, y=expected, times=sample_times)
                        if not check_chunk(path, expected, 2 * a.k * case.n_dimers, a.k, times):
                            raise RuntimeError(f"Invalid feature checkpoint {path}")
                        print(f"  {split} {end}/{len(labels)}", flush=True)
                    with np.load(path, allow_pickle=False) as chunk:
                        features.append(chunk["X"])
                        raw.append(chunk["raw"])
                arrays.update({f"X_{split}": np.concatenate(features), f"raw_{split}": np.concatenate(raw),
                               f"y_{split}": labels})
            arrays.update(sample_times_s=times, fingerprint=np.array(fingerprint))
            destination = directory / "features.npz"
            # Avoid rewriting identical complete archives, preserving independent readout caches.
            complete = directory / ".complete.json"
            if not (complete.exists() and destination.exists()
                    and json.loads(complete.read_text()).get("sha256") == digest(destination)):
                atomic_npz(destination, **arrays)
                atomic_json(complete, dict(fingerprint=fingerprint, sha256=digest(destination)))
            print("  complete", flush=True)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted. Completed checkpoints are retained.", file=sys.stderr)
        sys.exit(130)
