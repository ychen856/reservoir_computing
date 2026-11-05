"""Shared configuration, age ordering and complete result persistence."""
import argparse
import hashlib
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from src.circuits.pt_single import PTSingleReservoir
from src.circuits.pt_parallel import PTParallelReservoir
from src.simulation.runner import simulate_continuous_task
from src.utils.features import reorder_features_by_age

@dataclass(frozen=True)
class Case:
    architecture: str
    n_pairs: int
    mode: str
    ordering: str
    eta: float
    alpha: float
    seed: int

    @property
    def key(self):
        return (f'{self.architecture}_pairs{self.n_pairs}_{self.mode}_{self.ordering}'
                f'_eta{self.eta:g}_alpha{self.alpha:g}_seed{self.seed}')

    def reservoir(self):
        if self.architecture == 'single':
            return PTSingleReservoir(self.eta, self.alpha)
        return PTParallelReservoir(self.n_pairs, eta=self.eta, alpha=self.alpha, mode=self.mode)


def parser(kind, tasks):
    p = argparse.ArgumentParser(description=f'Parallel {kind}; all settings/results are saved.')
    p.add_argument('--tasks', nargs='+', choices=tasks, default=list(tasks))
    p.add_argument('--pairs', type=int, nargs='+', default=[1, 2, 4])
    p.add_argument('--etas', type=float, nargs='+', default=[0.5])
    p.add_argument('--alpha', type=float, default=8.33)
    p.add_argument('--seeds', type=int, nargs='+', default=[42])
    p.add_argument('--modes', nargs='+', choices=['interleaved', 'broadcast'], default=['interleaved'])
    p.add_argument('--orderings', nargs='+', choices=['age', 'physical'], default=['age'])
    p.add_argument('--no-single', action='store_true')
    p.add_argument('--symbols', type=int, default=3000)
    p.add_argument('--washout', type=int, default=200)
    p.add_argument('--train-ratio', type=float, default=.8)
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--points-per-symbol', type=int, default=16)
    p.add_argument('--cycles-per-symbol', type=float, default=None,
                   help='Override task-specific default: properties .5, benchmarks see README')
    p.add_argument('--amplitude', type=float, default=1e-4)
    p.add_argument('--ridge', type=float, default=1e-6)
    p.add_argument('--output', type=Path, default=Path('results')/f'parallel_{kind}')
    p.add_argument('--smoke', action='store_true', help='Small pipeline check, not scientific results')
    p.add_argument('--overwrite', action='store_true', help='Permit overwriting this output directory')
    return p


def validate(a):
    if a.smoke:
        a.symbols, a.washout = 180, 40
        if hasattr(a, 'permutations'):
            a.permutations, a.max_delay, a.nonlinear_max_delay, a.cross_max_delay = 2, 5, 5, 4
            a.trials, a.trial_symbols, a.convergence_symbols = 8, 12, 48
    if (a.symbols <= a.washout+30 or a.washout < 0 or not 0 < a.train_ratio < 1
        or any(n < 1 for n in a.pairs) or a.k < 1 or a.points_per_symbol < a.k+1
        or a.ridge < 0 or a.alpha < 0 or (a.cycles_per_symbol is not None and a.cycles_per_symbol <= 0)):
        raise ValueError('Invalid timing, count, split, or readout settings')
    if not all(np.isfinite(v) for v in [*a.etas, a.alpha, a.amplitude, a.ridge]):
        raise ValueError('Settings must be finite')


def cases(a):
    for seed in dict.fromkeys(a.seeds):
        for eta in dict.fromkeys(a.etas):
            if not a.no_single:
                yield Case('single', 1, 'single', 'physical', eta, a.alpha, seed)
            for n in dict.fromkeys(a.pairs):
                for mode in dict.fromkeys(a.modes):
                    for ordering in (a.orderings if mode == 'interleaved' else ['physical']):
                        yield Case('parallel', n, mode, ordering, eta, a.alpha, seed)


def simulate(case, u, a, cycles=.5):
    u = np.asarray(u, dtype=float)
    ts = (a.cycles_per_symbol if a.cycles_per_symbol is not None else cycles)/48e6
    sim = simulate_continuous_task(case.reservoir(), SimpleNamespace(u=u, y=np.zeros(len(u))),
            symbol_period=ts, points_per_symbol=a.points_per_symbol, k=a.k,
            amplitude_scale=a.amplitude, encoding='zoh', solver='exact_zoh')
    if sim.X.shape != (len(u), 2*case.n_pairs*a.k):
        raise AssertionError('Incorrect parallel feature dimension')
    if case.ordering == 'age':
        sim.X = reorder_features_by_age(sim.X, case.n_pairs, a.k,
                                        np.arange(len(u)))
    if not np.isfinite(sim.X).all():
        raise RuntimeError('Nonfinite features')
    return sim


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


class Results:
    def __init__(self, a):
        self.root = a.output
        if self.root.exists() and any(self.root.iterdir()) and not a.overwrite:
            raise FileExistsError(f'{self.root} is nonempty; choose --output or --overwrite')
        self.root.mkdir(parents=True, exist_ok=True)
        self.rows = []
        config = vars(a).copy()
        config['output'] = str(a.output)
        config['protocol'] = 'ZOH; fixed ridge; train-only normalization; age reordered before split'
        config['smoke_results_are_not_final'] = a.smoke
        source_root = Path(__file__).resolve().parents[1]
        config['source_sha256'] = {
            str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [source_root/'circuits/pt_single.py', source_root/'circuits/pt_parallel.py',
                         source_root/'simulation/runner.py', source_root/'utils/features.py',
                         source_root/'parallel/common.py', source_root/'parallel/properties.py',
                         source_root/'parallel/benchmarks.py']
        }
        (self.root/'config.json').write_text(json.dumps(config, indent=2))

    def directory(self, case, task):
        d = self.root/case.key/task
        d.mkdir(parents=True, exist_ok=True)
        return d

    def add(self, case, task, **values):
        row = dict(asdict(case), task=task, **values)
        self.rows.append(row)
        write_csv(self.root/'summary.csv', self.rows)
        # Checkpoint after every result, not only after the entire sweep.
        print(case.key, task, values, flush=True)

    def fail(self, case, task, exc):
        self.add(case, task, status='failed', error=f'{type(exc).__name__}: {exc}')
