"""Shared circuit construction and experiment persistence."""
import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.circuits.passive_gqc import GrapheneVaractor, PassiveGQCDesign, PassiveGQCReservoir
from src.circuits.passive_gqc_parallel import PassiveGQCParallelReservoir
from src.simulation.runner import simulate_continuous_task
from src.utils.csv_output import write_csv
from src.utils.features import reorder_features_by_age


@dataclass(frozen=True)
class ConstantVaractor:
    capacitance: float
    resistance: float

    def capacitance_and_resistance(self, voltage):
        return self.capacitance, self.resistance


@dataclass(frozen=True)
class Case:
    design_index: int
    seed: int
    mode: str
    width_um: float
    length_um: float
    bias_v: float
    c2_pf: float
    r2_ohm: float
    r1_ohm: float = 136.0
    l_nh: float = 0.4
    c1_pf: float = 10.0

    r2_ep_ohm: float = 0.0
    xi2: float = 1.0
    n_dimers: int = 1
    routing: str = "broadcast"
    ordering: str = "physical"

    @property
    def key(self):
        base = f"design{self.design_index:04d}_{self.mode}_seed{self.seed}"
        if self.n_dimers > 1:
            base += f"_{self.n_dimers}dimers_{self.routing}_{self.ordering}"
        return base if self.xi2 == 1.0 else f"{base}_xi2_{self.xi2!r}"

    def pole_diagnostics(self):
        device = GrapheneVaractor(width=self.width_um*1e-6, length=self.length_um*1e-6)
        capacitance, resistance = device.capacitance_and_resistance(self.bias_v)
        z = np.sqrt(self.l_nh*1e-9/(self.c1_pf*1e-12))
        r = resistance/z
        b = self.c2_pf/self.c1_pf
        kappa = capacitance/(self.c1_pf*1e-12)
        h, gamma = z/self.r2_ohm, z/self.r1_ohm
        matrix = np.array([
            [-gamma-1/r, 1/r, -1, 0, 1/r],
            [1/(b*r), -(h+1/r)/b, 0, -1/b, -1/(b*r)],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [1/(r*kappa), -1/(r*kappa), 0, 0, -1/(r*kappa)],
        ])
        poles = np.linalg.eigvals(matrix)*2*np.pi*self.frequency
        maximum = float(max(poles.real))
        upper = poles[poles.imag > 0]
        gap = float(abs(upper[0]-upper[1])) if len(upper) == 2 else None
        return dict(poles_per_s=[dict(real=float(p.real), imag=float(p.imag)) for p in poles],
                    max_real_pole_per_s=maximum, oscillatory_pair_gap_per_s=gap,
                    slowest_decay_tau_s=(-1/maximum if maximum < 0 else None),
                    small_signal_stable=maximum < 0)


    @property
    def frequency(self):
        return 1 / (2*np.pi*np.sqrt(self.l_nh*1e-9*self.c1_pf*1e-12))

    def reservoir(self):
        device = GrapheneVaractor(width=self.width_um*1e-6, length=self.length_um*1e-6)
        capacitance, resistance = device.capacitance_and_resistance(self.bias_v)
        if self.mode == "matched_linear":
            device = ConstantVaractor(capacitance, resistance)
        omega = 2*np.pi*self.frequency
        design = PassiveGQCDesign(self.l_nh*1e-9, omega, self.r1_ohm, capacitance)
        point = design.at_frequency_ratio(np.sqrt(self.c1_pf/self.c2_pf))
        dimer = PassiveGQCReservoir(point, self.r2_ohm, device, self.bias_v, True)
        if self.n_dimers == 1:
            return dimer
        return PassiveGQCParallelReservoir((dimer,) * self.n_dimers, self.routing)


def parser(kind, tasks, task=None):
    p = argparse.ArgumentParser(description=f"Passive-GQC {kind}")
    p.add_argument("--ep-table", type=Path, required=True, help="Device candidate CSV from run_passive_gqc_device_ep")
    p.add_argument("--design-indices", nargs="+", type=int, default=[37, 5] if task else None, help="Zero-based data row indices in the supplied CSV")
    p.add_argument("--modes", nargs="+", choices=["graphene", "matched_linear"], default=["graphene", "matched_linear"])
    p.add_argument("--xi2", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0, 8.0] if task else [1.0],
                   help="Normalized loss R2_EP/R2; 1 is the EP reference")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44] if task else [44])
    if task is None:
        p.add_argument("--tasks", nargs="+", choices=tasks, default=list(tasks))
    else:
        p.set_defaults(tasks=[task])
    p.add_argument("--n-dimers", type=int, default=1)
    p.add_argument("--routing", choices=["broadcast", "interleaved"], default="broadcast")
    p.add_argument("--ordering", choices=["physical", "age"], default="physical")
    p.add_argument("--symbols", type=int, default=3000)
    p.add_argument("--washout", type=int, default=300)
    p.add_argument("--train-ratio", type=float, default=0.8,
                   help="Held-out split is named test for compatibility; use it as validation during design selection")
    p.add_argument("--cycles-per-symbol", type=float, default=0.5)
    p.add_argument("--amplitude", type=float, default=1e-3, help="Input current scale in A")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--points-per-symbol", type=int, default=16)
    p.add_argument("--ridge", type=float, default=1e-6)
    p.add_argument("--output", type=Path, default=Path("results")/f"pass_gqc_{task or kind}")
    p.add_argument("--smoke", action="store_true")
    return p


def validate(a):
    if a.smoke:
        a.symbols, a.washout = 160, 40
    finite = [a.train_ratio, a.cycles_per_symbol, a.amplitude, a.ridge, *a.xi2]
    if (not np.isfinite(finite).all() or a.symbols <= a.washout+30
            or a.washout < 0 or not 0 < a.train_ratio < 1
            or min(a.cycles_per_symbol, a.amplitude) <= 0 or a.ridge < 0
            or a.n_dimers < 1 or (a.ordering == "age" and a.routing != "interleaved")
            or min(a.xi2) <= 0 or a.k < 1 or a.points_per_symbol < a.k+1):
        raise ValueError("Invalid timing, split, sampling or readout configuration")


def load_cases(a):
    with a.ep_table.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = set(a.design_indices) if a.design_indices is not None else set(range(len(rows)))
    if not selected or min(selected) < 0 or max(selected) >= len(rows):
        raise ValueError("design-indices must select existing CSV data rows")
    designs = []
    for index in sorted(selected):
        row = rows[index]
        if row.get("status") != "candidate":
            raise ValueError(f"Row {index} must have status=candidate")
        fields = {name: float(row[name]) for name in
                  ("width_um", "length_um", "bias_v", "c2_pf", "r1_ohm", "l_nh", "c1_pf")}
        fields["r2_ohm"] = float(row["r2_ep_ohm"])
        positive = [value for name, value in fields.items() if name != "bias_v"]
        if not np.isfinite(list(fields.values())).all() or min(positive) <= 0:
            raise ValueError(f"Invalid physical parameters in row {index}")
        device = GrapheneVaractor(width=fields["width_um"]*1e-6, length=fields["length_um"]*1e-6)
        capacitance, resistance = device.capacitance_and_resistance(fields["bias_v"])
        np.testing.assert_allclose([capacitance*1e12, resistance],
                                   [float(row["c_gqc_pf"]), float(row["esr_ohm"])], rtol=1e-8)
        # Recheck the small-signal EP with the current device/circuit settings.
        from src.run_passive_gqc_ep_esr import characteristic
        z = np.sqrt(fields["l_nh"]*1e-9/(fields["c1_pf"]*1e-12))
        fr = np.sqrt(fields["c1_pf"]/fields["c2_pf"])
        poly = characteristic(fr, z/fields["r2_ohm"], capacitance/(fields["c1_pf"]*1e-12),
                              z/fields["r1_ohm"], resistance/z)
        pole = complex(-float(row["decay_over_omega1"]), float(row["frequency_over_f1"]))
        residual = max(abs(poly(pole)), abs(poly.deriv()(pole)))
        if residual > 1e-9 or pole.real >= 0 or pole.imag <= 0:
            raise ValueError(f"EP validation failed for row {index}")
        for seed in dict.fromkeys(a.seeds):
            for mode in dict.fromkeys(a.modes):
                for xi2 in dict.fromkeys(a.xi2):
                    point_fields = dict(fields, r2_ohm=fields["r2_ohm"]/xi2)
                    case = Case(index, seed, mode, r2_ep_ohm=fields["r2_ohm"],
                                xi2=xi2, n_dimers=a.n_dimers, routing=a.routing,
                                ordering=a.ordering, **point_fields)
                    if not case.pole_diagnostics()["small_signal_stable"]:
                        raise ValueError(f"Unstable small-signal point: {case.key}")
                    designs.append(case)
    frequencies = np.array([case.frequency for case in designs])
    if not np.allclose(frequencies, frequencies[0], rtol=1e-12, atol=0):
        raise ValueError("Selected designs must share omega1 to preserve a common physical sampling clock")
    return designs


def symbol_period(case, a):
    return a.cycles_per_symbol/case.frequency


def simulate(case, u, a):
    result = simulate_continuous_task(
        case.reservoir(), SimpleNamespace(u=np.asarray(u), y=np.zeros(len(u))),
        symbol_period=symbol_period(case, a), points_per_symbol=a.points_per_symbol,
        k=a.k, amplitude_scale=a.amplitude, encoding="zoh", solver="exact_zoh")
    if case.ordering == "age":
        result.X = reorder_features_by_age(result.X, case.n_dimers, a.k)
    return result


class Results:
    def __init__(self, a):
        self.root = a.output
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError("Output directory is nonempty; choose a new --output")
        self.root.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.args = a
        self.diagnostics = {}
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(a).items()}
        config.setdefault("protocol", "Current ZOH; fixed omega1 clock; train-only normalization; held-out results for design validation")
        config["device_defaults"] = asdict(GrapheneVaractor())
        config["ep_table_sha256"] = hashlib.sha256(a.ep_table.read_bytes()).hexdigest()
        (self.root/"ep_table.csv").write_bytes(a.ep_table.read_bytes())
        source = Path(__file__).resolve().parents[1]
        paths = list((source/"pass_gqc").glob("*.py"))
        paths += [source/name for name in ("circuits/passive_gqc.py", "circuits/passive_gqc_parallel.py",
                                          "simulation/runner.py",
                                          "utils/features.py", "tasks/properties/suite.py", "tasks/benchmarks/suite.py",
                                          "utils/csv_output.py")]
        config["source_sha256"] = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in paths}
        (self.root/"config.json").write_text(json.dumps(config, indent=2))

    def directory(self, case, task):
        directory = self.root/case.key/task
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def add(self, case, task, **values):
        device = GrapheneVaractor(width=case.width_um*1e-6, length=case.length_um*1e-6)
        capacitance, resistance = device.capacitance_and_resistance(case.bias_v)
        if case.key not in self.diagnostics:
            diagnostics = case.pole_diagnostics()
            tau = diagnostics["slowest_decay_tau_s"]
            washout_s = self.args.washout*symbol_period(case, self.args)
            diagnostics["washout_s"] = washout_s
            diagnostics["washout_over_tau"] = washout_s/tau if tau is not None else None
            self.diagnostics[case.key] = diagnostics
            (self.directory(case, "diagnostics")/"poles.json").write_text(
                json.dumps(diagnostics, indent=2))
        diagnostics = self.diagnostics[case.key]
        self.rows.append(dict(asdict(case),
                              max_real_pole_per_s=diagnostics["max_real_pole_per_s"],
                              slowest_decay_tau_s=diagnostics["slowest_decay_tau_s"],
                              washout_over_tau=diagnostics["washout_over_tau"],
                              oscillatory_pair_gap_per_s=diagnostics["oscillatory_pair_gap_per_s"],
                              fr=float(np.sqrt(case.c1_pf/case.c2_pf)),
                              c_gqc_pf=capacitance*1e12, esr_ohm=resistance,
                              symbol_period_s=symbol_period(case, self.args), task=task, **values))
        write_csv(self.root/"summary.csv", self.rows)
        print(case.key, task, values, flush=True)

    def fail(self, case, task, error):
        self.add(case, task, status="failed", error=f"{type(error).__name__}: {error}")

    def finish(self):
        groups = {}
        identifiers = ("design_index", "mode", "xi2", "task", "method",
                       "channel", "forcing", "perturbation", "n_dimers", "routing", "ordering")
        metrics = ("train", "validation", "test", "train_mc", "test_mc", "excess_total",
                   "dimension", "rank", "tau_s", "fit_r_squared",
                   "tail_rms_voltage", "final_state_distance")
        for row in self.rows:
            if row.get("status") == "ok":
                key = tuple(row.get(name) for name in identifiers)
                groups.setdefault(key, []).append(row)
        summary = []
        for key, rows in groups.items():
            item = dict(zip(identifiers, key))
            item.update(r2_ohm=rows[0]["r2_ohm"], r2_ep_ohm=rows[0]["r2_ep_ohm"],
                        seeds=",".join(str(row["seed"]) for row in rows),
                        seed_count=len(rows), expected_seed_count=len(set(self.args.seeds)))
            for name in metrics:
                if name in rows[0]:
                    values = np.asarray([row[name] for row in rows], dtype=float)
                    item[name+"_mean"] = float(np.mean(values))
                    item[name+"_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            summary.append(item)
        write_csv(self.root/"sweep_summary.csv", summary)
        if any(row.get("status") == "failed" for row in self.rows):
            raise SystemExit("Some cases failed; inspect summary.csv")
