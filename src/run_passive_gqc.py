"""Single-dimer Passive-GQC pilot using the existing readout and metrics."""

import argparse
import json
from types import SimpleNamespace

import numpy as np

from src.circuits.passive_gqc import GrapheneVaractor, PassiveGQCDesign, PassiveGQCReservoir
from src.simulation.runner import simulate_continuous_task
from src.simulation.evaluation import evaluate_continuous_regression
from src.tasks.benchmarks.narma10 import generate_narma10
from src.tasks.properties.memory_capacity import evaluate_linear_memory_capacity
from src.utils.encoding import sample_hold_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c2-pf", type=float, nargs="+", default=[10.0],
                        help="Effective shunt capacitance candidates, e.g. 9 9.5 10 10.5 11")
    parser.add_argument("--r2", type=float, nargs="+", default=[136.0])
    parser.add_argument("--mode", choices=("linear", "graphene"), default="linear")
    parser.add_argument("--bias-voltage", type=float, default=0.0)
    parser.add_argument("--no-esr", action="store_true")
    parser.add_argument("--symbols", type=int, default=1000)
    parser.add_argument("--washout", type=int, default=100)
    parser.add_argument("--cycles", type=float, default=0.5)
    parser.add_argument("--amplitude-a", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()
    if (args.washout < 10 or args.symbols - args.washout < 20
            or not np.isfinite([args.cycles, args.amplitude_a, args.bias_voltage]).all()
            or args.cycles <= 0 or args.amplitude_a <= 0
            or any(not np.isfinite(v) or v <= 0 for v in args.c2_pf + args.r2)):
        parser.error("Require positive finite parameters, washout >= 10 and at least 20 scored symbols")

    c1, inductance = 10e-12, 0.4e-9
    omega1 = 1 / np.sqrt(inductance * c1)
    device = GrapheneVaractor() if args.mode == "graphene" else None
    c_gqc, esr = (device.capacitance_and_resistance(args.bias_voltage)
                  if device is not None else (10e-12, 0.0))
    design = PassiveGQCDesign(inductance, omega1, 136.0, c_gqc)
    ts = args.cycles / (omega1 / (2 * np.pi))
    u = np.random.default_rng(args.seed).uniform(-1, 1, args.symbols)
    narma = generate_narma10(n_samples=args.symbols, washout=args.washout,
                            train_ratio=0.8, seed=args.seed)
    split = dict(washout=args.washout, train_ratio=0.8, ridge=1e-6, normalize=True)
    for c2_pf in args.c2_pf:
        fr = np.sqrt(c1 / (c2_pf * 1e-12))
        for r2 in args.r2:
            reservoir = PassiveGQCReservoir(design.at_frequency_ratio(fr), r2, device,
                                            args.bias_voltage, not args.no_esr)
            config = dict(symbol_period=ts, amplitude_scale=args.amplitude_a, solver="exact_zoh")
            mc_sim = simulate_continuous_task(reservoir, SimpleNamespace(u=u, y=u), **config)
            mc = evaluate_linear_memory_capacity(mc_sim.X, u, max_delay=10, **split)
            narma_sim = simulate_continuous_task(reservoir, narma, **config)
            score = evaluate_continuous_regression(narma_sim.X, narma_sim.y, **split)
            drive = sample_hold_input(u[:args.washout], ts, args.amplitude_a)
            t = mc_sim.t_eval[:args.washout * 16]
            span = (0, args.washout * ts)
            x0 = np.zeros(reservoir.state_size)
            if reservoir.state_size == 5:
                x0[4] = 0.2
            x0[:2] = [0.1, -0.1]
            perturbed = reservoir.simulate(drive, span, t, x0=x0, solver="exact_zoh", symbol_period=ts)
            distance = np.linalg.norm(perturbed.y[:2] - mc_sim.solutions.y[:2, :len(t)], axis=0)
            print(json.dumps(dict(mode=args.mode, c2_pf=c2_pf, fr=fr, r2_ohm=r2,
                                  c_gqc_pf=c_gqc*1e12, esr_ohm=0.0 if args.no_esr else esr,
                                  bias_voltage=args.bias_voltage, symbol_period_s=ts,
                                  amplitude_a=args.amplitude_a, seed=args.seed,
                                  feature_dim=mc_sim.X.shape[1],
                                  linear_mc_train=mc.total_train_capacity,
                                  linear_mc_test=mc.total_test_capacity,
                                  narma_train_nrmse=score.train_nrmse,
                                  narma_test_nrmse=score.test_nrmse,
                                  convergence_final_v=float(distance[-1]),
                                  convergence_tail_rms_v=float(np.sqrt(np.mean(distance[-16:]**2))))))


if __name__ == "__main__":
    main()
