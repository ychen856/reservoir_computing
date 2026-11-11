"""Search graphene device designs on the EP branch continued from zero ESR.

Geometry and component bounds are screening assumptions, not process limits.
Run as python -m src.run_passive_gqc_device_ep after placing this file in src/.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from src.circuits.passive_gqc import GrapheneVaractor
from src.run_passive_gqc_ep import solve_ep
from src.run_passive_gqc_ep_esr import continue_ep


def ideal_seed(kappa, gamma1):
    """Invert the known ideal oscillatory EP branch, excluding endpoints."""
    fr_max = (1 - gamma1**2/4)**-0.5
    lower = 1 + (fr_max-1)*1e-3
    upper = 1 + (fr_max-1)*(1-1e-4)
    k_lower = solve_ep(lower, gamma1)[0]
    k_upper = solve_ep(upper, gamma1)[0]
    if not k_lower < kappa < k_upper:
        return None
    fr = brentq(lambda value: solve_ep(value, gamma1)[0]-kappa,
                lower, upper, xtol=1e-14)
    _, h, u, v = solve_ep(fr, gamma1)
    return fr, h, u, v


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1", type=float, default=136)
    parser.add_argument("--widths-um", type=float, nargs="+", default=[2, 5, 10, 20, 50, 100])
    parser.add_argument("--lengths-um", type=float, nargs="+", default=[0.1, 0.2, 0.5, 1, 2, 5, 10])
    parser.add_argument("--biases-v", type=float, nargs="+", default=[-1, -0.5, 0, 0.5, 1])
    parser.add_argument("--c2-bounds-pf", type=float, nargs=2, default=[9, 11])
    parser.add_argument("--r2-bounds-ohm", type=float, nargs=2, default=[10, 10000])
    parser.add_argument("--output", type=Path, default=Path("results/passive_gqc/device_ep.csv"))
    args = parser.parse_args()
    inductance, c1 = 0.4e-9, 10e-12
    impedance = np.sqrt(inductance/c1)
    positive = [args.r1, *args.widths_um, *args.lengths_um,
                *args.c2_bounds_pf, *args.r2_bounds_ohm]
    if (not np.isfinite(positive).all() or min(positive) <= 0
            or not np.isfinite(args.biases_v).all() or args.r1 <= impedance/2
            or args.c2_bounds_pf[0] >= args.c2_bounds_pf[1]
            or args.r2_bounds_ohm[0] >= args.r2_bounds_ohm[1]):
        parser.error("Invalid finite parameter values or component bounds")
    gamma1 = impedance/args.r1
    rows = []
    for width in args.widths_um:
        for length in args.lengths_um:
            device = GrapheneVaractor(width=width*1e-6, length=length*1e-6)
            for bias in args.biases_v:
                capacitance, resistance = device.capacitance_and_resistance(bias)
                kappa = capacitance/c1
                row = dict(width_um=width, length_um=length, bias_v=bias,
                           r1_ohm=args.r1, l_nh=inductance*1e9, c1_pf=c1*1e12,
                           c_gqc_pf=capacitance*1e12, esr_ohm=resistance,
                           kappa=kappa, status="outside_ideal_seed_branch",
                           fr="", c2_pf="", r2_ep_ohm="", frequency_over_f1="",
                           decay_over_omega1="", normalized_pole_gap="", detail="")
                seed = ideal_seed(kappa, gamma1)
                if seed is not None:
                    fr, h, u, v = seed
                    try:
                        fr, h, real, imag, gap = continue_ep(
                            fr, h, kappa, gamma1, resistance/impedance, u, v)
                        c2_pf, r2 = c1/fr**2*1e12, impedance/h
                        inside = (args.c2_bounds_pf[0] <= c2_pf <= args.c2_bounds_pf[1]
                                  and args.r2_bounds_ohm[0] <= r2 <= args.r2_bounds_ohm[1])
                        row.update(status="candidate" if inside else "outside_component_bounds",
                                   fr=fr, c2_pf=c2_pf, r2_ep_ohm=r2,
                                   frequency_over_f1=imag, decay_over_omega1=-real,
                                   normalized_pole_gap=gap)
                    except RuntimeError as error:
                        row.update(status="continuation_unresolved", detail=str(error))
                rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    candidates = [row for row in rows if row["status"] == "candidate"]
    candidates.sort(key=lambda row: row["fr"])
    candidate_path = args.output.with_name(args.output.stem+"_candidates.csv")
    with candidate_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(candidates)
    for status in sorted(set(row["status"] for row in rows)):
        print(f"{status}: {sum(row['status'] == status for row in rows)}")
    print(args.output)
    print(candidate_path)


if __name__ == "__main__":
    main()
