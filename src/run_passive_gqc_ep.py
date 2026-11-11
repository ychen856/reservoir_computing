"""Sweep the EP locus of the ideal, zero-ESR Passive-GQC dimer."""

import argparse
import csv
import sys

import numpy as np
from scipy.optimize import brentq

from src.circuits.passive_gqc import PassiveGQCDesign


def solve_ep(fr, gamma1):
    """Return kappa_EP and h_EP, where h = gamma2 / fr."""
    c2_ratio = fr**-2

    def ep_conditions(kappa):
        a = 1 + kappa
        b = c2_ratio + kappa
        determinant = a * b - kappa**2
        d = np.sqrt(determinant)

        # Match the quartic to D * (lambda**2 + u*lambda + v)**2.
        # The linear coefficient gives u; the cubic coefficient gives h.
        h = gamma1 * (d - b) / (a - d)
        u = (gamma1 + h) / (2 * d)
        v = 1 / d
        residual = (a + b + gamma1 * h) / determinant - (u**2 + 2 * v)
        return residual, h, u, v

    delta = 1 - c2_ratio
    # At the upper bracket h = 0. Interior roots have finite positive R2.
    kappa_limit = (delta + np.sqrt(delta**2 + 4 * delta * c2_ratio)) / 2
    kappa = brentq(lambda k: ep_conditions(k)[0], 0, kappa_limit, xtol=1e-15)
    _, h, u, v = ep_conditions(kappa)
    if h <= 0 or u**2 >= 4 * v:
        raise ValueError(f"No finite positive-R2 oscillatory EP at fr={fr:g}")
    return kappa, h, u, v


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1", type=float, help="R1 in ohms. Default reproduces gamma1=0.5.")
    args = parser.parse_args()

    inductance = 0.4e-9
    c1 = 10e-12
    omega1 = 1 / np.sqrt(inductance * c1)
    impedance = np.sqrt(inductance / c1)
    r1 = impedance / 0.5 if args.r1 is None else args.r1
    if not np.isfinite(r1) or r1 <= impedance / 2:
        parser.error("R1 must be finite and give 0 < gamma1 < 2")

    gamma1 = impedance / r1
    fr_max = (1 - gamma1**2 / 4)**-0.5
    print(f"gamma1={gamma1:.9g}, fr_max={fr_max:.10f}", file=sys.stderr)

    output = csv.writer(sys.stdout)
    output.writerow([
        "fr", "c2_pf", "kappa_ep", "cc_ep_pf", "r2_ep_ohm",
        "gamma2_ep", "frequency_over_f1", "decay_over_omega1",
        "normalized_pole_gap",
    ])

    for fraction in (0.10, 0.35, 0.60, 0.85, 0.97):
        fr = 1 + fraction * (fr_max - 1)
        kappa, h, u, v = solve_ep(fr, gamma1)
        cc = kappa * c1
        r2 = impedance / h

        design = PassiveGQCDesign(inductance, omega1, r1, cc)
        point = design.at_frequency_ratio(fr)
        coefficients = point.characteristic_coefficients(r2)
        quadratic = np.array([1.0, u, v])
        np.testing.assert_allclose(
            coefficients / coefficients[0],
            np.convolve(quadratic, quadratic),
            rtol=1e-9,
            atol=1e-12,
        )

        poles = point.poles(r2) / omega1
        upper = poles[poles.imag > 0]
        gap = abs(upper[0] - upper[1])
        output.writerow([
            fr, point.c2 * 1e12, kappa, cc * 1e12, r2,
            fr * h, np.sqrt(v - u**2 / 4), u / 2, gap,
        ])


if __name__ == "__main__":
    main()