"""Continue the ideal EP locus to a specified constant coupling ESR.

Run from the reservoir_computing root after placing this file in src/.
The output describes small-signal series-RC coupling requirements.
"""
import argparse
import csv
import sys

import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import root

from src.run_passive_gqc_ep import solve_ep


def characteristic(fr, h, kappa, gamma1, esr_ratio):
    first = Polynomial([1, gamma1, 1])
    second = Polynomial([1, h, fr**-2])
    series = Polynomial([1, esr_ratio * kappa])
    return series * first * second + Polynomial([0, 0, kappa]) * (first + second)


def continue_ep(fr, h, kappa, gamma1, esr_ratio, u, v):
    state = np.array([fr, h, -u / 2, np.sqrt(v - u*u / 4)])
    for resistance in np.linspace(0, esr_ratio, 101)[1:]:
        def residual(candidate):
            frequency_ratio, damping, real, imag = candidate
            polynomial = characteristic(frequency_ratio, damping, kappa, gamma1, resistance)
            pole = complex(real, imag)
            value = polynomial(pole)
            derivative = polynomial.deriv()(pole)
            return np.array([value.real, value.imag, derivative.real, derivative.imag])

        fit = root(residual, state, tol=1e-10)
        error = np.linalg.norm(residual(fit.x), ord=np.inf)
        if error > 1e-11 or not np.isfinite(fit.x).all():
            raise RuntimeError(f"EP continuation failed at Rs/Z={resistance:g}, residual={error:g}")
        state = fit.x
        if state[0] <= 0 or state[1] <= 0 or state[2] >= 0 or state[3] <= 0:
            raise RuntimeError("Continuation left the positive-resistance stable oscillatory branch")

    fr, h, real, imag = state
    # Independently construct the physical five-state matrix with currents
    # scaled by Z. Eigenvalues are normalized by omega1.
    b = fr**-2
    r = esr_ratio
    if r > 0:
        matrix = np.array([
            [-gamma1-1/r, 1/r, -1, 0, 1/r],
            [1/(b*r), -(h+1/r)/b, 0, -1/b, -1/(b*r)],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [1/(r*kappa), -1/(r*kappa), 0, 0, -1/(r*kappa)],
        ])
        poles = np.linalg.eigvals(matrix)
    else:
        poles = characteristic(fr, h, kappa, gamma1, r).roots()
        matrix = None

    pole = complex(real, imag)
    nearest = np.argsort(abs(poles - pole))[:2]
    gap = abs(poles[nearest[0]] - poles[nearest[1]])
    if max(poles.real) >= 0 or max(abs(poles[nearest] - pole)) > 1e-4:
        raise RuntimeError("Independent eigenvalue verification failed")
    if matrix is not None:
        singular = np.linalg.svd(matrix-pole*np.eye(5), compute_uv=False)
        if singular[-1] / singular[0] > 1e-8 or singular[-2] / singular[0] < 1e-7:
            raise RuntimeError("Expected a one-dimensional eigenspace at the repeated pole")
    return fr, h, real, imag, gap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1", type=float, default=136)
    parser.add_argument("--esr-ohm", type=float, default=45.00967355423566)
    args = parser.parse_args()
    inductance, c1 = 0.4e-9, 10e-12
    impedance = np.sqrt(inductance / c1)
    if (not np.isfinite([args.r1, args.esr_ohm]).all()
            or args.r1 <= impedance/2 or args.esr_ohm < 0):
        parser.error("Require R1 > Z/2 and ESR >= 0, both finite")

    gamma1 = impedance / args.r1
    ideal_max = (1-gamma1**2/4)**-0.5
    output = csv.writer(sys.stdout)
    output.writerow(["ideal_fr", "esr_ohm", "fr", "c2_pf", "kappa_ep",
                     "cc_ep_pf", "r2_ep_ohm", "frequency_over_f1",
                     "decay_over_omega1", "normalized_pole_gap"])
    for fraction in (0.10, 0.35, 0.60, 0.85, 0.97):
        ideal_fr = 1 + fraction*(ideal_max-1)
        kappa, h, u, v = solve_ep(ideal_fr, gamma1)
        fr, h, real, imag, gap = continue_ep(
            ideal_fr, h, kappa, gamma1, args.esr_ohm/impedance, u, v)
        output.writerow([ideal_fr, args.esr_ohm, fr, c1/fr**2*1e12,
                         kappa, kappa*c1*1e12, impedance/h, imag, -real, gap])


if __name__ == "__main__":
    main()
