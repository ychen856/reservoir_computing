"""Unloaded, small-signal Passive NH dimer with equal inductors."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PassiveGQCDesign:
    """Fix omega1 and R1, then select fr before sweeping R2.

    omega1 is in rad/s, inductance in H, capacitance in F, and resistance
    in ohms. coupling_capacitance is the differential GQC capacitance at
    the chosen bias. This model assumes zero GQC series resistance.
    """

    inductance: float
    omega1: float
    r1: float
    coupling_capacitance: float

    def __post_init__(self):
        _positive(self.inductance, self.omega1, self.r1)
        if not math.isfinite(self.coupling_capacitance) or self.coupling_capacitance < 0:
            raise ValueError("coupling_capacitance must be finite and nonnegative")

    @property
    def c1(self):
        return 1 / (self.inductance * self.omega1**2)

    def at_frequency_ratio(self, fr):
        _positive(fr)
        return PassiveGQCOperatingPoint(self, fr)


@dataclass(frozen=True)
class PassiveGQCOperatingPoint:
    design: PassiveGQCDesign
    fr: float

    @property
    def c2(self):
        return self.design.c1 / self.fr**2

    def _matrices(self, r2):
        _positive(r2)
        d = self.design
        kappa = d.coupling_capacitance / d.c1
        mass = np.array([[1 + kappa, -kappa], [-kappa, self.fr**-2 + kappa]])
        scale = math.sqrt(d.inductance / d.c1)
        damping = np.diag([scale / d.r1, scale / r2])
        return mass, damping

    def state_matrix(self, r2):
        """Return the matrix for [V1, V2, dV1/dtau, dV2/dtau], tau=omega1*t."""
        mass, damping = self._matrices(r2)
        identity = np.eye(2)
        return np.block([
            [np.zeros((2, 2)), identity],
            [-np.linalg.solve(mass, identity), -np.linalg.solve(mass, damping)],
        ])

    def characteristic_coefficients(self, r2):
        """Return descending coefficients in lambda = j*Omega."""
        mass, damping = self._matrices(r2)
        a, b = mass.diagonal()
        g1, g2 = damping.diagonal()
        return np.array([
            a * b - mass[0, 1]**2,
            a * g2 + b * g1,
            a + b + g1 * g2,
            g1 + g2,
            1.0,
        ])

    def poles(self, r2):
        """Return continuous-time poles in inverse seconds."""
        return self.design.omega1 * np.linalg.eigvals(self.state_matrix(r2))


def _positive(*values):
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("parameters must be finite and positive")
