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


@dataclass(frozen=True)
class GrapheneVaractor:
    """Behavioral C(V), R(V) from graphene_varactor.lib, with SI parameters."""

    width: float = 100e-6
    length: float = 10e-6
    oxide_thickness: float = 10e-9
    relative_permittivity: float = 20.0
    temperature: float = 300.0
    fermi_velocity: float = 1e6
    electron_mobility: float = 0.30
    hole_mobility: float = 0.22
    residual_density: float = 5e15
    neutrality_voltage: float = 0.0
    contact_resistance_width: float = 1e-3
    smoothing_voltage: float = 1e-6

    def __post_init__(self):
        _positive(self.width, self.length, self.oxide_thickness,
                  self.relative_permittivity, self.temperature, self.fermi_velocity,
                  self.electron_mobility, self.hole_mobility, self.residual_density,
                  self.smoothing_voltage)
        if not math.isfinite(self.neutrality_voltage):
            raise ValueError("neutrality_voltage must be finite")
        if not math.isfinite(self.contact_resistance_width) or self.contact_resistance_width < 0:
            raise ValueError("contact_resistance_width must be finite and nonnegative")

    def capacitance_and_resistance(self, voltage):
        """Return differential capacitance dQ/dV and lumped ESR."""
        q, kb, hbar, eps0 = 1.602176634e-19, 1.380649e-23, 1.054571817e-34, 8.8541878128e-12
        area = self.width * self.length
        cox_density = eps0 * self.relative_permittivity / self.oxide_thickness
        cox = area * cox_density
        dos_scale = math.pi * (hbar * self.fermi_velocity)**2
        beta = q**3 / (dos_scale * cox_density)
        overdrive = voltage - self.neutrality_voltage
        magnitude = math.hypot(overdrive, self.smoothing_voltage)
        vq = 2 * overdrive / (1 + math.sqrt(1 + 4 * beta * magnitude))
        thermal_energy = kb * self.temperature
        x = abs(q * vq / (2 * thermal_energy))
        cq = area * 4 * q*q * thermal_energy / dos_scale * (x + math.log1p(math.exp(-2*x)))
        capacitance = cox * cq / (cox + cq)

        net = q*q * vq * math.hypot(vq, self.smoothing_voltage) / dos_scale
        majority = (math.hypot(net, self.residual_density) + abs(net)) / 2
        minority = self.residual_density**2 / (4 * majority)
        electrons, holes = (majority, minority) if net >= 0 else (minority, majority)
        sheet = 1 / (q * (self.electron_mobility * electrons + self.hole_mobility * holes))
        resistance = self.contact_resistance_width / (2 * self.width)
        resistance += sheet * self.length / (12 * self.width)
        return capacitance, resistance


@dataclass(frozen=True)
class PassiveGQCReservoir:
    """Current drive into V1, observing V1 and V2 in volts.

    States are V1, V2, I1, I2, and (with ESR) intrinsic GQC AC voltage.
    bias_voltage is an ideal DC offset across the varactor. Bias-network
    loading and source/load impedances are outside this model.
    """

    operating_point: PassiveGQCOperatingPoint
    r2: float
    varactor: GrapheneVaractor = None
    bias_voltage: float = 0.0
    include_esr: bool = True

    voltage_state_indices = (0, 1)

    def __post_init__(self):
        _positive(self.r2, self.operating_point.fr)
        if not math.isfinite(self.bias_voltage):
            raise ValueError("bias_voltage must be finite")

    @property
    def state_size(self):
        return 5 if self.varactor is not None and self.include_esr else 4

    def simulate(self, input_func, t_span, t_eval, x0=None, solver="ivp",
                 symbol_period=None, pulse_duty=1.0, max_step=np.inf):
        """Integrate in normalized time, returning physical time and states.

        The runner's exact_zoh option uses numerical integration on each
        constant-input interval, including any pulse turn-off boundary.
        """
        from types import SimpleNamespace
        from scipy.integrate import solve_ivp

        if not max_step > 0:
            raise ValueError("max_step must be positive")
        start, end = t_span
        t_eval = np.asarray(t_eval, dtype=float)
        if (not np.isfinite([start, end]).all() or end <= start
                or t_eval.ndim != 1 or not t_eval.size or not np.isfinite(t_eval).all()
                or t_eval[0] < start or t_eval[-1] > end or np.any(np.diff(t_eval) <= 0)):
            raise ValueError("t_eval must increase within a finite, increasing t_span")
        state = np.zeros(self.state_size) if x0 is None else np.asarray(x0, dtype=float)
        if state.shape != (self.state_size,) or not np.isfinite(state).all():
            raise ValueError(f"x0 must contain {self.state_size} finite values")
        edges = np.array([start, end])
        if solver == "exact_zoh":
            if symbol_period is None:
                raise ValueError("exact_zoh requires symbol_period")
            _positive(symbol_period)
            if not 0 < pulse_duty <= 1:
                raise ValueError("pulse_duty must be in (0, 1]")
            symbols = np.arange(math.floor(start / symbol_period),
                                math.ceil(end / symbol_period)) * symbol_period
            boundaries = symbols
            if pulse_duty < 1:
                boundaries = np.r_[symbols, symbols + pulse_duty * symbol_period]
            edges = np.unique(np.r_[start, boundaries[(start < boundaries) & (boundaries < end)], end])
        elif solver != "ivp":
            raise ValueError(f"Unsupported solver: {solver}")

        point = self.operating_point
        d = point.design
        scale = d.omega1
        c1, c2 = d.c1, point.c2
        gqc = self.varactor
        with_esr = self.state_size == 5
        method = "Radau" if with_esr else "DOP853"
        samples = np.empty((self.state_size, len(t_eval)))
        for left, right in zip(edges[:-1], edges[1:]):
            held = input_func(left + (right-left)/2) if solver == "exact_zoh" else None

            def rhs(tau, x):
                v1, v2, i1, i2 = x[:4]
                current = input_func(left + tau/scale) if held is None else held
                a = current - i1 - v1/d.r1
                b = -i2 - v2/self.r2
                branch_voltage = v1-v2
                intrinsic = x[4] if with_esr else branch_voltage
                if gqc is None:
                    capacitance = d.coupling_capacitance
                else:
                    capacitance, resistance = gqc.capacitance_and_resistance(self.bias_voltage + intrinsic)
                if with_esr:
                    branch_current = (branch_voltage - intrinsic)/resistance
                    return np.array([(a-branch_current)/c1, (b+branch_current)/c2,
                                     v1/d.inductance, v2/d.inductance,
                                     branch_current/capacitance])/scale
                determinant = c1*c2 + capacitance*(c1+c2)
                return np.array([((c2+capacitance)*a+capacitance*b)/determinant,
                                 (capacitance*a+(c1+capacitance)*b)/determinant,
                                 v1/d.inductance, v2/d.inductance])/scale

            solution = solve_ivp(rhs, (0, (right-left)*scale), state, method=method,
                                 rtol=1e-8, atol=1e-11, dense_output=True,
                                 max_step=max_step*scale)
            if not solution.success:
                raise RuntimeError(solution.message)
            lo = np.searchsorted(t_eval, left)
            hi = np.searchsorted(t_eval, right, side="right" if right == end else "left")
            if hi > lo:
                samples[:, lo:hi] = solution.sol((t_eval[lo:hi]-left)*scale)
            state = solution.y[:, -1]
        return SimpleNamespace(t=t_eval, y=samples, success=True, message="Integration completed.")
