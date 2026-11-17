"""Passive dimers with broadcast or round-robin current routing."""
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import numpy as np

from src.circuits.passive_gqc import _integrate_passive_gqc, _positive


@dataclass(frozen=True)
class PassiveGQCParallelReservoir:
    dimers: tuple
    mode: str
    coupling_inductance: Optional[float] = None
    coupling_resistance: float = 1.0

    def __post_init__(self):
        if self.coupling_inductance is not None:
            _positive(self.coupling_inductance, self.coupling_resistance)
            if self.n_tanks < 2 or self.n_tanks % 2:
                raise ValueError("Inductive coupling requires a positive even number of dimers")

    @property
    def n_tanks(self):
        return len(self.dimers)

    @property
    def state_size(self):
        return sum(dimer.state_size for dimer in self.dimers) + self.n_links

    @property
    def n_links(self):
        return self.n_tanks // 2 if self.coupling_inductance is not None else 0

    @property
    def voltage_state_indices(self):
        offsets = np.cumsum([0, *(dimer.state_size for dimer in self.dimers)])
        return tuple(offset + i for offset in offsets[:-1] for i in (0, 1))

    def simulate(self, input_func, t_span, t_eval, x0=None, solver="ivp",
                 symbol_period=None, pulse_duty=1.0, *, active_lane=None, max_step=np.inf):
        """Use a fixed input lane for a piecewise-linear interleaved segment."""
        if active_lane is not None:
            if (self.coupling_inductance is None or self.mode != "interleaved"
                    or solver != "ivp" or not 0 <= active_lane < self.n_tanks):
                raise ValueError("A fixed lane requires coupled interleaved IVP integration")
        elif self.mode == "interleaved" and (solver != "exact_zoh" or not symbol_period):
            raise ValueError("Interleaved simulation requires an exact_zoh symbol clock")
        if self.coupling_inductance is not None:
            return self._simulate_coupled(input_func, t_span, t_eval, x0,
                                          solver, symbol_period, pulse_duty, active_lane, max_step)
        initial = np.zeros(self.state_size) if x0 is None else np.asarray(x0)
        if initial.shape != (self.state_size,):
            raise ValueError("Incorrect parallel initial state size")
        states = []
        offset = 0
        for lane, dimer in enumerate(self.dimers):
            def drive(time):
                if self.mode == "interleaved":
                    symbol = int(np.floor(time / symbol_period))
                    if symbol % self.n_tanks != lane:
                        return 0.0
                return input_func(time)

            end = offset + dimer.state_size
            result = dimer.simulate(drive, t_span, t_eval, x0=initial[offset:end],
                                    solver=solver, symbol_period=symbol_period,
                                    pulse_duty=pulse_duty, max_step=max_step)
            states.append(result.y)
            offset = end
        return SimpleNamespace(t=result.t, y=np.concatenate(states), success=True,
                               message="Parallel integration completed.")


    def _simulate_coupled(self, input_func, t_span, t_eval, x0, solver,
                          symbol_period, pulse_duty, active_lane, max_step):
        initial = np.zeros(self.state_size) if x0 is None else np.asarray(x0)
        if initial.shape != (self.state_size,):
            raise ValueError("Incorrect coupled initial state size")
        offsets = np.cumsum([0, *(dimer.state_size for dimer in self.dimers)])
        pair_states, link_states = [], []
        for pair in range(self.n_links):
            lane = 2 * pair
            start, stop = offsets[lane], offsets[lane + 2]
            pair_initial = np.r_[initial[start:stop], initial[offsets[-1] + pair]]
            result = self._simulate_pair(lane, input_func, t_span, t_eval,
                                         pair_initial, solver, symbol_period, pulse_duty,
                                         active_lane, max_step)
            pair_states.append(result.y[:-1])
            link_states.append(result.y[-1:])
        result.y = np.concatenate(pair_states + link_states)
        return result

    def _simulate_pair(self, lane, input_func, t_span, t_eval, x0, solver,
                       symbol_period, pulse_duty, active_lane, max_step):
        first, second = self.dimers[lane:lane + 2]
        offset = first.state_size
        inductance, resistance = self.coupling_inductance, self.coupling_resistance

        def drive(time):
            current = input_func(time)
            if self.mode == "interleaved":
                active = active_lane
                if active is None:
                    active = int(np.floor(time / symbol_period)) % self.n_tanks
                return (current if active == lane else 0.0,
                        current if active == lane + 1 else 0.0)
            return current, current

        def derivatives(state, currents):
            link_current = state[-1]  # Positive from V1 to V3.
            return np.r_[
                first._derivatives(state[:offset], currents[0] - link_current),
                second._derivatives(state[offset:-1], currents[1] + link_current),
                (state[0] - state[offset] - resistance * link_current) / inductance,
            ]

        return _integrate_passive_gqc(
            derivatives, drive, t_span, t_eval, x0, first.state_size + second.state_size + 1,
            first.operating_point.design.omega1, "Radau",
            solver, symbol_period, pulse_duty, max_step,
        )
