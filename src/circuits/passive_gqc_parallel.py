"""Independent passive dimers with broadcast or round-robin current routing."""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np


@dataclass(frozen=True)
class PassiveGQCParallelReservoir:
    dimers: tuple
    mode: str

    @property
    def n_tanks(self):
        return len(self.dimers)

    @property
    def state_size(self):
        return sum(dimer.state_size for dimer in self.dimers)

    @property
    def voltage_state_indices(self):
        offsets = np.cumsum([0, *(dimer.state_size for dimer in self.dimers)])
        return tuple(offset + i for offset in offsets[:-1] for i in (0, 1))

    def simulate(self, input_func, t_span, t_eval, x0=None, solver="ivp",
                 symbol_period=None, pulse_duty=1.0):
        if self.mode == "interleaved" and (solver != "exact_zoh" or not symbol_period):
            raise ValueError("Interleaved simulation requires an exact_zoh symbol clock")
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
                                    pulse_duty=pulse_duty)
            states.append(result.y)
            offset = end
        return SimpleNamespace(t=result.t, y=np.concatenate(states), success=True,
                               message="Parallel integration completed.")
