"""Independent PT pairs with broadcast or symbol-interleaved input routing."""
from types import SimpleNamespace
import numpy as np
from scipy.linalg import expm
from .pt_single import PTSingleReservoir


class PTParallelReservoir:
    """Each branch is one complete PTSingleReservoir, with four states.

    Switches route current only; they neither reset nor freeze branch states.
    Gains are per-branch current gains, not a fixed-total-power splitter.
    """

    def __init__(self, n_pairs=None, eta=0.5, alpha=8.33,
                 mode="interleaved", input_gains=None, *, n_tanks=None):
        # n_tanks remains an alias for earlier scripts; one branch is a PAIR.
        if n_pairs is not None and n_tanks is not None:
            raise ValueError("Specify n_pairs or legacy n_tanks, not both")
        n_tanks = n_pairs if n_pairs is not None else (4 if n_tanks is None else n_tanks)
        if isinstance(n_tanks, bool) or int(n_tanks) != n_tanks or n_tanks < 1:
            raise ValueError("n_pairs must be a positive integer")
        if mode not in ("broadcast", "interleaved"):
            raise ValueError("mode must be broadcast or interleaved")
        self.n_pairs = int(n_tanks)
        self.n_tanks = self.n_pairs  # backward-compatible alias
        self.mode = mode
        def vector(value, name):
            a = np.asarray(value, dtype=float)
            if a.ndim == 0:
                a = np.full(self.n_tanks, a.item())
            if a.shape != (self.n_tanks,) or not np.all(np.isfinite(a)):
                raise ValueError(f"{name} must be finite scalar or length n_pairs")
            return a
        self.etas = vector(eta, "eta")
        self.alphas = vector(alpha, "alpha")
        if np.any(self.alphas < 0):
            raise ValueError("alpha must be nonnegative")
        self.input_gains = vector(1.0 if input_gains is None else input_gains,
                                  "input_gains")
        self.voltage_state_indices = tuple(i for j in range(self.n_tanks) for i in (4*j, 4*j+1))
        self.branches = [PTSingleReservoir(e, a)
                         for e, a in zip(self.etas, self.alphas)]

    def simulate(self, input_func, t_span, t_eval, x0=None,
                 solver="exact_zoh", symbol_period=None, pulse_duty=1.0):
        """ZOH/pulse inputs only; result.y is branch-major (4*M, time).

        alpha=0 uses matrix exponentials; otherwise the original piecewise
        DOP853 method is reused. Every switching/pulse edge is integrated.
        pulse_duty specifies an input discontinuity; input_func supplies values.
        """
        if solver != "exact_zoh":
            raise ValueError("Parallel routing currently supports exact_zoh only")
        if symbol_period is None or not np.isfinite(symbol_period) or symbol_period <= 0:
            raise ValueError("positive symbol_period required")
        if not 0 < pulse_duty <= 1:
            raise ValueError("pulse_duty must be in (0, 1]")
        start, end = map(float, t_span)
        duration = (end - start) / symbol_period
        if not np.isfinite(duration) or duration <= 0 or abs(duration-round(duration)) > 1e-7:
            raise ValueError("duration must be a positive integer number of symbols")
        t = np.asarray(t_eval, dtype=float)
        if (t.ndim != 1 or not len(t) or not np.all(np.isfinite(t))
                or np.any(np.diff(t) <= 0) or t[0] < start or t[-1] >= end):
            raise ValueError("t_eval must increase within [start, end)")
        initial = np.zeros((self.n_tanks, 4)) if x0 is None else np.asarray(x0, dtype=float)
        if initial.size != 4*self.n_tanks or not np.all(np.isfinite(initial)):
            raise ValueError("x0 must contain 4*n_tanks finite states")
        initial = initial.reshape(self.n_tanks, 4)
        solutions = []
        for j, branch in enumerate(self.branches):
            def routed(time, j=j):
                n = int(np.floor((time-start)/symbol_period))
                enabled = self.mode == "broadcast" or n % self.n_tanks == j
                return self.input_gains[j]*float(input_func(time)) if enabled else 0.0
            if branch.alpha != 0:
                sol = branch.simulate_nonlinear_zoh(
                    routed, (start, end), t, x0=initial[j],
                    symbol_period=symbol_period, pulse_duty=pulse_duty)
            else:
                A = branch.tank.build_state_matrix(branch.eta)
                B = branch.tank.build_input_vector()
                augmented = np.zeros((5, 5))
                augmented[:4, :4] = A
                augmented[:4, 4] = B
                cache = {}
                def advance(state, h, current):
                    if h not in cache:
                        cache[h] = expm(augmented*h)
                    E = cache[h]
                    return E[:4, :4] @ state + E[:4, 4]*current
                states = np.empty((4, len(t)))
                state = initial[j].copy()
                for n in range(round(duration)):
                    a = start+n*symbol_period
                    b = start+(n+1)*symbol_period
                    edge = a+pulse_duty*symbol_period
                    for left, right in ((a, edge), (edge, b)):
                        if right <= left:
                            continue
                        current = routed((left+right)/2)
                        lo, hi = np.searchsorted(t, [left, right])
                        for q in range(lo, hi):
                            states[:, q] = advance(state, float(t[q]-left), current)
                        state = advance(state, right-left, current)
                sol = SimpleNamespace(t=t, y=states, success=True)
            if not np.all(np.isfinite(sol.y)):
                raise RuntimeError(f"nonfinite states in branch {j}")
            solutions.append(sol)
        return SimpleNamespace(t=t, y=np.vstack([s.y for s in solutions]),
                               branches=solutions, t_start=start, success=True,
                               message="Parallel PT simulation completed")

    def features(self, sol, n_symbols, symbol_period, k=8, order="physical"):
        """Concatenate [V1(t1), V2(t1), ...] for every branch.

        order='age' places the currently driven branch first, then older ones.
        Use physical order when branch parameters/gains differ.
        """
        if order not in ("physical", "age"):
            raise ValueError("order must be physical or age")
        if int(k) != k or k < 1 or int(n_symbols) != n_symbols or n_symbols < 1:
            raise ValueError("positive integer k and n_symbols required")
        if symbol_period <= 0:
            raise ValueError("positive symbol_period required")
        start = sol.t_start
        times = start + (np.arange(n_symbols)[:, None]
                        + np.arange(1, k+1)[None, :]/(k+1))*symbol_period
        if times.min() < sol.t[0] or times.max() > sol.t[-1]:
            raise ValueError("sampling grid does not cover feature times; increase points_per_symbol")
        blocks = np.empty((n_symbols, self.n_tanks, 2*k))
        for j in range(self.n_tanks):
            blocks[:, j, 0::2] = np.interp(times, sol.t, sol.y[4*j])
            blocks[:, j, 1::2] = np.interp(times, sol.t, sol.y[4*j+1])
        if order == "age" and self.mode == "interleaved":
            indices = (np.arange(n_symbols)[:, None]-np.arange(self.n_tanks)) % self.n_tanks
            blocks = np.take_along_axis(blocks, indices[:, :, None], axis=1)
        return blocks.reshape(n_symbols, 2*k*self.n_tanks)

