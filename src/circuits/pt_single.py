import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from types import SimpleNamespace

class pt_reservoir_tank:
    def __init__(self):
        # ============================================================
        # 1. Physical parameters
        # ============================================================
        self.L  = 1e-6       # 1 uH
        self.C  = 10e-12     # 10 pF
        self.Cc = 1e-12      # 1 pF


        # ============================================================
        # 2. Exact ideal-PT EP
        # ============================================================
        self.G_EP_sq = (2.0 / self.L) * (
            (self.C + self.Cc) - np.sqrt(self.C * (self.C + 2 * self.Cc))
        )

        self.G_EP = np.sqrt(self.G_EP_sq)
        self.R_EP = 1.0 / self.G_EP

        self.omega_EP = 1.0 / np.sqrt(
            self.L * np.sqrt(self.C * (self.C + 2 * self.Cc))
        )

        self.f_EP = self.omega_EP / (2 * np.pi)

        # ============================================================
        # 3. Common damping
        #
        # delta = G0 / G_EP
        # ============================================================

        self.delta = 0.20

        self.G0 = self.delta * self.G_EP
        self.R0 = 1.0 / self.G0


        # ============================================================
        # 4. Capacitance matrix
        # ============================================================

        self.Cmat = np.array([
            [self.C + self.Cc, -self.Cc],
            [-self.Cc, self.C + self.Cc]
        ])

        self.Cmat_inv = np.linalg.inv(self.Cmat)


    def build_state_matrix(self, eta):

        # Balanced gain/loss conductance
        G = eta * self.G_EP

        # Voltage dynamics:
        #
        # Cmat [dV1/dt, dV2/dt]^T
        #
        # = [
        #    (G - G0)V1 - IL1
        #   -(G + G0)V2 - IL2
        #   ]

        Mv = np.array([
            [ G - self.G0,        0.0, -1.0,  0.0],
            [      0.0, -(G + self.G0),  0.0, -1.0]
        ])

        self.A_voltage = self.Cmat_inv @ Mv

        # Inductor dynamics
        self.A_current = np.array([
            [1.0 / self.L, 0.0, 0.0, 0.0],
            [0.0, 1.0 / self.L, 0.0, 0.0]
        ])

        A = np.vstack([
            self.A_voltage,
            self.A_current
        ])

        return A

    def build_input_vector(self):
        b_voltage = self.Cmat_inv @ np.array([1.0, 0.0])

        B = np.array([
            b_voltage[0],
            b_voltage[1],
            0.0,
            0.0
        ])

        return B

    def gain_saturation(self, V1, eta, alpha):
        """
        Nonlinear saturated gain.

        alpha = 0 -> linear baseline
        """
        G = eta * self.G_EP

        G_eff = G / (1.0 + alpha * V1**2)

        return G_eff

    def rhs_history(self, t, x, input_func):
        """
        Driven PT-RLC dynamics.
        x = [V1, V2, IL1, IL2]
        """

        # Same A matrix as Stage 1B
        A = self.build_state_matrix()

        # Natural dynamics
        dx = A @ x

        # External current injected into node 1
        i_in = input_func(t)

        # Cmat * dV/dt receives [i_in, 0]^T
        input_voltage = self.Cmat_inv @ np.array([
            i_in,
            0.0
        ])

        dx[0] += input_voltage[0]
        dx[1] += input_voltage[1]

        return dx

    def nonlinear_rhs(self, t, x, eta, alpha, input_func):

        V1, V2, IL1, IL2 = x

        # Current input
        u = input_func(t)

        # Nonlinear gain
        G_eff = self.gain_saturation(V1, eta, alpha)

        # Voltage equations
        rhs_voltage = np.array([
            (G_eff - self.G0) * V1 - IL1 + u,
            -(eta * self.G_EP + self.G0) * V2 - IL2
        ])

        dV = self.Cmat_inv @ rhs_voltage

        # Inductor equations
        dIL1 = V1 / self.L
        dIL2 = V2 / self.L

        return np.array([
            dV[0],
            dV[1],
            dIL1,
            dIL2
        ])

class PTSingleReservoir:
    def __init__(self, eta=1.0, alpha=0.0):
        self.tank = pt_reservoir_tank()
        self.eta = eta
        self.alpha = alpha

    def simulate_nonlinear(
        self,
        input_func,
        t_span,
        t_eval,
        x0=None,
    ):
        if x0 is None:
            x0 = np.zeros(4)

        sol = solve_ivp(
            lambda t, x: self.tank.nonlinear_rhs(
                t,
                x,
                self.eta,
                self.alpha,
                input_func
            ),
            t_span,
            x0,
            t_eval=t_eval,
            method="DOP853",
            rtol=1e-8,
            atol=1e-11
        )

        if not sol.success:
            raise RuntimeError(
                f"Nonlinear simulation failed: {sol.message}"
            )

        return sol

    def simulate_nonlinear_zoh(
        self,
        input_func,
        t_span,
        t_eval,
        x0=None,
        symbol_period=None,
    ):
        """
        Piecewise DOP853 integration for nonlinear PT
        with zero-order-hold input.
        """

        if symbol_period is None or symbol_period <= 0:
            raise ValueError("Valid symbol_period is required")



        if x0 is None:
            x0 = np.zeros(4)

        t_eval = np.asarray(t_eval, dtype=float)

        t_start, t_end = t_span

        n_symbols = int(round(
            (t_end - t_start) / symbol_period
        ))

        if not np.isclose(
            t_start + n_symbols * symbol_period,
            t_end,
            rtol=1e-9,
            atol=1e-15
        ):
            raise ValueError(
                "Simulation duration must be a multiple of symbol_period"
            )

        if t_eval.ndim != 1:
            raise ValueError("t_eval must be one-dimensional")
        
        if len(t_eval) == 0:
            raise ValueError("t_eval cannot be empty")
        
        if np.any(np.diff(t_eval) <= 0):
            raise ValueError("t_eval must be strictly increasing")
        
        if t_eval[0] < t_start or t_eval[-1] >= t_end:
            raise ValueError(
                "t_eval must lie within [t_start, t_end)"
            )

        states = np.empty((4, len(t_eval)), dtype=float)
        x = np.asarray(x0, dtype=float).copy()

        for n in range(n_symbols):

            a = t_start + n * symbol_period
            b = t_start + (n + 1) * symbol_period

            # Hold input constant within this symbol.
            u_n = float(input_func(0.5 * (a + b)))

            def rhs(t, state):
                return self.tank.nonlinear_rhs(
                    t,
                    state,
                    self.eta,
                    self.alpha,
                    lambda _: u_n
                )

            # Select output samples belonging to this symbol.
            i0 = np.searchsorted(t_eval, a, side="left")
            i1 = np.searchsorted(t_eval, b, side="left")

            local_times = t_eval[i0:i1]

            # Dense output avoids requiring t_eval to include
            # the right boundary of every symbol.
            sol = solve_ivp(
                rhs,
                (a, b),
                x,
                method="DOP853",
                dense_output=True,
                rtol=1e-8,
                atol=1e-11,
            )

            if not sol.success:
                raise RuntimeError(
                    f"ZOH integration failed at symbol {n}: "
                    f"{sol.message}"
                )

            if len(local_times):
                states[:, i0:i1] = sol.sol(local_times)

            # Pass the final state to the next symbol.
            x = sol.y[:, -1].copy()

        return SimpleNamespace(
            t=t_eval,
            y=states,
            success=True,
            message="Piecewise nonlinear ZOH completed",
        )

    '''def simulate_history(self, t_end, t_eval, input_func):
        """
        Simulate the same PT-RLC reservoir with an arbitrary input history.
        """

        x0 = np.zeros(4)

        sol = solve_ivp(
            lambda t, x: self.tank.rhs_history(
                t,
                x,
                self.eta,
                input_func
            ),
            (0.0, t_end),
            x0,
            t_eval=t_eval,
            rtol=1e-9,
            atol=1e-12
        )

        return sol'''


    def simulate_linear(
        self,
        input_func,
        t_span,
        t_eval,
        x0=None
    ):
        """
        State:
            x = [V1, V2, IL1, IL2]

        Readout later uses only:
            r = [V1, V2]
        """

        if x0 is None:
            x0 = np.zeros(4)

        A = self.tank.build_state_matrix(self.eta)
        B = self.tank.build_input_vector()

        def rhs(t, x):
            u = input_func(t)
            return A @ x + B * u

        sol = solve_ivp(
            rhs,
            t_span,
            x0,
            t_eval=t_eval,
            method="DOP853",
            rtol=1e-9,
            atol=1e-12
        )

        '''if not sol.success:
            raise RuntimeError(sol.message)'''
        if not sol.success:

            t_fail = sol.t[-1]

            fc = 48e6
            tc = 1.0 / fc
            ts = tc * 0.5

            symbol_position = t_fail / ts
            nearest_boundary = round(symbol_position)

            boundary_time = nearest_boundary * ts

            print("\n[Linear solver failure]")

            print("Message:", sol.message)

            print("Last returned time:", t_fail)

            print("Symbol position:", symbol_position)

            print("Nearest symbol index:", nearest_boundary)

            print(
                "Distance to nearest symbol boundary:",
                abs(t_fail - boundary_time),
            )

            print("Last state:", sol.y[:, -1])

            print(
                "Max absolute state:",
                np.max(np.abs(sol.y)),
            )

            raise RuntimeError(sol.message)

        return sol


    def simulate_linear_zoh(
        self,
        input_func,
        t_span,
        t_eval,
        x0=None,
        symbol_period=None,
    ):
        """
        Exact ZOH integration for the linear PT reservoir.

        dx/dt = A x + B u(t)

        Assumptions:
            1. alpha == 0
            2. Input is constant within each symbol period
            3. t_eval is uniformly spaced
            4. t_span starts at zero
        """

        if self.alpha != 0.0:
            raise ValueError(
                "Exact ZOH integration requires alpha=0"
            )

        if symbol_period is None:
            raise ValueError(
                "symbol_period is required for exact ZOH"
            )

        t_eval = np.asarray(t_eval, dtype=float)

        if len(t_eval) < 2:
            raise ValueError(
                "t_eval must contain at least two points"
            )

        if not np.isclose(t_span[0], 0.0, atol=1e-15):
            raise ValueError(
                "Exact ZOH currently requires t_span[0]=0"
            )

        if x0 is None:
            x0 = np.zeros(4)

        A = self.tank.build_state_matrix(self.eta)
        B = self.tank.build_input_vector()

        n_states = A.shape[0]

        B = np.asarray(B).reshape(n_states, 1)

        dt = t_eval[1] - t_eval[0]

        if not np.allclose(
            np.diff(t_eval),
            dt,
            rtol=1e-8,
            atol=1e-15,
        ):
            raise ValueError(
                "Exact ZOH requires uniformly spaced t_eval"
            )

        points_per_symbol = int(
            round(symbol_period / dt)
        )

        if not np.isclose(
            points_per_symbol * dt,
            symbol_period,
            rtol=1e-8,
            atol=1e-15,
        ):
            raise ValueError(
                "symbol_period must be an integer multiple of dt"
            )

        # --------------------------------------------------
        # Exact state transition
        # --------------------------------------------------

        M = np.zeros((n_states + 1, n_states + 1))

        M[:n_states, :n_states] = A
        M[:n_states, n_states:] = B

        E = expm(M * dt)

        Phi = E[:n_states, :n_states]
        Gamma = E[:n_states, n_states]

        # --------------------------------------------------
        # Simulation
        # --------------------------------------------------

        states = np.zeros(
            (n_states, len(t_eval)),
            dtype=float,
        )

        x = np.asarray(x0, dtype=float).copy()

        for i, t in enumerate(t_eval):

            states[:, i] = x

            # Evaluate at the center of the current step.
            # This avoids ambiguity exactly at a ZOH boundary.
            u = input_func(t + 0.5 * dt)

            x = Phi @ x + Gamma * u

        return SimpleNamespace(
            t=t_eval,
            y=states,
            success=True,
            message="Exact ZOH integration completed",
        )


    
    def simulate(
        self,
        input_func,
        t_span,
        t_eval,
        x0=None,
        solver='ivp',
        symbol_period=None,
    ):
        print(
            f"[PTSingleReservoir] "
            f"eta={self.eta}, "
            f"alpha={self.alpha}, "
            f"solver={solver}"
        )
        if solver == 'exact_zoh':
            if self.alpha == 0.0:
                return self.simulate_linear_zoh(
                    input_func=input_func,
                    t_span=t_span,
                    t_eval=t_eval,
                    x0=x0,
                    symbol_period=symbol_period,
                )
            return self.simulate_nonlinear_zoh(
                input_func=input_func,
                t_span=t_span,
                t_eval=t_eval,
                x0=x0,
                symbol_period=symbol_period,
            )
            
        elif solver == 'ivp':
            if self.alpha == 0.0:
                return self.simulate_linear(
                    input_func=input_func,
                    t_span=t_span,
                    t_eval=t_eval,
                    x0=x0
                )

            return self.simulate_nonlinear(
                input_func=input_func,
                t_span=t_span,
                t_eval=t_eval,
                x0=x0
            )
        else:
            raise ValueError(f"Unknown solver: {solver}")
