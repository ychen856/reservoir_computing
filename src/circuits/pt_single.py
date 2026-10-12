import numpy as np
from scipy.integrate import solve_ivp

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
        x0=None
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
            rtol=1e-8,
            atol=1e-11
        )

        return sol


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

        if not sol.success:
            raise RuntimeError(sol.message)

        return sol


    
    def simulate(
        self,
        input_func,
        t_span,
        t_eval,
        x0=None
    ):
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
