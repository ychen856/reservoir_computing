import numpy as np

class reservoir_tank:
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

    def rhs_history(self, t, x, eta, G0, input_func):
        """
        Driven PT-RLC dynamics.
        x = [V1, V2, IL1, IL2]
        """

        # Same A matrix as Stage 1B
        A = self.build_state_matrix(eta)

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

