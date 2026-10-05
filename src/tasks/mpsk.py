import numpy as np
from scipy.integrate import solve_ivp


# ============================================================
# 4. BPSK phases
# ============================================================

def input_bpsk_0(t, t_center, sigma, Iin_peak, fc, phase_0):
    return phase_modulated_pulse(
        t=t,
        center=t_center,
        sigma=sigma,
        I_peak=Iin_peak,
        fc=fc,
        phase=phase_0
    )


def input_bpsk_1(t, t_center, sigma, Iin_peak, fc, phase_1):
    return phase_modulated_pulse(
        t=t,
        center=t_center,
        sigma=sigma,
        I_peak=Iin_peak,
        fc=fc,
        phase=phase_1
    )


# ============================================================
# 3. Phase-modulated Gaussian carrier
# ============================================================

def phase_modulated_pulse(
    t,
    center,
    sigma,
    I_peak,
    fc,
    phase
):
    envelope = np.exp(
        -0.5 * ((t - center) / sigma)**2
    )

    carrier = np.cos(
        2.0 * np.pi * fc * (t - center) + phase
    )

    return I_peak * envelope * carrier


def get_mpsk_phases(M):
    return 2.0 * np.pi * np.arange(M) / M


def simulate_history(reservoir_tank, t_end, t_eval, eta, input_func):
    """
    Simulate the same PT-RLC reservoir with an arbitrary input history.
    """

    x0 = np.zeros(4)

    sol = solve_ivp(
        lambda t, x: reservoir_tank.rhs_history(
            t,
            x,
            eta,
            reservoir_tank.G0,
            input_func
        ),
        (0.0, t_end),
        x0,
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-12
    )

    return sol


def simulate_mpsk(
    M,
    t_center,
    t_eval,
    t_end,
    sigma,
    Iin_peak,
    fc,
    eta,
    alpha=0.0,
    add_noise=False,
    snr_db=20.0,
    seed=42
):

    phases = get_mpsk_phases(M)

    sols = []
    clean_inputs = []
    actual_inputs = []

    rng = np.random.default_rng(seed)

    for phase in phases:

        # ====================================================
        # 1. Generate clean M-PSK waveform
        # ====================================================

        u_clean = phase_modulated_pulse(
            t=t_eval,
            center=t_center,
            sigma=sigma,
            I_peak=Iin_peak,
            fc=fc,
            phase=phase
        )

        # ====================================================
        # 2. Optional AWGN
        # ====================================================

        if add_noise:

            mask_signal = (
                np.abs(t_eval - t_center)
                <= 3.0 * sigma
            )

            P_signal = np.mean(
                u_clean[mask_signal]**2
            )

            P_noise = (
                P_signal /
                (10.0**(snr_db / 10.0))
            )

            noise = rng.normal(
                loc=0.0,
                scale=np.sqrt(P_noise),
                size=len(t_eval)
            )

            u_input = u_clean + noise

        else:

            u_input = u_clean.copy()

        # ====================================================
        # 3. Fixed waveform -> function for solve_ivp
        # ====================================================

        def input_func(t, u=u_input):

            return np.interp(
                t,
                t_eval,
                u,
                left=0.0,
                right=0.0
            )

        # ====================================================
        # 4. Reservoir
        # ====================================================

        if alpha == 0.0:
            sol = simulate_history(
                t_end=t_end,
                t_eval=t_eval,
                eta=eta,
                input_func=input_func
            )

        else:

            sol = simulate_nonlinear(
                eta=eta,
                input_func=input_func,
                alpha=alpha,
                t_span=(t_eval[0], t_eval[-1]),
                t_eval=t_eval
            )

        sols.append(sol)

        # Keep inputs for plotting/debug
        clean_inputs.append(u_clean)
        actual_inputs.append(u_input)

    return (
        phases,
        sols,
        clean_inputs,
        actual_inputs
    )


def simulate_nonlinear(
    reservoir_tank,
    eta,
    input_func,
    alpha,
    t_span,
    t_eval,
    x0=None
):
    if x0 is None:
        x0 = np.zeros(4)

    sol = solve_ivp(
        lambda t, x: reservoir_tank.nonlinear_rhs(
            t,
            x,
            eta,
            alpha,
            input_func
        ),
        t_span,
        x0,
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-11
    )

    return sol

def simulate_reservoir_linear(
    reservoir_tank,
    eta,
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

    A = reservoir_tank.build_state_matrix(eta)
    B = reservoir_tank.build_input_vector()

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
