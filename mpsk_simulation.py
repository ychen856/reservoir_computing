import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from reservoir_tank import reservoir_tank
from input_wave import input_wave

reservoir_tank = reservoir_tank()
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


def get_mpsk_phases(M):
    return 2.0 * np.pi * np.arange(M) / M

phases_bpsk = get_mpsk_phases(2)
phases_qpsk = get_mpsk_phases(4)
phases_8psk = get_mpsk_phases(8)
phases_16psk = get_mpsk_phases(16)
phases_32psk = get_mpsk_phases(32)


def simulate_mpsk(reservoir_tank, M, t_center, t_eval, t_end, sigma, Iin_peak, fc, eta=1.0, alpha=0.0):

    phases = get_mpsk_phases(M)
    solutions = []

    for phase in phases:
        def input_func(t, phase=phase):
            return phase_modulated_pulse(t, t_center, sigma, Iin_peak, fc, phase)

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
                t_span=(0.0, t_end),
                t_eval=t_eval

            )

        solutions.append(sol)

    return phases, solutions


# ============================================================
# Step 2A — BPSK through linear PT-RLC reservoir
# ============================================================


def simulate_reservoir_linear(
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

def simulate_history(t_end, t_eval, eta, input_func):
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


def simulate_nonlinear(
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



def main():
    # BPSK input verification: generate two phase-modulated Gaussian pulses and check that they are out of phase by pi radians.
    # ============================================================
    # 1. Carrier / pulse parameters
    # ============================================================

    # Start close to the PT-RLC resonance
    fc = 48.0e6                 # carrier frequency [Hz]
    Tc = 1.0 / fc               # carrier period [s]

    print(f"fc = {fc/1e6:.3f} MHz")
    print(f"Tc = {Tc*1e9:.3f} ns")


    # Pulse amplitude
    Iin_peak = 1.0e-2           # A


    # ============================================================
    # 2. Pulse duration
    # ============================================================

    N_cycles = 8

    pulse_duration = N_cycles * Tc

    # Gaussian pulse:
    # roughly +/- 3 sigma contains almost the whole pulse
    sigma = pulse_duration / 6.0

    # Center of pulse
    t_center = 0.5e-6


    print(f"Carrier cycles = {N_cycles}")
    print(f"Pulse duration = {pulse_duration*1e9:.3f} ns")
    print(f"sigma = {sigma*1e9:.3f} ns")



    # ============================================================
    # 5. Time grid
    # ============================================================

    samples_per_cycle = 40

    dt = Tc / samples_per_cycle

    t_start = 0.0
    t_end = 1.2e-6

    t_eval = np.arange(
        t_start,
        t_end + dt,
        dt
    )


    print(f"dt = {dt*1e9:.4f} ns")
    print(f"Sampling rate = {1/dt/1e9:.3f} GHz")
    print(f"Number of time points = {len(t_eval)}")


    # ============================================================
    # 5. Generate MPSK waveforms
    # ============================================================
    M = 32
    eta = 1.0
    phases, sols = simulate_mpsk(
        reservoir_tank,
        M=M,
        t_center=t_center,
        t_eval=t_eval,
        t_end=t_end,
        sigma=sigma,
        Iin_peak=Iin_peak,
        fc=fc,
        eta=eta,
        alpha=0.0,
    )

    mod_name = {
        2: "BPSK",
        4: "QPSK",
        8: "8-PSK",
        16: "16-PSK",
        32: "32-PSK"
    }.get(M, f"{M}-PSK")


    # ============================================================
    # 7. Plot full MPSK pulses
    # ============================================================

    plt.figure(figsize=(10, 5))

    for phase, sol in zip(phases, sols):

        plt.plot(
            (sol.t - t_center) * 1e6,
            sol.y[0],
            label=fr"$\phi={phase/np.pi:.2g}\pi$"
        )

    plt.axvline(0, linestyle=":", label="Pulse center")

    plt.xlabel(r"Time relative to pulse center ($\mu$s)")
    plt.ylabel(r"$V_1$ (V)")
    plt.title(rf"{mod_name} continuous reservoir response at $V_1$ ($\eta=1$)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(10, 5))

    for phase, sol in zip(phases, sols):

        plt.plot(
            (sol.t - t_center) * 1e6,
            sol.y[1],
            label=fr"$\phi={phase/np.pi:.2g}\pi$"
        )

    plt.axvline(0, linestyle=":", label="Pulse center")

    plt.xlabel(r"Time relative to pulse center ($\mu$s)")
    plt.ylabel(r"$V_2$ (V)")
    plt.title(rf"{mod_name} continuous reservoir response at $V_2$ ($\eta=1$)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


    # plot trajectory in V1-V2 phase space
    plt.figure(figsize=(7, 7))

    for phase, sol in zip(phases, sols):

        plt.plot(
            sol.y[0],
            sol.y[1],
            label=fr"$\phi={phase/np.pi:.2g}\pi$"
        )

    plt.xlabel(r"$V_1$ (V)")
    plt.ylabel(r"$V_2$ (V)")
    plt.title(rf"{mod_name} reservoir trajectories ($\eta=1$)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



    # ============================================================
    # M-PSK voltage-space separation check
    # ============================================================

    mask_post = sols[0].t >= t_center
    t_post_us = (sols[0].t[mask_post] - t_center) * 1e6

    plt.figure(figsize=(10, 5))

    for k in range(M):

        k_next = (k + 1) % M

        # voltage-state vectors [V1, V2]
        R_k = np.vstack([
            sols[k].y[0],
            sols[k].y[1]
        ])

        R_next = np.vstack([
            sols[k_next].y[0],
            sols[k_next].y[1]
        ])

        # same definition as original BPSK D_V
        D_V = np.linalg.norm(
            R_k - R_next,
            axis=0
        )

        D_post = D_V[mask_post]

        phi_k = phases[k] / np.pi
        phi_next = phases[k_next] / np.pi

        plt.plot(
            t_post_us,
            D_post,
            label=rf"$\phi={phi_k:g}\pi$ vs ${phi_next:g}\pi$"
        )

    plt.xlabel(r"Time after pulse center ($\mu$s)")
    plt.ylabel(r"$D_V(t)$ (V)")
    plt.title(
        rf"{mod_name} voltage-state separation "
        rf"($\eta={eta}$)"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()




if __name__=="__main__":
    main()