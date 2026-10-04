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
# add gaussian noise
# ============================================================
def add_awgn(
    signal,
    snr_db,
    rng,
    power_mask=None
):
    """
    Add AWGN to a sampled clean waveform.

    Parameters
    ----------
    signal : ndarray
        Clean waveform samples.

    snr_db : float
        Desired SNR in dB.

    rng : np.random.Generator
        Random-number generator.

    power_mask : ndarray of bool, optional
        Region used to estimate signal power.
        Noise is still added to the whole waveform.

    Returns
    -------
    noisy_signal : ndarray
    noise : ndarray
    signal_power : float
    noise_power_target : float
    """

    signal = np.asarray(signal)

    if power_mask is None:
        signal_for_power = signal
    else:
        signal_for_power = signal[power_mask]

    signal_power = np.mean(signal_for_power**2)

    noise_power_target = (
        signal_power / (10.0**(snr_db / 10.0))
    )

    noise_std = np.sqrt(noise_power_target)

    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=signal.shape
    )

    noisy_signal = signal + noise

    return (
        noisy_signal,
        noise,
        signal_power,
        noise_power_target
    )

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
    t_input = t_eval.copy()


    print(f"dt = {dt*1e9:.4f} ns")
    print(f"Sampling rate = {1/dt/1e9:.3f} GHz")
    print(f"Number of time points = {len(t_eval)}")

    # ============================================================
    # noise configuration
    # ============================================================
    noise_enabled = True
    snr_db = 20.0
    noise_seed = 42


    # ============================================================
    # 5. Generate MPSK waveforms
    # ============================================================
    M = 32
    eta = 1.0
    phases, sols, clean_input, actual_input = simulate_mpsk(
        M=M,
        t_center=t_center,
        t_eval=t_eval,
        t_end=t_end,
        sigma=sigma,
        Iin_peak=Iin_peak,
        fc=fc,
        eta=eta,
        alpha=0.0,
        add_noise=noise_enabled,
        snr_db=snr_db,
        seed=noise_seed
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


    # print AWGN noise details
    # ============================================================
    # AWGN sanity check
    # ============================================================
    if noise_enabled:

        # Check one M-PSK symbol, e.g. phi = 0
        k_check = 0

        u_clean = clean_input[k_check]
        u_noisy = actual_input[k_check]

        # Recover the actual noise realization
        noise = u_noisy - u_clean

        # Use the same signal region used for SNR definition
        power_mask = (
            np.abs(t_eval - t_center)
            <= 3.0 * sigma
        )

        # Measured signal/noise power
        Ps_measured = np.mean(
            u_clean[power_mask]**2
        )

        Pn_measured = np.mean(
            noise[power_mask]**2
        )

        snr_measured_db = 10.0 * np.log10(
            Ps_measured / Pn_measured
        )

        print("\n=== AWGN sanity check ===")

        print(
            f"M = {M}"
        )

        print(
            f"Checked phase = "
            f"{phases[k_check] / np.pi:.3f} pi"
        )

        print(
            f"Requested SNR = "
            f"{snr_db:.2f} dB"
        )

        print(
            f"Measured signal power = "
            f"{Ps_measured:.6e} A^2"
        )

        print(
            f"Measured noise power = "
            f"{Pn_measured:.6e} A^2"
        )

        print(
            f"Measured SNR = "
            f"{snr_measured_db:.2f} dB"
        )

        print(
            f"Noise std = "
            f"{np.std(noise[power_mask]):.6e} A"
        )

        print(
            f"Clean max |u| = "
            f"{np.max(np.abs(u_clean)):.6e} A"
        )

        print(
            f"Noisy max |u| = "
            f"{np.max(np.abs(u_noisy)):.6e} A"
        )



if __name__=="__main__":
    main()