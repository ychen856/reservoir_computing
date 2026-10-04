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
    # 6. Generate clean BPSK waveforms
    # ============================================================
    phase_0 = 0.0
    phase_1 = np.pi

    u0 = np.array([
        input_bpsk_0(t, t_center, sigma, Iin_peak, fc, phase_0)
        for t in t_eval
    ])

    u1 = np.array([
        input_bpsk_1(t, t_center, sigma, Iin_peak, fc, phase_1)
        for t in t_eval 
    ])

    print("\n=== BPSK input sanity check ===")

    print(f"max |u0| = {np.max(np.abs(u0)):.6e} A")
    print(f"max |u1| = {np.max(np.abs(u1)):.6e} A")

    print(
        "max |u0 + u1| =",
        np.max(np.abs(u0 + u1))
    )



    # ============================================================
    # 7. Plot full BPSK pulses
    # ============================================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        t_eval * 1e6,
        u0,
        label=r"$\phi=0$"
    )

    plt.plot(
        t_eval * 1e6,
        u1,
        label=r"$\phi=\pi$",
        linestyle="--"
    )

    plt.axvline(
        t_center * 1e6,
        linestyle=":",
        label="Pulse center"
    )

    plt.xlabel("Time ($\\mu$s)")
    plt.ylabel("Input current (A)")

    plt.title(
        "Clean BPSK input: phase-modulated Gaussian carrier"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # ============================================================
    # 8. Zoom around pulse center
    # ============================================================

    zoom_half_width = 3.0 * Tc

    mask_zoom = (
        (t_eval >= t_center - zoom_half_width)
        &
        (t_eval <= t_center + zoom_half_width)
    )


    plt.figure(figsize=(10, 5))

    plt.plot(
        (t_eval[mask_zoom] - t_center) * 1e9,
        u0[mask_zoom],
        label=r"$\phi=0$"   
    )

    plt.plot(
        (t_eval[mask_zoom] - t_center) * 1e9,
        u1[mask_zoom],
        label=r"$\phi=\pi$",
        linestyle="--"
    )

    plt.axvline(
        0.0,
        linestyle=":"
    )

    plt.xlabel("Time relative to pulse center (ns)")
    plt.ylabel("Input current (A)")

    plt.title("BPSK carrier phase check")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



    # start BPSK reservoir simulation
    eta_test = 1.0

    input_wave_instance = input_wave(
        Iin_peak=Iin_peak,
        sigma=sigma,
        t_start=t_start,
        t_end=t_end,
        t_center=t_center,
        t_eval=t_eval
    )
    # ------------------------------------------------------------
    # BPSK symbol 0: phi = 0
    # ------------------------------------------------------------
    def input_symbol_0(t):
        return input_bpsk_0(
            t,
            t_center,
            sigma,
            Iin_peak,
            fc,
            phase_0
        )

    # ------------------------------------------------------------
    # BPSK symbol 1: phi = pi
    # ------------------------------------------------------------
    def input_symbol_pi(t):
        return input_bpsk_1(
            t,
            t_center,
            sigma,
            Iin_peak,
            fc,
            phase_1
        )

    

    sol_0 = simulate_reservoir_linear(
        eta=eta_test,
        input_func=input_symbol_0,
        t_span=(t_eval[0], t_eval[-1]),
        t_eval=t_eval
    )

    sol_pi = simulate_reservoir_linear(
        eta=eta_test,
        input_func=input_symbol_pi,
        t_span=(t_eval[0], t_eval[-1]),
        t_eval=t_eval
    )

    # Full internal state is still stored in sol.y
    #
    # x(t) = [V1, V2, IL1, IL2]
    #
    # But observable/readout:
    #
    # r(t) = [V1, V2]

    R_0 = sol_0.y[:2, :]
    R_pi = sol_pi.y[:2, :]

    V1_0 = R_0[0]
    V2_0 = R_0[1]

    V1_pi = R_pi[0]
    V2_pi = R_pi[1]


    # sanity check
    err_V1_sign = np.max(np.abs(V1_0 + V1_pi))
    err_V2_sign = np.max(np.abs(V2_0 + V2_pi))

    scale_V1 = np.max(np.abs(V1_0))
    scale_V2 = np.max(np.abs(V2_0))

    rel_err_V1 = err_V1_sign / max(scale_V1, 1e-30)
    rel_err_V2 = err_V2_sign / max(scale_V2, 1e-30)

    print("\n=== Linear BPSK sign-symmetry check ===")
    print(f"max |V1(phi=0) + V1(phi=pi)| = {err_V1_sign:.6e} V")
    print(f"max |V2(phi=0) + V2(phi=pi)| = {err_V2_sign:.6e} V")

    print(f"relative V1 error = {rel_err_V1:.6e}")
    print(f"relative V2 error = {rel_err_V2:.6e}")


    # plot voltage response at V1 for both BPSK symbols
    t_us = (t_eval - t_center) * 1e6


    plt.figure(figsize=(10, 5))

    plt.plot(
        t_us,
        V1_0,
        label=r"$V_1,\ \phi=0$"
    )

    plt.plot(
        t_us,
        V1_pi,
        "--",
        label=r"$V_1,\ \phi=\pi$"
    )

    plt.axvline(
        0.0,
        linestyle=":",
        label="Pulse center"
    )

    plt.xlabel(r"Time relative to pulse center ($\mu$s)")
    plt.ylabel("Voltage (V)")
    plt.title(
        rf"BPSK reservoir response at $V_1$ ($\eta={eta_test}$)"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # plot voltage response at V2 for both BPSK symbols
    plt.figure(figsize=(10, 5))

    plt.plot(
        t_us,
        V2_0,
        label=r"$V_2,\ \phi=0$"
    )

    plt.plot(
        t_us,
        V2_pi,
        "--",
        label=r"$V_2,\ \phi=\pi$"
    )

    plt.axvline(
        0.0,
        linestyle=":",
        label="Pulse center"
    )

    plt.xlabel(r"Time relative to pulse center ($\mu$s)")
    plt.ylabel("Voltage (V)")
    plt.title(
        rf"BPSK reservoir response at $V_2$ ($\eta={eta_test}$)"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


    # ============================================================
    # voltage-space seperation check
    # ============================================================
    D_V = np.linalg.norm(
        R_0 - R_pi,
        axis=0
    )

    mask_post = t_eval >= t_center

    t_post_us = (t_eval[mask_post] - t_center) * 1e6
    D_post = D_V[mask_post]


    plt.figure(figsize=(10, 5))

    plt.plot(
        t_post_us,
        D_post
    )

    plt.xlabel(r"Time after pulse center ($\mu$s)")
    plt.ylabel(r"$D_V(t)$ (V)")
    plt.title(
        rf"BPSK voltage-state separation ($\eta={eta_test}$)"
    )

    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # plot V2-V1 phase space trajectory for both BPSK symbols
    plt.figure(figsize=(6, 6))

    plt.plot(
        V1_0,
        V2_0,
        label=r"$\phi=0$"
    )

    plt.plot(
        V1_pi,
        V2_pi,
        "--",
        label=r"$\phi=\pi$"
    )

    plt.xlabel(r"$V_1$ (V)")
    plt.ylabel(r"$V_2$ (V)")
    plt.title(
        rf"BPSK reservoir trajectory ($\eta={eta_test}$)"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()





if __name__=="__main__":
    main()