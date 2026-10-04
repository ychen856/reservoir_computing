from linear_readout import predict_linear_readout, train_linear_readout
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from reservoir_tank import reservoir_tank
from input_wave import input_wave
from utils import build_mpsk_feature_matrix, build_repeated_noise_dataset, centroid_distance_matrix, compute_class_statistics, pairwise_feature_distances, project_reservoir_features_pca

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

def generate_impaired_qpsk_pulse(
    t,
    center,
    sigma,
    I_peak,
    fc,
    phase,
    amp_scale=1.0,
    time_delay=0.0,
    phase_offset=0.0,
    freq_offset=0.0,
):
    t_shift = t - time_delay

    envelope = np.exp(
        -0.5 * ((t_shift - center) / sigma)**2
    )

    carrier = np.cos(
        2.0 * np.pi * (fc + freq_offset) * t_shift
        + phase
        + phase_offset
    )

    return amp_scale * I_peak * envelope * carrier

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
# add impairment
# ============================================================
def apply_single_symbol_impairments(
    t,
    envelope,
    phi,
    Iin_peak,
    fc,
    amp_scale=1.0,
    time_delay=0.0,
    phase_offset=0.0,
    freq_offset=0.0,
):
    # timing offset
    t_shift = t - time_delay

    # If envelope is Gaussian:
    # envelope_shift = exp(-(t_shift - t_center)^2 / (2*sigma^2))
    #
    # Better: pass t_center and sigma here and reconstruct it.
    envelope_shift = envelope   # temporary if delay only applied to carrier

    phase = (
        2.0 * np.pi * (fc + freq_offset) * t_shift
        + phi
        + phase_offset
    )

    return (
        amp_scale
        * Iin_peak
        * envelope_shift
        * np.cos(phase)
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

 
def simulate_mpsk_repeated_noise(
    M,
    t_center,
    t_eval,
    t_end,
    sigma,
    Iin_peak,
    fc,
    eta,
    n_repeat,
    alpha=0.0,
    add_noise=False,
    snr_db=20.0,
    seed=42,
    adc_enabled=False,
    fs_adc=None,
    adc_bits=None
):
    """
    Simulate repeated M-PSK inputs through the reservoir.

    For each M-PSK phase:
        1. Generate a clean phase-modulated waveform.
        2. Generate n_repeat independent AWGN realizations
           if add_noise=True.
        3. Pass each waveform through the reservoir.
        4. Store the full continuous reservoir solution.

    Parameters
    Returns
    -------
    phases : ndarray, shape (M,)
        M-PSK phases.

    sols_by_class : list[list]
        sols_by_class[m][r] is the reservoir solution
        for phase m and realization r.
    """

    # for adc
    rng = np.random.default_rng(seed)

    sols_by_class = []

    # ========================================================
    # ADC sampling grid
    # ========================================================

    if adc_enabled:

        if fs_adc is None:
            raise ValueError(
                "fs_adc must be specified when adc_enabled=True"
            )

        dt_adc = 1.0 / fs_adc

        t_adc = np.arange(
            t_eval[0],
            t_eval[-1] + dt_adc,
            dt_adc
        )

    else:

        dt_adc = None
        t_adc = None

    

    # ========================================================
    # M-PSK constellation phases
    # ========================================================

    phases = get_mpsk_phases(M)

    # One RNG for the whole experiment:
    # reproducible, but every realization gets different noise.
    rng = np.random.default_rng(seed)

    sols_by_class = []

    # Region used to define signal power / SNR
    power_mask = (
        np.abs(t_eval - t_center)
        <= 3.0 * sigma
    )

    # ========================================================
    # Loop over M-PSK phases
    # ========================================================

    for m, phase in enumerate(phases):

        # ----------------------------------------------------
        # Clean input for this phase
        # ----------------------------------------------------

        u_clean = phase_modulated_pulse(
            t=t_eval,
            center=t_center,
            sigma=sigma,
            I_peak=Iin_peak,
            fc=fc,
            phase=phase
        )

        # ----------------------------------------------------
        # Determine AWGN power
        # ----------------------------------------------------

        if add_noise:

            P_signal = np.mean(
                u_clean[power_mask] ** 2
            )

            P_noise = (
                P_signal /
                (10.0 ** (snr_db / 10.0))
            )

            noise_std = np.sqrt(P_noise)

        class_sols = []

        # ====================================================
        # Independent realizations for this phase
        # ====================================================

        for r in range(n_repeat):

            # ------------------------------------------------
            # Build actual reservoir input
            # ------------------------------------------------

            if add_noise:

                noise = rng.normal(
                    loc=0.0,
                    scale=noise_std,
                    size=t_eval.shape
                )

                u_input = u_clean + noise

            else:

                u_input = u_clean.copy()

            # ------------------------------------------------
            # Fixed waveform -> callable input function
            # ------------------------------------------------
            def input_func(t, u=u_input):

                return np.interp(
                    t,
                    t_eval,
                    u,
                    left=0.0,
                    right=0.0
                )

            if adc_enabled:
                # --------------------------------------------
                # ADC sampling
                # --------------------------------------------

                u_adc = input_func(t_adc)

                # --------------------------------------------
                # Optional ADC quantization
                # --------------------------------------------

                if adc_bits is not None:

                    adc_full_scale = 1.2 * Iin_peak

                    levels = 2 ** adc_bits

                    u_adc_clip = np.clip(
                        u_adc,
                        -adc_full_scale,
                        adc_full_scale
                    )

                    delta = (
                        2.0 * adc_full_scale
                        / (levels - 1)
                    )

                    u_adc_used = (
                        np.round(
                            (u_adc_clip + adc_full_scale)
                            / delta
                        )
                        * delta
                        - adc_full_scale
                    )

                else:
                    u_adc_used = u_adc


                # --------------------------------------------
                # Zero-order hold reconstruction
                # --------------------------------------------

                def input_func(
                    t,
                    t_adc=t_adc,
                    u_adc_used=u_adc_used,
                    dt_adc=dt_adc
                ):

                    if t < t_adc[0] or t > t_adc[-1]:
                        return 0.0

                    idx = int(
                        np.floor(
                            (t - t_adc[0]) / dt_adc
                        )
                    )

                    idx = np.clip(
                        idx,
                        0,
                        len(u_adc_used) - 1
                    )

                    

                    return u_adc_used[idx]

                # ------------------------------------------------------------
                # ADC sanity-check plot
                # Only plot the first phase / first noise realization
                # ------------------------------------------------------------
                                    
                if (
                    adc_enabled
                    and m == 0
                    and r == 0
                ):
                    print("\n=== ADC sanity check ===")
                    print(f"Carrier frequency = {fc/1e6:.3f} MHz")
                    print(f"ADC sampling rate = {fs_adc/1e6:.3f} MS/s")
                    print(f"ADC samples/cycle = {fs_adc/fc:.3f}")
                    print(f"ADC sample interval = {dt_adc*1e9:.3f} ns")
                
                    if adc_bits is None:
                        print("ADC quantization = OFF")
                    else:
                        print(f"ADC resolution = {adc_bits} bits")
                    '''plot_adc_sanity_check(
                        t_eval=t_eval,
                        u_input=u_input,
                        t_adc=t_adc,
                        u_adc_used=u_adc_used,
                        input_func=input_func,
                        t_center=t_center,
                        fc=fc,
                        adc_bits=adc_bits
                    )'''



            # ------------------------------------------------
            # Reservoir simulation
            # ------------------------------------------------

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

            class_sols.append(sol)

        sols_by_class.append(class_sols)

        print(
            f"M={M}: completed phase "
            f"{m + 1}/{M}, "
            f"phi={phase / np.pi:.3f} pi"
        )

    return phases, sols_by_class


def simulate_mpsk_repeated_noise_offset(
    M,
    t_center,
    t_eval,
    t_end,
    sigma,
    Iin_peak,
    fc,
    eta,
    n_repeat,
    alpha=0.0,
    add_noise=False,
    snr_db=20.0,
    seed=42,
    adc_enabled=False,
    fs_adc=None,
    adc_bits=None,

    amp_scale=1.0,
    time_delay=0.0,
    phase_offset=0.0,
    freq_offset=0.0,
):
    """
    Simulate repeated M-PSK inputs through the reservoir.

    For each M-PSK phase:
        1. Generate a clean phase-modulated waveform.
        2. Generate n_repeat independent AWGN realizations
           if add_noise=True.
        3. Pass each waveform through the reservoir.
        4. Store the full continuous reservoir solution.

    Parameters
    Returns
    -------
    phases : ndarray, shape (M,)
        M-PSK phases.

    sols_by_class : list[list]
        sols_by_class[m][r] is the reservoir solution
        for phase m and realization r.
    """

    # for adc
    rng = np.random.default_rng(seed)

    sols_by_class = []

    # ========================================================
    # ADC sampling grid
    # ========================================================

    if adc_enabled:

        if fs_adc is None:
            raise ValueError(
                "fs_adc must be specified when adc_enabled=True"
            )

        dt_adc = 1.0 / fs_adc

        t_adc = np.arange(
            t_eval[0],
            t_eval[-1] + dt_adc,
            dt_adc
        )

    else:

        dt_adc = None
        t_adc = None

    

    # ========================================================
    # M-PSK constellation phases
    # ========================================================

    phases = get_mpsk_phases(M)

    # One RNG for the whole experiment:
    # reproducible, but every realization gets different noise.
    rng = np.random.default_rng(seed)

    sols_by_class = []

    # Region used to define signal power / SNR
    power_mask = (
        np.abs(t_eval - (t_center + time_delay))
        <= 3.0 * sigma
    )

    # ========================================================
    # Loop over M-PSK phases
    # ========================================================

    for m, phase in enumerate(phases):

        # ----------------------------------------------------
        # Clean input for this phase
        # ----------------------------------------------------

        '''u_clean = phase_modulated_pulse(
            t=t_eval,
            center=t_center,
            sigma=sigma,
            I_peak=Iin_peak,
            fc=fc,
            phase=phase
        )'''
        u_clean = generate_impaired_qpsk_pulse(
            t=t_eval,
            center=t_center,
            sigma=sigma,
            I_peak=Iin_peak,
            fc=fc,
            phase=phase,
            amp_scale=amp_scale,
            time_delay=time_delay,
            phase_offset=phase_offset,
            freq_offset=freq_offset,
        )

        # ----------------------------------------------------
        # Determine AWGN power
        # ----------------------------------------------------

        if add_noise:

            P_signal = np.mean(
                u_clean[power_mask] ** 2
            )

            P_noise = (
                P_signal /
                (10.0 ** (snr_db / 10.0))
            )

            noise_std = np.sqrt(P_noise)

        class_sols = []

        # ====================================================
        # Independent realizations for this phase
        # ====================================================

        for r in range(n_repeat):

            # ------------------------------------------------
            # Build actual reservoir input
            # ------------------------------------------------

            if add_noise:

                noise = rng.normal(
                    loc=0.0,
                    scale=noise_std,
                    size=t_eval.shape
                )

                u_input = u_clean + noise

            else:

                u_input = u_clean.copy()

            # ------------------------------------------------
            # Fixed waveform -> callable input function
            # ------------------------------------------------
            def input_func(t, u=u_input):

                return np.interp(
                    t,
                    t_eval,
                    u,
                    left=0.0,
                    right=0.0
                )

            if adc_enabled:
                # --------------------------------------------
                # ADC sampling
                # --------------------------------------------

                u_adc = input_func(t_adc)

                # --------------------------------------------
                # Optional ADC quantization
                # --------------------------------------------

                if adc_bits is not None:

                    adc_full_scale = 1.2 * Iin_peak

                    levels = 2 ** adc_bits

                    u_adc_clip = np.clip(
                        u_adc,
                        -adc_full_scale,
                        adc_full_scale
                    )

                    delta = (
                        2.0 * adc_full_scale
                        / (levels - 1)
                    )

                    u_adc_used = (
                        np.round(
                            (u_adc_clip + adc_full_scale)
                            / delta
                        )
                        * delta
                        - adc_full_scale
                    )

                else:
                    u_adc_used = u_adc


                # --------------------------------------------
                # Zero-order hold reconstruction
                # --------------------------------------------

                def input_func(
                    t,
                    t_adc=t_adc,
                    u_adc_used=u_adc_used,
                    dt_adc=dt_adc
                ):

                    if t < t_adc[0] or t > t_adc[-1]:
                        return 0.0

                    idx = int(
                        np.floor(
                            (t - t_adc[0]) / dt_adc
                        )
                    )

                    idx = np.clip(
                        idx,
                        0,
                        len(u_adc_used) - 1
                    )

                    

                    return u_adc_used[idx]

                # ------------------------------------------------------------
                # ADC sanity-check plot
                # Only plot the first phase / first noise realization
                # ------------------------------------------------------------
                                    
                if (
                    adc_enabled
                    and m == 0
                    and r == 0
                ):
                    print("\n=== ADC sanity check ===")
                    print(f"Carrier frequency = {fc/1e6:.3f} MHz")
                    print(f"ADC sampling rate = {fs_adc/1e6:.3f} MS/s")
                    print(f"ADC samples/cycle = {fs_adc/fc:.3f}")
                    print(f"ADC sample interval = {dt_adc*1e9:.3f} ns")
                
                    if adc_bits is None:
                        print("ADC quantization = OFF")
                    else:
                        print(f"ADC resolution = {adc_bits} bits")
                    '''plot_adc_sanity_check(
                        t_eval=t_eval,
                        u_input=u_input,
                        t_adc=t_adc,
                        u_adc_used=u_adc_used,
                        input_func=input_func,
                        t_center=t_center,
                        fc=fc,
                        adc_bits=adc_bits
                    )'''



            # ------------------------------------------------
            # Reservoir simulation
            # ------------------------------------------------

            if alpha == 0.0:
                # temporary phase-offset sanity check
                if m == 0 and r == 0:
                    i_check = np.array([
                        input_func(ti) for ti in t_eval
                    ])

                    plt.figure(figsize=(10, 4))
                    plt.plot(
                        (t_eval - t_center) * 1e9,
                        i_check
                    )

                    plt.xlim(-100, 100)
                    plt.xlabel("Time relative to pulse center (ns)")
                    plt.ylabel("Input current (A)")
                    plt.title(
                        rf"Input waveform: "
                        rf"$\Delta\phi={np.rad2deg(phase_offset):.1f}^\circ$"
                    )
                    plt.grid(True)
                    plt.tight_layout()
                    plt.show()


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

            class_sols.append(sol)

        sols_by_class.append(class_sols)

        print(
            f"M={M}: completed phase "
            f"{m + 1}/{M}, "
            f"phi={phase / np.pi:.3f} pi"
        )

    return phases, sols_by_class

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

def plot_pca_clusters(
    X,
    y,
    phases,
    M,
    K,
    snr_db
):

    X_pca, pca, explained = (
        project_reservoir_features_pca(X)
    )

    plt.figure(figsize=(7, 6))

    for m in range(M):

        mask = (y == m)

        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            s=45,
            alpha=0.75,
            label=rf"$\phi={phases[m]/np.pi:.2f}\pi$"
        )
        centroid_2d = np.mean(
            X,
            axis=0
        )
        
        plt.scatter(
            centroid_2d[0],
            centroid_2d[1],
            marker="x",
            s=100,
            linewidths=2
        )

    plt.xlabel(
        f"PC1 ({explained[0]*100:.1f}% variance)"
    )

    plt.ylabel(
        f"PC2 ({explained[1]*100:.1f}% variance)"
    )

    plt.title(
        rf"{M}-PSK Reservoir Feature Clustering"
        "\n"
        rf"$K={K}$, SNR={snr_db:.0f} dB"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n=== PCA summary ===")
    print(
        f"PC1 explained variance = "
        f"{explained[0]*100:.3f}%"
    )
    print(
        f"PC2 explained variance = "
        f"{explained[1]*100:.3f}%"
    )
    print(
        f"Total 2D explained variance = "
        f"{np.sum(explained)*100:.3f}%"
    )


def adc_zoh_input(t, t_start, dt_adc, u_adc_quant):
    idx = np.floor(
        (t - t_start) / dt_adc
    ).astype(int)

    idx = np.clip(
        idx,
        0,
        len(u_adc_quant) - 1
    )

    return u_adc_quant[idx]    
    

def plot_adc_sanity_check(
    t_eval,
    u_input,
    t_adc,
    u_adc_used,
    input_func,
    t_center,
    fc,
    adc_bits=None
):
    """
    Compare:
      1. numerical analog input waveform
      2. ADC sample values
      3. ZOH waveform actually seen by the reservoir
    """

    # ------------------------------------------------------------
    # Plot only a short window around the pulse center
    # ------------------------------------------------------------
    Tc = 1.0 / fc

    # +/- 4 carrier cycles around pulse center
    t_left = t_center - 4.0 * Tc
    t_right = t_center + 4.0 * Tc

    mask = (
        (t_eval >= t_left)
        & (t_eval <= t_right)
    )

    t_plot = t_eval[mask]

    # Original numerical/analog waveform
    u_analog_plot = u_input[mask]

    # Evaluate the actual ZOH input seen by reservoir
    u_zoh_plot = np.array([
        input_func(t)
        for t in t_plot
    ])

    # ADC samples within plotting window
    mask_adc = (
        (t_adc >= t_left)
        & (t_adc <= t_right)
    )

    t_adc_plot = t_adc[mask_adc]
    u_adc_plot = u_adc_used[mask_adc]


    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    plt.figure(figsize=(11, 5))

    plt.plot(
        (t_plot - t_center) * 1e9,
        u_analog_plot * 1e3,
        linewidth=1.8,
        label="Analog input"
    )

    plt.step(
        (t_plot - t_center) * 1e9,
        u_zoh_plot * 1e3,
        where="post",
        linewidth=1.5,
        label="ADC + ZOH input"
    )

    plt.scatter(
        (t_adc_plot - t_center) * 1e9,
        u_adc_plot * 1e3,
        s=35,
        zorder=5,
        label="ADC samples"
    )

    plt.axvline(
        0.0,
        linestyle="--",
        linewidth=1.0
    )

    plt.xlabel("Time relative to pulse center (ns)")
    plt.ylabel("Input current (mA)")

    if adc_bits is None:
        adc_text = "sampling + ZOH"
    else:
        adc_text = f"{adc_bits}-bit ADC + ZOH"

    plt.title(
        f"Input ADC sanity check: {adc_text}"
    )

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


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
    # ADC configuration
    # ============================================================

    adc_enabled = True

    #fs_adc = 480e6          # ADC sampling rate [Hz]
    fs_adc = 48.0e6*4
    adc_bits = None        # None = sampling + ZOH only
                        # later: 8, 10, 12, ...

    print("\n=== Input ADC configuration ===")
    print(f"ADC enabled = {adc_enabled}")
    print(f"ADC sampling rate = {fs_adc/1e6:.1f} MS/s")
    print(f"ADC samples/cycle = {fs_adc/fc:.2f}")

    if adc_bits is None:
        print("ADC quantization = disabled")
    else:
        print(f"ADC resolution = {adc_bits} bits")


    # ============================================================
    # Temporal sampling configuration
    # ============================================================

    T_obs = 0.7e-6

    t_sample_start = t_center
    t_sample_end = t_center + T_obs

    #K_list = [1, 2, 4, 8, 16, 32]
    K=4

    # ============================================================
    # noise configuration
    # ============================================================
    noise_enabled = True
    snr_db = 20.0
    noise_seed = 42
    noise_seed_test = 3


    # ============================================================
    # 5. Generate MPSK waveforms
    # ============================================================
    M = 4
    eta = 1.0
    N_REPEAT = 500


    train_phases, train_sols = simulate_mpsk_repeated_noise(
        M,
        t_center,
        t_eval,
        t_end,
        sigma,
        Iin_peak,
        fc,
        eta,
        n_repeat=N_REPEAT,
        alpha=0.0,
        add_noise=noise_enabled,
        snr_db=snr_db,
        seed=noise_seed,
        adc_enabled=adc_enabled,
        fs_adc=fs_adc,
        adc_bits=adc_bits
    )

    sample_times, x_train, y_train = build_repeated_noise_dataset(
        sols_by_class=train_sols,
        K=K,
        t_start=t_sample_start,
        t_end=t_sample_end
    )

    mu = x_train.mean(axis=0)
    std = x_train.std(axis=0)
            
    print("X_train shape:", x_train.shape)
    print("mu shape:", mu.shape)
    print("std shape:", std.shape)
    print("std:", std)
    std[std < 1e-12] = 1.0
    
    x_train_n = (x_train - mu) / std

    Wout = train_linear_readout(
        x_train_n,
        y_train,
        n_classes=4,
        ridge=1e-6
    )

    for phase_deg in [0, 20, 45]:
        test_phases, test_sols = simulate_mpsk_repeated_noise_offset(
            M,
            t_center,
            t_eval,
            t_end,
            sigma,
            Iin_peak,
            fc,
            eta,
            n_repeat=200,
            alpha=0.0,
            add_noise=noise_enabled,
            snr_db=snr_db,
            seed=noise_seed_test,
            adc_enabled=adc_enabled,
            fs_adc=fs_adc,
            adc_bits=adc_bits,
            amp_scale=1.0,
            time_delay=0.0,
            phase_offset=np.deg2rad(phase_deg),
            freq_offset=0.0,
        )


        sample_times, x_test, y_test = build_repeated_noise_dataset(
            sols_by_class=test_sols,
            K=K,
            t_start=t_sample_start,
            t_end=t_sample_end
        )

        x_test_n  = (x_test  - mu) / std
            
    
        x_pred, x_scores = predict_linear_readout(
            x_train_n,
            Wout
        )
        y_pred, y_scores = predict_linear_readout(
            x_test_n,
            Wout
        )
        print(f"Train accuracy = {np.mean(x_pred == y_train)*100:.2f}%")
        print(f"Test accuracy  = {np.mean(y_pred == y_test)*100:.2f}%")

        print(confusion_matrix(y_test, y_pred))
        acc = np.mean(y_pred == y_test)
        
        print(f"Test accuracy = {100*acc:.2f}%")



    '''mod_name = {
        2: "BPSK",
        4: "QPSK",
        8: "8-PSK",
        16: "16-PSK",
        32: "32-PSK"
    }.get(M, f"{M}-PSK")

    

    # ============================================================
    # sampling check
    # ============================================================

    print("\n=== Repeated-noise temporal sampling sweep ===")

    for K in K_list:
        print('k: ', K)
        # --------------------------------------------------------
        # Build sampled reservoir feature dataset
        # --------------------------------------------------------
        sample_times, x_train, y_train = build_repeated_noise_dataset(
            sols_by_class=train_sols,
            K=K,
            t_start=t_sample_start,
            t_end=t_sample_end
        )
        sample_times, x_test, y_test = build_repeated_noise_dataset(
            sols_by_class=test_sols,
            K=K,
            t_start=t_sample_start,
            t_end=t_sample_end
        )



        mu = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        std = np.where(std < 1e-12, 1.0, std)

        x_train_n = (x_train - mu) / std
        x_test_n  = (x_test  - mu) / std

        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(x_train_n)
        X_test_pca  = pca.transform(x_test_n)
        plt.figure(figsize=(8, 7))

        for c in np.unique(y_train):
            idx_tr = (y_train == c)
            idx_te = (y_test == c)

            plt.scatter(
                X_train_pca[idx_tr, 0],
                X_train_pca[idx_tr, 1],
                alpha=0.35,
                s=25,
                label=f"class {c} train"
            )

            plt.scatter(
                X_test_pca[idx_te, 0],
                X_test_pca[idx_te, 1],
                marker="x",
                s=35,
                label=f"class {c} test"
            )

        plt.xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}% variance)")
        plt.ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}% variance)")
        plt.title(f"Train/Test PCA, K={K}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
            
        
        mu = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        print("X_train shape:", x_train.shape)
        print("mu shape:", mu.shape)
        print("std shape:", std.shape)
        print("std:", std)
        std[std < 1e-12] = 1.0
        
    
        Wout = train_linear_readout(
            x_train_n,
            y_train,
            n_classes=4,
            ridge=1e-6
        )

        x_pred, x_scores = predict_linear_readout(
            x_train_n,
            Wout
        )
        y_pred, y_scores = predict_linear_readout(
            x_test_n,
            Wout
        )
        print(f"Train accuracy = {np.mean(x_pred == y_train)*100:.2f}%")
        print(f"Test accuracy  = {np.mean(y_pred == y_test)*100:.2f}%")

        print(confusion_matrix(y_test, y_pred))
        
        acc = np.mean(y_pred == y_test)
        
        print(f"Test accuracy = {100*acc:.2f}%")'''

if __name__=="__main__":
    main()