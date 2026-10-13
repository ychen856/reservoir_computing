import numpy as np


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



def simulate_mpsk(
    pt_reservoir,
    M,
    t_center,
    t_eval,
    t_end,
    sigma,
    Iin_peak,
    fc,
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

        '''if alpha == 0.0:
            sol = pt_reservoir.simulate_history(
                t_end=t_end,
                t_eval=t_eval,
                eta=eta,
                input_func=input_func
            )

        else:

            sol = pt_reservoir.simulate_nonlinear(
                eta=eta,
                input_func=input_func,
                alpha=alpha,
                t_span=(t_eval[0], t_eval[-1]),
                t_eval=t_eval
            )'''
        sol = pt_reservoir.simulate(
            input_func=input_func,
            t_span=(t_eval[0], t_eval[-1]),
            t_eval=t_eval,
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
    pt_reservoir,
    M,
    t_center,
    t_eval,
    t_end,
    sigma,
    Iin_peak,
    fc,
    n_repeat,
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




            # ------------------------------------------------
            # Reservoir simulation
            # ------------------------------------------------
            sol = pt_reservoir.simulate(
                input_func=input_func,
                t_span=(t_eval[0], t_eval[-1]),
                t_eval=t_eval,
            )

            class_sols.append(sol)

        sols_by_class.append(class_sols)

        print(
            f"M={M}: completed phase "
            f"{m + 1}/{M}, "
            f"phi={phase / np.pi:.3f} pi"
        )

    return phases, sols_by_class
