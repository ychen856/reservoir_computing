import numpy as np
from src.tasks.base import TaskData


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



def apply_adc_to_waveform(
    waveform,
    t_eval,
    fs_adc,
    adc_bits=None,
    full_scale=None,
):
    """
    ADC sampling + optional quantization + ZOH reconstruction.

    Returns waveform evaluated on the original t_eval grid.
    """

    waveform = np.asarray(waveform, dtype=float)
    t_eval = np.asarray(t_eval, dtype=float)

    dt_adc = 1.0 / fs_adc

    t_adc = np.arange(
        t_eval[0],
        t_eval[-1] + dt_adc,
        dt_adc,
    )

    # Sample analog waveform
    u_adc = np.interp(
        t_adc,
        t_eval,
        waveform,
    )

    # Optional quantization
    if adc_bits is not None:

        if full_scale is None:
            raise ValueError(
                "full_scale is required for ADC quantization"
            )

        levels = 2 ** adc_bits

        u_adc_clip = np.clip(
            u_adc,
            -full_scale,
            full_scale,
        )

        code = np.round(
            (u_adc_clip + full_scale)
            / (2.0 * full_scale)
            * (levels - 1)
        )

        u_adc = (
            code / (levels - 1)
            * 2.0 * full_scale
            - full_scale
        )

    # Zero-order hold on original simulation grid
    indices = np.searchsorted(
        t_adc,
        t_eval,
        side="right",
    ) - 1

    indices = np.clip(
        indices,
        0,
        len(u_adc) - 1,
    )

    return u_adc[indices]







def simulate_mpsk_dataset(
    reservoir,
    task_data,
):
    """
    Simulate each MPSK waveform independently.
    """

    t_eval = task_data.metadata["t_eval"]

    sols = []

    for i, waveform in enumerate(task_data.u):

        def input_func(t, u=waveform):
            return np.interp(
                t,
                t_eval,
                u,
                left=0.0,
                right=0.0,
            )

        sol = reservoir.simulate(
            input_func=input_func,
            t_span=(t_eval[0], t_eval[-1]),
            t_eval=t_eval,
        )

        if not sol.success:
            raise RuntimeError(
                f"Trial {i}: {sol.message}"
            )

        if not np.all(np.isfinite(sol.y)):
            raise RuntimeError(
                f"Trial {i}: NaN/Inf in reservoir states"
            )

        sols.append(sol)

    return sols


def generate_mpsk_task(
    M,
    n_repeat,
    t_eval,
    t_center,
    sigma,
    Iin_peak,
    fc,
    add_noise=False,
    snr_db=20.0,
    seed=42,
    adc_enabled=False,
    fs_adc=None,
    adc_bits=None,
):
    """
    Generate independent MPSK waveform trials.

    Each row of U is one complete waveform.
    """

    rng = np.random.default_rng(seed)

    phases = get_mpsk_phases(M)

    waveforms = []
    labels = []

    power_mask = (
        np.abs(t_eval - t_center) <= 3.0 * sigma
    )

    for m, phase in enumerate(phases):

        u_clean = phase_modulated_pulse(
            t=t_eval,
            center=t_center,
            sigma=sigma,
            I_peak=Iin_peak,
            fc=fc,
            phase=phase,
        )

        if add_noise:
            signal_power = np.mean(
                u_clean[power_mask] ** 2
            )

            noise_std = np.sqrt(
                signal_power / (10.0 ** (snr_db / 10.0))
            )


        for _ in range(n_repeat):

            if add_noise:
                noise = rng.normal(
                    0.0,
                    noise_std,
                    size=len(t_eval),
                )

                u = u_clean + noise
            else:
                u = u_clean.copy()

            if adc_enabled:
                if fs_adc is None:
                    raise ValueError(
                        "fs_adc must be specified when ADC is enabled"
                    )
            
                u = apply_adc_to_waveform(
                    waveform=u,
                    t_eval=t_eval,
                    fs_adc=fs_adc,
                    adc_bits=adc_bits,
                    full_scale=1.2 * Iin_peak,
                )

            waveforms.append(u)
            labels.append(m)

    U = np.asarray(waveforms)
    y = np.asarray(labels, dtype=int)

    return TaskData(
        name=f"{M}psk",
        u=U,
        y=y,
        task_type="classification",
        washout=0,
        train_ratio=0.8,
        target_offset=0,
        metadata={
            "M": M,
            "phases": phases,
            "t_eval": t_eval,
            "t_center": t_center,
            "sigma": sigma,
            "fc": fc,
            "Iin_peak": Iin_peak,
            "n_repeat": n_repeat,
            "add_noise": add_noise,
            "snr_db": snr_db,
            "seed": seed,
            "simulation_mode": "independent_trials",
            "readout_type": "onehot_ridge_classifier",
            "adc_enabled": adc_enabled,
            "fs_adc": fs_adc,
            "adc_bits": adc_bits,
        },
    )
