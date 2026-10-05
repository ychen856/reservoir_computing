import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from src.circuits.pt_single import pt_reservoir_tank
from src.tasks.mpsk import simulate_nonlinear, simulate_history, phase_modulated_pulse, get_mpsk_phases
from src.readout.ridge import train_linear_readout, predict_linear_readout
from src.utils.utils import build_repeated_noise_dataset

def simulate_mpsk_repeated_noise(
    reservoir_tank,
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
                    reservoir_tank,
                    t_end=t_end,
                    t_eval=t_eval,
                    eta=eta,
                    input_func=input_func
                )

            else:

                sol = simulate_nonlinear(
                    reservoir_tank,
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


reservoir_tank = pt_reservoir_tank()
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
    adc_bits = 3        # None = sampling + ZOH only
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

    K_list = [1, 2, 4, 8, 16, 32]

    # ============================================================
    # noise configuration
    # ============================================================
    noise_enabled = True
    snr_db = 20.0
    noise_seed = 42


    # ============================================================
    # 5. Generate MPSK waveforms
    # ============================================================
    M = 4
    eta = 1.0
    N_REPEAT = 500


    train_phases, train_sols = simulate_mpsk_repeated_noise(
        reservoir_tank,
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
    test_phases, test_sols = simulate_mpsk_repeated_noise(
        reservoir_tank, 
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
        seed=noise_seed,
        adc_enabled=adc_enabled,
        fs_adc=fs_adc,
        adc_bits=adc_bits
    )
    

    mod_name = {
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
        
        print(f"Test accuracy = {100*acc:.2f}%")

if __name__=="__main__":
    main()