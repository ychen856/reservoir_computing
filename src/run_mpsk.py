import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from src.circuits.pt_single import PTSingleReservoir
from src.tasks.applications.mpsk import simulate_mpsk_repeated_noise
from src.readout.ridge import train_linear_readout, predict_linear_readout
from src.utils.utils import build_repeated_noise_dataset
from src.readout.preprocessing import fit_standardizer, transform_standardizer
from src.readout.metrics import accuracy, symbol_error_rate




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


    pt_reservoir = PTSingleReservoir()

    train_phases, train_sols = simulate_mpsk_repeated_noise(
        pt_reservoir,
        M,
        t_center,
        t_eval,
        t_end,
        sigma,
        Iin_peak,
        fc,
        n_repeat=N_REPEAT,
        add_noise=noise_enabled,
        snr_db=snr_db,
        seed=noise_seed,
        adc_enabled=adc_enabled,
        fs_adc=fs_adc,
        adc_bits=adc_bits
    )
    test_phases, test_sols = simulate_mpsk_repeated_noise(
        pt_reservoir, 
        M,
        t_center,
        t_eval,
        t_end,
        sigma,
        Iin_peak,
        fc,
        n_repeat=200,
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



        mu, std = fit_standardizer(x_train)

        x_train_n = transform_standardizer(x_train, mu, std)
        x_test_n = transform_standardizer(x_test, mu, std)

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

        train_acc = accuracy(y_train, x_pred)
        test_acc = accuracy(y_test, y_pred)

        train_ser = symbol_error_rate(y_train, x_pred)
        test_ser = symbol_error_rate(y_test, y_pred)

        print(f"Train accuracy = {100*train_acc:.2f}%")
        print(f"Test accuracy  = {100*test_acc:.2f}%")

        print(confusion_matrix(y_test, y_pred))
        print(f"Train symbol error rate = {100*train_ser:.2f}%")
        print(f"Test symbol error rate  = {100*test_ser:.2f}%")

if __name__=="__main__":
    main()