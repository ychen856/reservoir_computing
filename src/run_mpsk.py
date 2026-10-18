import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from src.circuits.pt_single import PTSingleReservoir
from src.tasks.applications.mpsk import simulate_mpsk_dataset, generate_mpsk_task
from src.readout.ridge import train_linear_readout, predict_linear_readout
from src.utils.features import build_mpsk_feature_matrix
from src.readout.preprocessing import fit_standardizer, transform_standardizer
from src.readout.metrics import accuracy, symbol_error_rate
from src.simulation.runner import simulate_independent_trials
from src.simulation.evaluation import evaluate_independent_classification


def main():
    # BPSK input verification: generate two phase-modulated Gaussian pulses and check that they are out of phase by pi radians.
    # ============================================================
    # 1. Carrier / pulse parameters
    # ============================================================
    TRAIN_SEED = 42
    TEST_SEED = 3

    N_TRAIN_REPEAT = 300
    N_TEST_REPEAT = 200

    M = 4

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


    # ============================================================
    # 5. Generate MPSK waveforms
    # ============================================================
    M = 4
    eta = 1.0
    N_REPEAT = 500


    pt_reservoir = PTSingleReservoir()
    train_data = generate_mpsk_task(
        M=M,
        n_repeat=N_TRAIN_REPEAT,
        t_eval=t_eval,
        t_center=t_center,
        sigma=sigma,
        Iin_peak=Iin_peak,
        fc=fc,
        add_noise=noise_enabled,
        snr_db=snr_db,
        seed=TRAIN_SEED,
        adc_enabled=adc_enabled,
        fs_adc=fs_adc,
        adc_bits=adc_bits,
    )
    test_data = generate_mpsk_task(
        M=M,
        n_repeat=N_TEST_REPEAT,
        t_eval=t_eval,
        t_center=t_center,
        sigma=sigma,
        Iin_peak=Iin_peak,
        fc=fc,
        add_noise=noise_enabled,
        snr_db=snr_db,
        seed=TEST_SEED,
        adc_enabled=adc_enabled,
        fs_adc=fs_adc,
        adc_bits=adc_bits,
    )
    

    #train_sols = simulate_mpsk_dataset(pt_reservoir, train_data)
    #test_sols = simulate_mpsk_dataset(pt_reservoir, test_data)
    train_result = simulate_independent_trials(
        reservoir=pt_reservoir,
        task_data=train_data,
        extract_features=False,
    )

    test_result = simulate_independent_trials(
        reservoir=pt_reservoir,
        task_data=test_data,
        extract_features=False,
    )

    train_sols = train_result.solutions
    test_sols = test_result.solutions


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
    RIDGE = 1e-6

    for K in K_list:

        print("k: ", K)

        sample_times, x_train = build_mpsk_feature_matrix(
            sols=train_sols,
            K=K,
            t_start=t_sample_start,
            t_end=t_sample_end,
        )

        _, x_test = build_mpsk_feature_matrix(
            sols=test_sols,
            K=K,
            t_start=t_sample_start,
            t_end=t_sample_end,
        )

        eval_result = evaluate_independent_classification(
            X_train=x_train,
            y_train=train_data.y,
            X_test=x_test,
            y_test=test_data.y,
            ridge=RIDGE,
            normalize=True,
            n_classes=M,
        )

        print(f"Train accuracy = {eval_result.train_accuracy:.2%}")
        print(f"Test accuracy  = {eval_result.test_accuracy:.2%}")
        print(eval_result.confusion_matrix)

        print(f"Train symbol error rate = {eval_result.train_ser:.2%}")
        print(f"Test symbol error rate  = {eval_result.test_ser:.2%}")


if __name__=="__main__":
    main()