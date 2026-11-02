import csv
from pathlib import Path

import numpy as np

from src.tasks.applications.mpsk import generate_mpsk_task
from src.simulation.evaluation import evaluate_independent_classification


# ============================================================
# 1. Experiment configuration
# ============================================================

M_LIST = [4, 8, 16, 32]
K_LIST = [1, 2, 4, 8, 16, 32]

RIDGE = 1e-6

TRAIN_SEED = 42
TEST_SEED = 3

N_TRAIN_REPEAT = 300
N_TEST_REPEAT = 200


# ============================================================
# 2. Carrier / pulse configuration
# ============================================================

FC = 48.0e6
TC = 1.0 / FC

N_CYCLES = 8

PULSE_DURATION = N_CYCLES * TC
SIGMA = PULSE_DURATION / 6.0

T_CENTER = 0.5e-6

IIN_PEAK = 1.0e-4


# ============================================================
# 3. Time grid
# ============================================================

SAMPLES_PER_CYCLE = 40

DT = TC / SAMPLES_PER_CYCLE

T_START = 0.0
T_END = 1.2e-6

T_EVAL = np.arange(
    T_START,
    T_END + DT,
    DT
)


# ============================================================
# 4. ADC configuration
# ============================================================

ADC_ENABLED = True

FS_ADC = FC * 4
ADC_BITS = 3


# ============================================================
# 5. Noise configuration
# ============================================================

NOISE_ENABLED = True
SNR_DB = 20.0


# ============================================================
# 6. Sampling windows
# ============================================================

SAMPLING_WINDOWS = {

    # Same observation window as PT reservoir
    "same_time": (
        T_CENTER,
        T_CENTER + 0.7e-6
    ),

    # Gaussian pulse: center +/- 3 sigma
    "pulse_aligned": (
        T_CENTER - 3.0 * SIGMA,
        T_CENTER + 3.0 * SIGMA
    ),
}


# ============================================================
# 7. Output
# ============================================================

OUTPUT_CSV = Path("mpsk_input_baseline_comparison.csv")


# ============================================================
# 8. Input-only feature extraction
# ============================================================

def build_input_feature_matrix(
    waveforms,
    t_eval,
    K,
    t_start,
    t_end,
):
    """
    Extract K temporal samples directly from input waveforms.

    Parameters
    ----------
    waveforms:
        Shape (N_trials, N_time)

    t_eval:
        Shape (N_time,)

    K:
        Number of temporal sampling points.

    t_start, t_end:
        Sampling window.

    Returns
    -------
    sample_times:
        Shape (K,)

    X:
        Shape (N_trials, K)
    """

    sample_times = np.linspace(
        t_start,
        t_end,
        K
    )

    waveforms = np.asarray(waveforms)

    X = np.stack([
        np.interp(
            sample_times,
            t_eval,
            waveform
        )
        for waveform in waveforms
    ])

    return sample_times, X


# ============================================================
# 9. Evaluation
# ============================================================

def evaluate_baseline(
    train_data,
    test_data,
    M,
    K,
    window_name,
    t_start,
    t_end,
):

    sample_times, X_train = build_input_feature_matrix(
        waveforms=train_data.u,
        t_eval=T_EVAL,
        K=K,
        t_start=t_start,
        t_end=t_end,
    )

    _, X_test = build_input_feature_matrix(
        waveforms=test_data.u,
        t_eval=T_EVAL,
        K=K,
        t_start=t_start,
        t_end=t_end,
    )

    ev = evaluate_independent_classification(
        X_train=X_train,
        y_train=train_data.y,
        X_test=X_test,
        y_test=test_data.y,
        ridge=RIDGE,
        normalize=True,
        n_classes=M,
    )

    result = {
        "M": M,
        "window": window_name,
        "K": K,
        "feature_dim": ev.feature_dim,
        "ridge": RIDGE,
        "train_accuracy": ev.train_accuracy,
        "test_accuracy": ev.test_accuracy,
        "test_ser": ev.test_ser,
        "t_start_ns": t_start * 1e9,
        "t_end_ns": t_end * 1e9,
        "sample_times_ns": ";".join(
            f"{t * 1e9:.4f}"
            for t in sample_times
        ),
    }

    return result


# ============================================================
# 10. Main
# ============================================================

def main():

    print("\n========================================")
    print("M-PSK Input-only Baseline Comparison")
    print("========================================")

    print(f"Carrier frequency = {FC / 1e6:.3f} MHz")
    print(f"Input current     = {IIN_PEAK:.2e} A")
    print(f"Pulse sigma       = {SIGMA * 1e9:.3f} ns")
    print(f"Noise enabled     = {NOISE_ENABLED}")
    print(f"SNR               = {SNR_DB:.1f} dB")
    print(f"ADC enabled       = {ADC_ENABLED}")
    print(f"ADC sampling rate = {FS_ADC / 1e6:.1f} MS/s")
    print(f"ADC resolution    = {ADC_BITS} bits")
    print(f"Ridge             = {RIDGE:.0e}")

    print("\nSampling windows:")

    for name, (t_start, t_end) in SAMPLING_WINDOWS.items():

        print(
            f"  {name:15s}: "
            f"{t_start * 1e9:.2f} - "
            f"{t_end * 1e9:.2f} ns"
        )

    all_results = []

    # ========================================================
    # Sweep modulation order
    # ========================================================

    for M in M_LIST:

        print("\n========================================")
        print(f"M = {M}")
        print("========================================")

        # ----------------------------------------------------
        # Generate training waveforms
        # ----------------------------------------------------

        train_data = generate_mpsk_task(
            M=M,
            n_repeat=N_TRAIN_REPEAT,
            t_eval=T_EVAL,
            t_center=T_CENTER,
            sigma=SIGMA,
            Iin_peak=IIN_PEAK,
            fc=FC,
            add_noise=NOISE_ENABLED,
            snr_db=SNR_DB,
            seed=TRAIN_SEED,
            adc_enabled=ADC_ENABLED,
            fs_adc=FS_ADC,
            adc_bits=ADC_BITS,
        )

        # ----------------------------------------------------
        # Generate testing waveforms
        # ----------------------------------------------------

        test_data = generate_mpsk_task(
            M=M,
            n_repeat=N_TEST_REPEAT,
            t_eval=T_EVAL,
            t_center=T_CENTER,
            sigma=SIGMA,
            Iin_peak=IIN_PEAK,
            fc=FC,
            add_noise=NOISE_ENABLED,
            snr_db=SNR_DB,
            seed=TEST_SEED,
            adc_enabled=ADC_ENABLED,
            fs_adc=FS_ADC,
            adc_bits=ADC_BITS,
        )

        print(f"Training samples = {len(train_data.y)}")
        print(f"Testing samples  = {len(test_data.y)}")

        # ----------------------------------------------------
        # Evaluate both sampling windows
        # ----------------------------------------------------

        for K in K_LIST:

            print(f"\nK = {K}")

            for window_name, (t_start, t_end) in SAMPLING_WINDOWS.items():

                result = evaluate_baseline(
                    train_data=train_data,
                    test_data=test_data,
                    M=M,
                    K=K,
                    window_name=window_name,
                    t_start=t_start,
                    t_end=t_end,
                )

                all_results.append(result)

                print(
                    f"{window_name:15s} | "
                    f"dim={result['feature_dim']:2d} | "
                    f"train={result['train_accuracy']:.2%} | "
                    f"test={result['test_accuracy']:.2%}"
                )

    # ========================================================
    # Save CSV
    # ========================================================

    with OUTPUT_CSV.open(
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(all_results[0].keys())
        )

        writer.writeheader()
        writer.writerows(all_results)

    print("\n========================================")
    print("Baseline comparison completed")
    print("========================================")

    print(f"Results saved to: {OUTPUT_CSV.resolve()}")

    # ========================================================
    # Best result summary
    # ========================================================

    print("\n========================================")
    print("Best Test Accuracy Summary")
    print("========================================")

    for M in M_LIST:

        print(f"\nM = {M}")

        for window_name in SAMPLING_WINDOWS:

            candidates = [
                r for r in all_results
                if r["M"] == M
                and r["window"] == window_name
            ]

            best = max(
                candidates,
                key=lambda r: r["test_accuracy"]
            )

            print(
                f"{window_name:15s} | "
                f"best K={best['K']:2d} | "
                f"dim={best['feature_dim']:2d} | "
                f"test={best['test_accuracy']:.2%}"
            )


if __name__ == "__main__":
    main()