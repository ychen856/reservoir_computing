"""Standalone PT voltage-decay experiment.

Place at src/run_voltage_decay.py and run from project root:
    python3 -m src.run_voltage_decay
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

from src.circuits.pt_single import PTSingleReservoir
from src.simulation.runner import validate_solution
from src.tasks.properties.voltage_decay import evaluate_voltage_decay

# Fixed PT configuration, consistent with the other property runners.
ETA = 1.0
ALPHA = 8.33
FC = 48e6
INPUT_SCALE = 1e-4  # A, peak current
SOLVER = "ivp"       # continuous Gaussian-modulated input, NOT ZOH

T_END = 1.5e-6
N_TIME = 6000
PULSE_CENTER = 0.12e-6
PULSE_SIGMA = 0.035e-6
PULSE_OFF = PULSE_CENTER + 4 * PULSE_SIGMA  # hard cutoff: 0.26 us
FIT_START = 0.30e-6
FIT_END = 1.25e-6
FLOOR_FRACTION = 0.02
OUTPUT = Path("results/voltage_decay")


def input_current(t):
    # Smooth pulse, with a small (~0.034% of peak) discontinuity at cutoff.
    if t >= PULSE_OFF:
        return 0.0
    envelope = np.exp(-0.5 * ((t - PULSE_CENTER) / PULSE_SIGMA) ** 2)
    return INPUT_SCALE * envelope * np.cos(2 * np.pi * FC * (t - PULSE_CENTER))


def main():
    if not (PULSE_OFF < FIT_START < FIT_END < T_END):
        raise ValueError("Fit must start strictly after input cutoff")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    t_eval = np.linspace(0.0, T_END, N_TIME, endpoint=False)
    reservoir = PTSingleReservoir(eta=ETA, alpha=ALPHA)
    solution = reservoir.simulate(
        input_func=input_current,
        t_span=(0.0, T_END),
        t_eval=t_eval,
        solver=SOLVER,
        symbol_period=None,
    )
    validate_solution(solution)
    t = np.asarray(solution.t)
    v = np.asarray(solution.y[:2])
    if v.shape != (2, t.size):
        raise RuntimeError(f"Unexpected voltage shape: {v.shape}")

    results = []
    envelopes = np.abs(hilbert(v, axis=1))
    rows = []
    for channel in range(2):
        result = evaluate_voltage_decay(
            t, v[channel], fit_start=FIT_START, fit_end=FIT_END,
            floor_fraction=FLOOR_FRACTION,
        )
        results.append(result)
        rows.append({
            "channel": f"V{channel + 1}", "eta": ETA, "alpha": ALPHA,
            "input_peak_A": INPUT_SCALE, "pulse_off_s": PULSE_OFF,
            "fit_start_s": FIT_START, "fit_end_s": FIT_END,
            "tau_s": result.tau_seconds, "tau_us": result.tau_seconds * 1e6,
            "slope_per_s": result.slope, "fit_r_squared": result.r_squared,
            "n_fit": result.n_fit, "solver": SOLVER,
        })

    with (OUTPUT / "decay_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    np.savez(OUTPUT / "waveforms.npz", t=t, V1=v[0], V2=v[1],
             envelope_V1=envelopes[0], envelope_V2=envelopes[1],
             input_current=np.array([input_current(ti) for ti in t]))

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t * 1e6, v[i], lw=0.6, alpha=0.5, label=f"V{i+1}(t)")
        ax.plot(t * 1e6, envelopes[i], lw=1.2, label="Hilbert envelope")
        ax.axvline(PULSE_OFF * 1e6, color="black", ls=":", label="Input off")
        ax.axvspan(FIT_START * 1e6, FIT_END * 1e6, alpha=0.10,
                   label="Fit interval")
        result = results[i]
        if np.isfinite(result.tau_seconds):
            fit_t = t[(t >= FIT_START) & (t <= FIT_END)]
            # Display the fitted exponential with its slope, anchored to
            # the envelope at the beginning of the fit interval.
            idx = np.argmin(np.abs(t - FIT_START))
            fit_y = envelopes[i, idx] * np.exp(result.slope * (fit_t - t[idx]))
            ax.plot(fit_t * 1e6, fit_y, "--", lw=1.4,
                    label=f"Decay slope (tau={result.tau_seconds*1e6:.3f} us)")
        ax.set_ylabel(f"V{i+1} [V]")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time [us]")
    fig.suptitle(f"PT Single: Voltage Decay (eta={ETA}, alpha={ALPHA})")
    fig.tight_layout()
    fig.savefig(OUTPUT / "voltage_decay.png", dpi=200)
    plt.close(fig)

    print("\n=== Voltage Decay ===")
    print(f"Circuit       : eta={ETA}, alpha={ALPHA}")
    print(f"Input         : Gaussian RF pulse, peak={INPUT_SCALE:g} A, fc={FC/1e6:g} MHz")
    print(f"Input off     : {PULSE_OFF*1e6:.3f} us (hard cutoff)")
    print(f"Fit interval  : {FIT_START*1e6:.3f}..{FIT_END*1e6:.3f} us")
    print(f"Solver        : {SOLVER}")
    for channel, result in enumerate(results, 1):
        print(f"V{channel} tau       : {result.tau_seconds*1e6:.6f} us; "
              f"R^2={result.r_squared:.5f}; n={result.n_fit}")
    print(f"Saved to      : {OUTPUT}")
    print("Note: single-exponential tau is descriptive; assess fit R^2 and plots.")


if __name__ == "__main__":
    main()
