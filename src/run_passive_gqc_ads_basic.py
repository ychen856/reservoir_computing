"""Export a shared input current and raw Passive-GQC node voltages."""
from pathlib import Path

import numpy as np

from src.pass_gqc.common import Case


def main():
    # Circuit parameters. The defaults reproduce Design 37 at its EP.
    mode = "matched_linear"  # Change to "graphene" for the nonlinear model.
    l_nh = 0.4
    c1_pf = 10.0
    c2_pf = 10.00494191270113
    r1_ohm = 136.0
    r2_ohm = 152.92106495354943
    width_um = 2.0
    length_um = 1.0
    bias_v = -0.5

    # Input and export settings.
    waveform = "pulse"  # pulse or sine
    amplitude_ma = 1.0
    frequency_ghz = 2.5  # Used for sine.
    stop_ns = 16.0
    input_dt_ps = 2.0
    output_dt_ps = 20.0  # Set to 100.0 for 100 ps spacing.
    output_dir = Path("results/ads_compare/basic")

    if mode not in ("matched_linear", "graphene"):
        raise ValueError("mode must be matched_linear or graphene")
    case = Case(
        design_index=37, seed=44, mode=mode,
        width_um=width_um, length_um=length_um, bias_v=bias_v,
        c2_pf=c2_pf, r2_ohm=r2_ohm, r1_ohm=r1_ohm,
        l_nh=l_nh, c1_pf=c1_pf,
    )
    rc = case.reservoir()
    stop_ps = stop_ns * 1000
    if min(stop_ps, input_dt_ps, output_dt_ps) <= 0:
        raise ValueError("Duration and sampling intervals must be positive")
    t_in = np.r_[np.arange(0, stop_ps, input_dt_ps), stop_ps] * 1e-12
    t_out = np.r_[np.arange(0, stop_ps, output_dt_ps), stop_ps] * 1e-12

    # Positive current flows from ground into V1. Use these same PWL
    # samples in ADS, with linear interpolation between sample times.
    if waveform == "pulse":
        shape = np.interp(t_in * 1e9, [0, 1, 1.1, 3, 3.1], [0, 0, 1, 1, 0])
    elif waveform == "sine":
        shape = np.sin(2 * np.pi * frequency_ghz * 1e9 * t_in)
    else:
        raise ValueError("waveform must be pulse or sine")
    i_in = amplitude_ma * 1e-3 * shape

    # The reservoir defaults to zero initial voltages and currents.
    raw = rc.simulate(
        lambda t: float(np.interp(t, t_in, i_in)),
        (0.0, t_out[-1]), t_out,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / "input.csv", np.column_stack((t_in, i_in)),
               delimiter=",", header="time_s,i_in_a", comments="")
    np.savetxt(output_dir / "output.csv",
               np.column_stack((raw.t, raw.y[0], raw.y[1])),
               delimiter=",", header="time_s,v1_v,v2_v", comments="")

    capacitance, esr = rc.varactor.capacitance_and_resistance(bias_v)
    print(f"Mode: {mode}")
    print(f"GQC at bias: C = {capacitance * 1e12:.12g} pF, ESR = {esr:.12g} ohm")
    print(f"Saved input.csv and output.csv to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
