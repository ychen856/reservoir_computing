import numpy as np

from dataclasses import dataclass
from typing import Any, Optional

from src.utils.encoding import (
    sample_hold_input,
    carrier_encoded_input,
)

from src.utils.features import (
    extract_temporal_features,
    build_mpsk_feature_matrix,
)


# ============================================================
# Simulation result
# ============================================================

@dataclass
class SimulationResult:

    X: np.ndarray
    y: np.ndarray

    solutions: Any

    t_eval: np.ndarray

    metadata: dict


# ============================================================
# Validation
# ============================================================

def validate_solution(sol):

    if not sol.success:
        raise RuntimeError(
            f"Reservoir simulation failed: {sol.message}"
        )

    if not np.all(np.isfinite(sol.y)):
        raise RuntimeError(
            "NaN/Inf detected in reservoir states"
        )


# ============================================================
# Continuous sequence simulation
# ============================================================

def simulate_continuous_task(
    reservoir,
    task_data,
    symbol_period,
    points_per_symbol=16,
    k=8,
    amplitude_scale=1e-2,
    encoding="zoh",
    carrier_frequency=None,
):
    """
    Continuous reservoir simulation.

    Supported examples
    ------------------
    Sine -> Cube
    NARMA-10
    Henon
    Rossler
    PAM4

    Returns
    -------
    SimulationResult
    """

    u = np.asarray(task_data.u)

    if u.ndim != 1:
        raise ValueError(
            "Continuous simulation requires 1D input"
        )

    n_symbols = len(u)

    # --------------------------------------------------------
    # Input encoding
    # --------------------------------------------------------

    if encoding == "zoh":

        input_func = sample_hold_input(
            u,
            symbol_period=symbol_period,
            amplitude_scale=amplitude_scale,
        )

    elif encoding == "carrier":

        if carrier_frequency is None:
            raise ValueError(
                "carrier_frequency is required"
            )

        input_func = carrier_encoded_input(
            u,
            symbol_period=symbol_period,
            carrier_frequency=carrier_frequency,
            amplitude_scale=amplitude_scale,
        )

    else:

        raise ValueError(
            f"Unsupported encoding: {encoding}"
        )

    # --------------------------------------------------------
    # Simulation time
    # --------------------------------------------------------

    dt = symbol_period / points_per_symbol

    t_end = n_symbols * symbol_period

    t_eval = np.arange(
        0.0,
        t_end,
        dt,
    )

    # --------------------------------------------------------
    # Reservoir simulation
    # --------------------------------------------------------

    sol = reservoir.simulate(
        input_func=input_func,
        t_span=(0.0, t_end),
        t_eval=t_eval,
    )

    validate_solution(sol)

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    X = extract_temporal_features(
        sol,
        n_symbols=n_symbols,
        symbol_period=symbol_period,
        k=k,
    )

    if len(X) != len(task_data.y):
        raise RuntimeError(
            "Feature/target length mismatch"
        )

    return SimulationResult(
        X=X,
        y=np.asarray(task_data.y),
        solutions=sol,
        t_eval=t_eval,
        metadata={
            "simulation_mode": "continuous",
            "n_symbols": n_symbols,
            "symbol_period": symbol_period,
            "points_per_symbol": points_per_symbol,
            "k": k,
            "feature_dim": X.shape[1],
            "encoding": encoding,
            "amplitude_scale": amplitude_scale,
        },
    )


# ============================================================
# Independent trial simulation
# ============================================================

def simulate_independent_trials(
    reservoir,
    task_data,
    k=8,
    t_start=None,
    t_end=None,
    extract_features=False,
):
    """
    Independent waveform simulation.

    Supported examples
    ------------------
    BPSK
    QPSK
    8PSK
    16PSK
    32PSK

    Each waveform starts from the reservoir's default
    initial state.

    Returns
    -------
    SimulationResult
    """

    u = np.asarray(task_data.u)

    if u.ndim != 2:
        raise ValueError(
            "Independent trials require 2D input"
        )

    t_eval = np.asarray(
        task_data.metadata["t_eval"]
    )

    if u.shape[1] != len(t_eval):
        raise ValueError(
            "Waveform/time-grid length mismatch"
        )

    if t_start is None:
        t_start = t_eval[0]

    if t_end is None:
        t_end = t_eval[-1]

    # --------------------------------------------------------
    # Reservoir simulations
    # --------------------------------------------------------

    solutions = []

    for i, waveform in enumerate(u):

        def input_func(t, waveform=waveform):

            return np.interp(
                t,
                t_eval,
                waveform,
                left=0.0,
                right=0.0,
            )

        sol = reservoir.simulate(
            input_func=input_func,
            t_span=(t_eval[0], t_eval[-1]),
            t_eval=t_eval,
        )

        validate_solution(sol)

        solutions.append(sol)

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    sample_times, X = build_mpsk_feature_matrix(
        sols=solutions,
        K=k,
        t_start=t_start,
        t_end=t_end,
    )

    if len(X) != len(task_data.y):
        raise RuntimeError(
            "Feature/target length mismatch"
        )

    return SimulationResult(
        X=X,
        y=np.asarray(task_data.y),
        solutions=solutions,
        t_eval=t_eval,
        metadata={
            "simulation_mode": "independent_trials",
            "n_trials": len(u),
            "k": k,
            "feature_dim": X.shape[1],
            "sample_times": sample_times,
        },
    )