import numpy as np
import matplotlib.pyplot as plt
from src.circuits.pt_single import PTSingleReservoir
from src.tasks.benchmarks.rossler import generate_rossler

# Reuse these imports/paths from your existing Hénon runner if their module differs.
from src.simulation.evaluation import evaluate_continuous_regression
from src.simulation.runner import extract_continuous_features
from src.simulation.experiment import ExperimentConfig, ReadoutExperiment


# ---------------- Configuration ----------------
N_SAMPLES = 3000
WASHOUT = 200
TRAIN_RATIO = 0.8

ROSS_DT = 0.1
ROSS_TRANSIENT = 1000
A, B, C = 0.2, 0.2, 5.7

FC = 48e6
TC = 1.0 / FC
N_CYCLES_PER_SYMBOL = 8.0 #1 / 2
T_SYMBOL = TC * N_CYCLES_PER_SYMBOL

POINTS_PER_SYMBOL = 16
K = 8
I_SCALE = 1e-4
ALPHA = 8.33 #2.78
ETA = 1.0
RIDGE = 1e-6
HISTORY_TAPS = 16



def fit_eval(name, X, Y, train_ratio):

    result = evaluate_continuous_regression(
        X=X,
        y=Y,
        washout=0,
        train_ratio=train_ratio,
        ridge=RIDGE,
        normalize=True,
    )

    vals = {
        "name": name,
        "dim": result.feature_dim,

        "y_train": result.train_nrmse_per_output[0],
        "y_test": result.test_nrmse_per_output[0],

        "z_train": result.train_nrmse_per_output[1],
        "z_test": result.test_nrmse_per_output[1],

        "yr_train": result.train_r_per_output[0],
        "yr_test": result.test_r_per_output[0],

        "zr_train": result.train_r_per_output[1],
        "zr_test": result.test_r_per_output[1],
    }

    print(f"\n=== {name} ===")
    print(f"Feature dim  : {vals['dim']}")

    print(
        f"y Train/Test NRMSE : "
        f"{vals['y_train']:.6f} / {vals['y_test']:.6f}"
    )

    print(
        f"z Train/Test NRMSE : "
        f"{vals['z_train']:.6f} / {vals['z_test']:.6f}"
    )

    print(
        f"y Train/Test r     : "
        f"{vals['yr_train']:.6f} / {vals['yr_test']:.6f}"
    )

    print(
        f"z Train/Test r     : "
        f"{vals['zr_train']:.6f} / {vals['zr_test']:.6f}"
    )

    return vals


def main():
    task_data = generate_rossler(
        n_samples=N_SAMPLES, dt=ROSS_DT, transient=ROSS_TRANSIENT,
        a=A, b=B, c=C, washout=WASHOUT, train_ratio=TRAIN_RATIO
    )
    u = task_data.u
    Y_all = task_data.y
    x = task_data.metadata['raw_x']
    input_scale = task_data.metadata['input_scale']
    y = Y_all[:, 0]
    z = Y_all[:, 1]
    n_symbols = len(u)


    # ============================================================
    # Common evaluation set
    # ============================================================

    valid_idx = np.arange(WASHOUT, n_symbols)

    Y_eval = Y_all[valid_idx]

    n_train = int(len(valid_idx) * TRAIN_RATIO)

    print("\n=== Common evaluation set ===")
    print(f"Valid sample range : [{valid_idx[0]}, {valid_idx[-1]}]")
    print(f"Samples            : {len(valid_idx)}")
    print(f"Train samples      : {n_train}")
    print(f"Test samples       : {len(valid_idx) - n_train}")


    X_current = u[valid_idx, None]

    fit_eval(
        "Current observable x[n]",
        X_current,
        Y_eval,
        TRAIN_RATIO,
    )

    X_history = np.column_stack([
        np.roll(u, d)
        for d in range(HISTORY_TAPS)
    ])

    X_history = X_history[valid_idx]

    fit_eval(
        "10-tap input history",
        X_history,
        Y_eval,
        TRAIN_RATIO,
    )



    # Only x is observable/input. Normalize only for physical drive.
    
    '''input_func = sample_hold_input(u, T_SYMBOL, amplitude_scale=I_SCALE)
    t_end = n_symbols * T_SYMBOL
    t_eval = np.linspace(0.0, t_end, n_symbols * POINTS_PER_SYMBOL, endpoint=False)

    reservoir = PTSingleReservoir(eta=ETA, alpha=ALPHA)
    sol = reservoir.simulate(input_func=input_func,
                             t_span=(0.0, t_end),
                             t_eval=t_eval,
                             x0=np.zeros(4))

    if not sol.success:
        raise RuntimeError(
            f"Reservoir simulation failed: {sol.message}"
        )

    if not np.all(np.isfinite(sol.y)):
        raise RuntimeError(
            "Reservoir simulation contains NaN or Inf"
        )

    Xpt = extract_temporal_features(sol, n_symbols, T_SYMBOL, k=K)'''

    # ============================================================
    # Common reservoir simulation
    # ============================================================

    reservoir = PTSingleReservoir(
        eta=ETA,
        alpha=ALPHA,
    )


    # ============================================================
    # Experiment configuration
    # ============================================================

    #K_LIST = [1, 2, 4, 8]
    #RIDGE_LIST = [1e-8, 1e-6, 1e-4]

    K_LIST = [8]
    RIDGE_LIST = [1e-6]
    config = ExperimentConfig(
        symbol_period=T_SYMBOL,
        points_per_symbol=POINTS_PER_SYMBOL,
        k=8,
        amplitude_scale=I_SCALE,
        encoding="zoh",
        ridge=RIDGE,
        normalize=True,
        evaluation_type="regression",
        solver='exact_zoh',
    )
    
    experiment = ReadoutExperiment(
        reservoir=reservoir,
        task_data=task_data,
        config=config,
        reservoir_name="PT Single",
    )

    # Run reservoir simulation once for all K/ridge combinations
    simulation = experiment.simulate()

    result = experiment.evaluate(
        simulation=simulation,
        k=K,
        ridge=RIDGE,
    )

    ev = result.evaluation

    print("\n=== PT Single reservoir ===")
    print(f"Feature dim : {ev.feature_dim}")

    print(
        f"y Train/Test NRMSE : "
        f"{ev.train_nrmse_per_output[0]:.6f} / "
        f"{ev.test_nrmse_per_output[0]:.6f}"
    )

    print(
        f"z Train/Test NRMSE : "
        f"{ev.train_nrmse_per_output[1]:.6f} / "
        f"{ev.test_nrmse_per_output[1]:.6f}"
    )

    print(
        f"y Train/Test r     : "
        f"{ev.train_r_per_output[0]:.6f} / "
        f"{ev.test_r_per_output[0]:.6f}"
    )

    print(
        f"z Train/Test r     : "
        f"{ev.train_r_per_output[1]:.6f} / "
        f"{ev.test_r_per_output[1]:.6f}"
    )



if __name__ == '__main__':
    main()
