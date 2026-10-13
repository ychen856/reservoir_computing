import numpy as np

from src.circuits.pt_single import PTSingleReservoir
from src.tasks.benchmarks.rossler import generate_rossler

# Reuse these imports/paths from your existing Hénon runner if their module differs.
from src.utils.encoding import sample_hold_input
from src.utils.features import extract_temporal_features
from src.readout.ridge import train_ridge_regression, predict_ridge_regression
from src.readout.metrics import nrmse
from src.readout.preprocessing import fit_standardizer, transform_standardizer


# ---------------- Configuration ----------------
N_SAMPLES = 3000
WASHOUT = 200
TRAIN_RATIO = 0.8

ROSS_DT = 0.1
ROSS_TRANSIENT = 1000
A, B, C = 0.2, 0.2, 5.7

FC = 48e6
TC = 1.0 / FC
N_CYCLES_PER_SYMBOL = 1 / 5
T_SYMBOL = TC * N_CYCLES_PER_SYMBOL

POINTS_PER_SYMBOL = 16
K = 8
I_SCALE = 1e-2
ALPHA = 2.78
ETA = 1.0
RIDGE = 1e-6
HISTORY_TAPS = 10


def pearson_r(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if np.std(y_true) < 1e-15 or np.std(y_pred) < 1e-15:
        return np.nan
    return np.corrcoef(y_true, y_pred)[0, 1]


def fit_eval(name, X, Y, n_train):
    Xtr, Xte = X[:n_train], X[n_train:]
    Ytr, Yte = Y[:n_train], Y[n_train:]


    mu, std = fit_standardizer(Xtr)

    Xtr = transform_standardizer(Xtr, mu, std)
    Xte = transform_standardizer(Xte, mu, std)

    W = train_ridge_regression(Xtr, Ytr, ridge=RIDGE)
    Ptr = predict_ridge_regression(Xtr, W)
    Pte = predict_ridge_regression(Xte, W)

    vals = {
        'name': name,
        'dim': X.shape[1],
        'y_train': nrmse(Ytr[:, 0], Ptr[:, 0]),
        'y_test': nrmse(Yte[:, 0], Pte[:, 0]),
        'z_train': nrmse(Ytr[:, 1], Ptr[:, 1]),
        'z_test': nrmse(Yte[:, 1], Pte[:, 1]),
        'yr_train': pearson_r(Ytr[:, 0], Ptr[:, 0]),
        'yr_test': pearson_r(Yte[:, 0], Pte[:, 0]),
        'zr_train': pearson_r(Ytr[:, 1], Ptr[:, 1]),
        'zr_test': pearson_r(Yte[:, 1], Pte[:, 1]),
    }

    print(f"\n=== {name} ===")
    print(f"Feature dim  : {vals['dim']}")
    print(f"y Train/Test NRMSE : {vals['y_train']:.6f} / {vals['y_test']:.6f}")
    print(f"z Train/Test NRMSE : {vals['z_train']:.6f} / {vals['z_test']:.6f}")
    print(f"y Train/Test r     : {vals['yr_train']:.6f} / {vals['yr_test']:.6f}")
    print(f"z Train/Test r     : {vals['zr_train']:.6f} / {vals['zr_test']:.6f}")
    return vals


def main():
    _, x, y, z = generate_rossler(
        n_samples=N_SAMPLES, dt=ROSS_DT, transient=ROSS_TRANSIENT,
        a=A, b=B, c=C
    )

    # Only x is observable/input. Normalize only for physical drive.
    input_scale = np.max(np.abs(x))
    u = x / input_scale

    input_func = sample_hold_input(u, T_SYMBOL, amplitude_scale=I_SCALE)
    t_end = N_SAMPLES * T_SYMBOL
    t_eval = np.linspace(0.0, t_end, N_SAMPLES * POINTS_PER_SYMBOL, endpoint=False)

    reservoir = PTSingleReservoir(eta=ETA, alpha=ALPHA)
    sol = reservoir.simulate(input_func=input_func,
                             t_span=(0.0, t_end),
                             t_eval=t_eval,
                             x0=np.zeros(4))
    Xpt = extract_temporal_features(sol, N_SAMPLES, T_SYMBOL, k=K)

    # Common range must support the explicit history baseline.
    start = max(WASHOUT, HISTORY_TAPS - 1)
    idx = np.arange(start, N_SAMPLES)
    Y = np.column_stack([y[idx], z[idx]])

    X_current = x[idx, None]
    X_history = np.column_stack([
        x[idx - d] for d in range(HISTORY_TAPS)
    ])
    X_pt = Xpt[idx]

    n = len(idx)
    n_train = int(TRAIN_RATIO * n)

    print(f"Tc               = {TC*1e9:.4f} ns")
    print(f"T_SYMBOL         = {T_SYMBOL*1e9:.4f} ns")
    print(f"dt               = {T_SYMBOL/POINTS_PER_SYMBOL*1e12:.4f} ps")
    print(f"points/symbol    = {POINTS_PER_SYMBOL:.1f}")
    print(f"temporal nodes K = {K}")
    print(f"t_eval points    = {len(t_eval)}")
    print(f"Rossler a,b,c    = {A}, {B}, {C}")
    print(f"Rossler dt       = {ROSS_DT}")
    print(f"input scale      = {input_scale:.6f}")
    print(f"x range          = {x.min():.6f} {x.max():.6f}")
    print(f"y range          = {y.min():.6f} {y.max():.6f}")
    print(f"z range          = {z.min():.6f} {z.max():.6f}")
    print(f"normalized input = {u.min():.6f} {u.max():.6f}")
    print(f"V1 range         = {sol.y[0].min():.6f} {sol.y[0].max():.6f}")
    print(f"V2 range         = {sol.y[1].min():.6f} {sol.y[1].max():.6f}")

    print("\n=== Common evaluation set ===")
    print(f"Valid sample range : [{idx[0]}, {idx[-1]}]")
    print(f"Samples            : {n}")
    print(f"Train samples      : {n_train}")
    print(f"Test samples       : {n-n_train}")
    print(f"Ridge              : {RIDGE:.1e}")

    print("\n" + "="*68)
    print("ROSSLER OBSERVER BENCHMARK")
    print("="*68)
    print("Task               : x[n] -> [y[n], z[n]]")
    print("Reservoir          : PT Single")
    print(f"alpha              : {ALPHA}")
    print(f"T_symbol / Tc      : {N_CYCLES_PER_SYMBOL}")
    print(f"I_scale            : {I_SCALE:.3e} A")
    print(f"K                  : {K}")
    print(f"history taps       : {HISTORY_TAPS}")

    results = [
        fit_eval("Current observable x[n]", X_current, Y, n_train),
        fit_eval(f"{HISTORY_TAPS}-tap x history", X_history, Y, n_train),
        fit_eval("PT Single reservoir", X_pt, Y, n_train),
    ]

    print("\n=== Test summary ===")
    print(f"{'Method':30s} {'Dim':>5s} {'y NRMSE':>10s} {'z NRMSE':>10s} {'y r':>10s} {'z r':>10s}")
    print("-"*82)
    for r in results:
        print(f"{r['name']:30s} {r['dim']:5d} {r['y_test']:10.6f} {r['z_test']:10.6f} "
              f"{r['yr_test']:10.6f} {r['zr_test']:10.6f}")


if __name__ == '__main__':
    main()
