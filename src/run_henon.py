import numpy as np

from src.circuits.pt_single import PTSingleReservoir
from src.tasks.benchmarks.henon import generate_henon
from src.utils.encoding import sample_hold_input
from src.utils.features import extract_temporal_features
from src.readout.ridge import train_ridge_regression, predict_ridge_regression
from src.readout.metrics import nrmse
from src.readout.preprocessing import fit_standardizer, transform_standardizer
from src.readout.metrics import pearson_r, nrmse

# ============================================================
# Configuration
# ============================================================

N_SAMPLES = 3000
TRANSIENT = 1000
WASHOUT = 200
TRAIN_RATIO = 0.8

FC = 48e6
Tc = 1.0 / FC

N_CYCLES_PER_SYMBOL = 1 / 2
T_SYMBOL = N_CYCLES_PER_SYMBOL * Tc

I_SCALE = 1e-2
K = 8
POINTS_PER_SYMBOL = 16

RIDGE = 1e-6
ALPHA = 2.78

HENON_A = 1.4
HENON_B = 0.3


# ============================================================
# Helpers
# ============================================================


def evaluate_ridge(X_train, X_test, y_train, y_test, ridge=1e-6):
    mu, std = fit_standardizer(X_train)

    X_train_n = transform_standardizer(X_train, mu, std)
    X_test_n = transform_standardizer(X_test, mu, std)

    W = train_ridge_regression(X_train_n, y_train, ridge=ridge)
    y_train_pred = predict_ridge_regression(X_train_n, W)
    y_test_pred = predict_ridge_regression(X_test_n, W)

    return {
        "train_nrmse": nrmse(y_train, y_train_pred),
        "test_nrmse": nrmse(y_test, y_test_pred),
        "train_r": pearson_r(y_train, y_train_pred),
        "test_r": pearson_r(y_test, y_test_pred),
    }


def print_result(name, feature_dim, result):
    print(f"\n=== {name} ===")
    print(f"Feature dim  : {feature_dim}")
    print(f"Train NRMSE  : {result['train_nrmse']:.6f}")
    print(f"Test NRMSE   : {result['test_nrmse']:.6f}")
    print(f"Train r      : {result['train_r']:.6f}")
    print(f"Test r       : {result['test_r']:.6f}")


# ============================================================
# Generate Hénon trajectory
# ============================================================

task_data = generate_henon(
    n_samples=N_SAMPLES,
    a=HENON_A,
    b=HENON_B,
    transient=TRANSIENT,
    washout=WASHOUT,
    train_ratio=TRAIN_RATIO,
)
u = task_data.u
y = task_data.y

x = task_data.metadata['raw_x']
u_scale = task_data.metadata['input_scale']

n_symbols = len(u)

# Observable supplied to the reservoir: x[n]
# Hidden state to reconstruct: y[n]
# Normalize only the physical input amplitude. The target remains physical y[n].

input_func = sample_hold_input(
    u,
    symbol_period=T_SYMBOL,
    amplitude_scale=I_SCALE,
)


# ============================================================
# Physical simulation time
# ============================================================

dt = T_SYMBOL / POINTS_PER_SYMBOL
t_end = n_symbols * T_SYMBOL
t_eval = np.arange(0.0, t_end, dt)

print(f"Tc               = {Tc * 1e9:.4f} ns")
print(f"T_SYMBOL         = {T_SYMBOL * 1e9:.4f} ns")
print(f"dt               = {dt * 1e12:.4f} ps")
print(f"points/symbol    = {T_SYMBOL / dt:.1f}")
print(f"temporal nodes K = {K}")
print(f"t_eval points    = {len(t_eval)}")
print(f"Hénon a          = {HENON_A}")
print(f"Hénon b          = {HENON_B}")
print(f"input scale      = {u_scale:.6f}")


# ============================================================
# Reservoir
# ============================================================

reservoir = PTSingleReservoir(
    eta=1.0,
    alpha=ALPHA,
)

sol = reservoir.simulate(
    input_func=input_func,
    t_span=(0.0, t_end),
    t_eval=t_eval,
)

X = extract_temporal_features(
    sol,
    n_symbols=n_symbols,
    symbol_period=T_SYMBOL,
    k=K,
)

print("x shape:", x.shape)
print("y shape:", y.shape)
print("X shape:", X.shape)
print("x range:", np.min(x), np.max(x))
print("y range:", np.min(y), np.max(y))
print("normalized input range:", np.min(u), np.max(u))
print("V1 range:", np.min(sol.y[0]), np.max(sol.y[0]))
print("V2 range:", np.min(sol.y[1]), np.max(sol.y[1]))


# ============================================================
# Common evaluation set
# ============================================================
# We evaluate n >= WASHOUT and need x[n-1] for the oracle/history
# baseline.  All methods use exactly the same samples and split.
#
# Important interpretation:
#   y[n] = b*x[n-1]
# Therefore [x[n], x[n-1]] is an oracle/history sanity baseline,
# not a fair memoryless competitor.  It should reconstruct y almost
# perfectly and verifies that the target/alignment are correct.
# ============================================================

start = max(task_data.washout, 1)
indices = np.arange(start, n_symbols)

y_eval = y[indices]
X_pt = X[indices]
X_current = x[indices].reshape(-1, 1)
X_history2 = np.column_stack([
    x[indices],
    x[indices - 1],
])

n_train = int(task_data.train_ratio * len(indices))

y_train = y_eval[:n_train]
y_test = y_eval[n_train:]

print("\n=== Common evaluation set ===")
print(f"Valid sample range : [{start}, {n_symbols - 1}]")
print(f"Samples            : {len(y_eval)}")
print(f"Train samples      : {len(y_train)}")
print(f"Test samples       : {len(y_test)}")
print(f"Ridge              : {RIDGE:.1e}")


# ============================================================
# 1) Memoryless baseline: x[n] -> y[n]
# ============================================================

result_current = evaluate_ridge(
    X_current[:n_train],
    X_current[n_train:],
    y_train,
    y_test,
    ridge=RIDGE,
)


# ============================================================
# 2) Explicit one-step history baseline
#    [x[n], x[n-1]] -> y[n]
# ============================================================

result_history2 = evaluate_ridge(
    X_history2[:n_train],
    X_history2[n_train:],
    y_train,
    y_test,
    ridge=RIDGE,
)


# ============================================================
# 3) PT Single temporal state -> y[n]
# ============================================================

result_pt = evaluate_ridge(
    X_pt[:n_train],
    X_pt[n_train:],
    y_train,
    y_test,
    ridge=RIDGE,
)


# ============================================================
# Results
# ============================================================

print("\n" + "=" * 60)
print("HENON OBSERVER BENCHMARK")
print("=" * 60)
print("Task               : x[n] -> y[n]")
print("Reservoir          : PT Single")
print(f"alpha              : {ALPHA}")
print(f"T_symbol / Tc      : {N_CYCLES_PER_SYMBOL}")
print(f"I_scale            : {I_SCALE:.3e} A")
print(f"K                  : {K}")

print_result("Current observable x[n]", X_current.shape[1], result_current)
print_result("2-tap history [x[n], x[n-1]]", X_history2.shape[1], result_history2)
print_result("PT Single reservoir", X_pt.shape[1], result_pt)

print("\n=== Test summary ===")
print(f"{'Method':<34} {'Dim':>5} {'NRMSE':>12} {'Pearson r':>12}")
print("-" * 68)
print(
    f"{'x[n] only':<34} {X_current.shape[1]:>5d} "
    f"{result_current['test_nrmse']:>12.6f} "
    f"{result_current['test_r']:>12.6f}"
)
print(
    f"{'2-tap history':<34} {X_history2.shape[1]:>5d} "
    f"{result_history2['test_nrmse']:>12.6f} "
    f"{result_history2['test_r']:>12.6f}"
)
print(
    f"{'PT Single':<34} {X_pt.shape[1]:>5d} "
    f"{result_pt['test_nrmse']:>12.6f} "
    f"{result_pt['test_r']:>12.6f}"
)
