import numpy as np

from src.circuits.pt_single import PTSingleReservoir
from src.tasks.applications.pam4 import apply_linear_channel, generate_pam4, quantize_pam4_class
from src.utils.encoding import sample_hold_input
from src.utils.features import extract_temporal_features
from src.readout.ridge import train_ridge_regression, predict_ridge_regression
from src.readout.metrics import nrmse
from src.readout.preprocessing import fit_standardizer, transform_standardizer
from src.readout.metrics import accuracy, symbol_error_rate

# ============================================================
# Configuration
# ============================================================

N_SYMBOLS = 3000
WASHOUT = 200
TRAIN_RATIO = 0.8

FC = 48e6
Tc = 1.0 / FC

N_CYCLES_PER_SYMBOL = 1 / 320
T_SYMBOL = N_CYCLES_PER_SYMBOL * Tc

I_SCALE = 1e-2
K = 8
POINTS_PER_SYMBOL = 16

RIDGE = 1e-6
ALPHA = 2.78
SEED = 42

# Linear channel uses n-2 ... n+2, so exclude the final two edge samples
# from ALL three methods for a strictly fair comparison.
CHANNEL_RADIUS = 2


# ============================================================
# Helpers
# ============================================================

def build_centered_window(sequence, radius=2):
    """Return centered windows [q[n-r], ..., q[n], ..., q[n+r]]."""
    sequence = np.asarray(sequence, dtype=float)
    n_valid = len(sequence) - 2 * radius
    Xw = np.zeros((n_valid, 2 * radius + 1))

    for i, n in enumerate(range(radius, len(sequence) - radius)):
        Xw[i] = sequence[n - radius:n + radius + 1]

    return Xw


def evaluate_ridge(X_train, X_test, y_train, y_test, ridge=1e-6):
    mu, std = fit_standardizer(X_train)
    
    X_train_n = transform_standardizer(X_train, mu, std)
    X_test_n = transform_standardizer(X_test, mu, std)

    W = train_ridge_regression(X_train_n, y_train, ridge=ridge)
    y_train_pred = predict_ridge_regression(X_train_n, W)
    y_test_pred = predict_ridge_regression(X_test_n, W)

    y_train_hat = quantize_pam4_class(y_train_pred)
    y_test_hat = quantize_pam4_class(y_test_pred)


    train_acc = accuracy(y_train, y_train_hat)
    test_acc = accuracy(y_test, y_test_hat)

    train_ser = symbol_error_rate(y_train, y_train_hat)
    test_ser = symbol_error_rate(y_test, y_test_hat)

    return {
        "train_nrmse": nrmse(y_train, y_train_pred),
        "test_nrmse": nrmse(y_test, y_test_pred),
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_ser": train_ser,
        "test_ser": test_ser,
    }


def print_result(name, feature_dim, result):
    print(f"\n=== {name} ===")
    print(f"Feature dim    : {feature_dim}")
    print(f"Train NRMSE    : {result['train_nrmse']:.6f}")
    print(f"Test NRMSE     : {result['test_nrmse']:.6f}")
    print(f"Train accuracy : {result['train_acc']:.4%}")
    print(f"Test accuracy  : {result['test_acc']:.4%}")
    print(f"Train SER      : {result['train_ser']:.4e}")
    print(f"Test SER       : {result['test_ser']:.4e}")


# ============================================================
# Generate PAM4 + linear ISI channel
# ============================================================

symbols, y = generate_pam4(
    n_symbols=N_SYMBOLS,
    seed=SEED,
)

q_raw = apply_linear_channel(symbols)

# Normalize channel output to [-1, 1] before current injection.
u_scale = np.max(np.abs(q_raw))
u = q_raw / u_scale


# ============================================================
# Encode input
# ============================================================

input_func = sample_hold_input(
    u,
    symbol_period=T_SYMBOL,
    amplitude_scale=I_SCALE,
)


# ============================================================
# Physical simulation time
# ============================================================

dt = T_SYMBOL / POINTS_PER_SYMBOL
t_end = N_SYMBOLS * T_SYMBOL

t_eval = np.arange(0.0, t_end, dt)

print(f"Tc               = {Tc * 1e9:.4f} ns")
print(f"T_SYMBOL         = {T_SYMBOL * 1e9:.4f} ns")
print(f"dt               = {dt * 1e12:.4f} ps")
print(f"points/symbol    = {T_SYMBOL / dt:.1f}")
print(f"temporal nodes K = {K}")
print(f"t_eval points    = {len(t_eval)}")
print(f"channel scale    = {u_scale:.6f}")


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


# ============================================================
# Reservoir feature extraction
# ============================================================

X = extract_temporal_features(
    sol,
    n_symbols=N_SYMBOLS,
    symbol_period=T_SYMBOL,
    k=K,
)

print("u shape:", u.shape)
print("y shape:", y.shape)
print("X shape:", X.shape)
print("Input range:", np.min(u), np.max(u))
print("Target range:", np.min(y), np.max(y))
print("V1 range:", np.min(sol.y[0]), np.max(sol.y[0]))
print("V2 range:", np.min(sol.y[1]), np.max(sol.y[1]))


# ============================================================
# Common valid symbol range for all three methods
# ============================================================
#
# apply_linear_channel() uses a centered n-2...n+2 channel and leaves
# edge values undefined/zero.  Washout already removes the first edge;
# remove the final two symbols so PT, q[n], and 5-tap are evaluated on
# exactly the same labels and chronological train/test split.
# ============================================================

start = max(WASHOUT, CHANNEL_RADIUS)
stop = N_SYMBOLS - CHANNEL_RADIUS       # Python-exclusive
indices = np.arange(start, stop)

y_eval = y[indices]
X_pt = X[indices]
X_current = u[indices].reshape(-1, 1)

# Build [q[n-2], q[n-1], q[n], q[n+1], q[n+2]] directly at the same n.
X_5tap = np.column_stack([
    u[indices - 2],
    u[indices - 1],
    u[indices],
    u[indices + 1],
    u[indices + 2],
])

n_train = int(TRAIN_RATIO * len(indices))

y_train = y_eval[:n_train]
y_test = y_eval[n_train:]

print("\n=== Common evaluation set ===")
print(f"Valid symbol range : [{start}, {stop - 1}]")
print(f"Samples            : {len(y_eval)}")
print(f"Train samples      : {len(y_train)}")
print(f"Test samples       : {len(y_test)}")
print(f"Ridge              : {RIDGE:.1e}")


# ============================================================
# 1) Current channel sample baseline: q[n] only
# ============================================================

result_current = evaluate_ridge(
    X_current[:n_train],
    X_current[n_train:],
    y_train,
    y_test,
    ridge=RIDGE,
)


# ============================================================
# 2) 5-tap centered linear equalizer baseline
#    [q[n-2], q[n-1], q[n], q[n+1], q[n+2]] -> class[n]
# ============================================================

result_5tap = evaluate_ridge(
    X_5tap[:n_train],
    X_5tap[n_train:],
    y_train,
    y_test,
    ridge=RIDGE,
)


# ============================================================
# 3) PT Single reservoir: 16D temporal state
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
print("PAM4 LINEAR-ISI BENCHMARK")
print("=" * 60)
print(f"Reservoir          : PT Single")
print(f"alpha              : {ALPHA}")
print(f"T_symbol / Tc      : {N_CYCLES_PER_SYMBOL}")
print(f"I_scale            : {I_SCALE:.3e} A")
print(f"K                  : {K}")

print_result("Current-sample baseline q[n]", X_current.shape[1], result_current)
print_result("5-tap linear equalizer", X_5tap.shape[1], result_5tap)
print_result("PT Single reservoir", X_pt.shape[1], result_pt)

print("\n=== Test summary ===")
print(f"{'Method':<28} {'Dim':>5} {'NRMSE':>10} {'Accuracy':>12} {'SER':>12}")
print("-" * 72)
print(
    f"{'q[n] only':<28} {X_current.shape[1]:>5d} "
    f"{result_current['test_nrmse']:>10.6f} "
    f"{result_current['test_acc']:>11.2%} "
    f"{result_current['test_ser']:>12.4e}"
)
print(
    f"{'5-tap equalizer':<28} {X_5tap.shape[1]:>5d} "
    f"{result_5tap['test_nrmse']:>10.6f} "
    f"{result_5tap['test_acc']:>11.2%} "
    f"{result_5tap['test_ser']:>12.4e}"
)
print(
    f"{'PT Single':<28} {X_pt.shape[1]:>5d} "
    f"{result_pt['test_nrmse']:>10.6f} "
    f"{result_pt['test_acc']:>11.2%} "
    f"{result_pt['test_ser']:>12.4e}"
)
