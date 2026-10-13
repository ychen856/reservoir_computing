import numpy as np

from src.circuits.pt_single import PTSingleReservoir
from src.tasks.applications.pam4 import apply_linear_channel, build_channel_window, quantize_pam4_class, generate_pam4_task
from src.utils.encoding import sample_hold_input, carrier_encoded_input
from src.utils.features import extract_temporal_features
from src.readout.ridge import train_ridge_regression, predict_ridge_regression
from src.readout.metrics import nrmse
from src.readout.preprocessing import fit_standardizer, transform_standardizer
from src.readout.metrics import accuracy, symbol_error_rate


def evaluate_pam4(
    name,
    X,
    y,
    train_indices,
    test_indices,
    ridge=1e-6,
):
    X_train = X[train_indices]
    X_test = X[test_indices]

    y_train = y[train_indices]
    y_test = y[test_indices]

    # Train-only feature normalization
    mu, std = fit_standardizer(X_train)

    X_train_n = transform_standardizer(
        X_train, mu, std
    )

    X_test_n = transform_standardizer(
        X_test, mu, std
    )

    # Scalar ridge regression
    W = train_ridge_regression(
        X_train_n,
        y_train,
        ridge=ridge,
    )

    y_train_pred = predict_ridge_regression(
        X_train_n, W
    )

    y_test_pred = predict_ridge_regression(
        X_test_n, W
    )

    # Convert continuous predictions into class labels
    y_train_hat = quantize_pam4_class(
        y_train_pred
    )

    y_test_hat = quantize_pam4_class(
        y_test_pred
    )

    results = {
        "name": name,
        "feature_dim": X.shape[1],
        "train_nrmse": nrmse(
            y_train, y_train_pred
        ),
        "test_nrmse": nrmse(
            y_test, y_test_pred
        ),
        "train_accuracy": accuracy(
            y_train, y_train_hat
        ),
        "test_accuracy": accuracy(
            y_test, y_test_hat
        ),
        "train_ser": symbol_error_rate(
            y_train, y_train_hat
        ),
        "test_ser": symbol_error_rate(
            y_test, y_test_hat
        ),
    }

    print(f"\n=== {name} ===")
    print(f"Feature dim   : {results['feature_dim']}")
    print(f"Train NRMSE   : {results['train_nrmse']:.6f}")
    print(f"Test NRMSE    : {results['test_nrmse']:.6f}")
    print(f"Train accuracy: {results['train_accuracy']:.4%}")
    print(f"Test accuracy : {results['test_accuracy']:.4%}")
    print(f"Train SER     : {results['train_ser']:.4e}")
    print(f"Test SER      : {results['test_ser']:.4e}")

    return results



# ============================================================
# Configuration
# ============================================================

N_SYMBOLS = 3000
TRAIN_RATIO = 0.8

FC = 48e6
Tc = 1.0 / FC

N_CYCLES_PER_SYMBOL = 1/40
T_SYMBOL = N_CYCLES_PER_SYMBOL * Tc

I_SCALE = 1e-2

K = 8
POINTS_PER_SYMBOL = 16

RIDGE = 1e-6

ALPHA = 2.78
SEED = 42


# ============================================================
# Generate task
# ============================================================

'''symbols, y = generate_pam4(
    n_symbols=N_SYMBOLS,
    seed=SEED,
)'''
task_data = generate_pam4_task(
    n_symbols=N_SYMBOLS,
    seed=SEED,
    washout=200,
    train_ratio=TRAIN_RATIO,
)

u = task_data.u
y = task_data.y

n_symbols = len(u)
channel_radius = task_data.metadata['channel_radius']

# ============================================================
# Encode input
# ============================================================

input_func = sample_hold_input(
    u,
    symbol_period=T_SYMBOL,
    amplitude_scale=I_SCALE
)
'''input_func = carrier_encoded_input(
    u,
    symbol_period=T_SYMBOL,
    carrier_frequency=FC,
    amplitude_scale=I_SCALE
)'''

# ============================================================
# Physical simulation time
# ============================================================

dt = T_SYMBOL / POINTS_PER_SYMBOL

t_end = n_symbols * T_SYMBOL

t_eval = np.arange(
    0.0,
    t_end,
    dt
)

print(f"Tc               = {Tc*1e9:.4f} ns")
print(f"T_SYMBOL         = {T_SYMBOL*1e9:.4f} ns")
print(f"dt               = {dt*1e12:.4f} ps")
print(f"points/symbol    = {T_SYMBOL/dt:.1f}")
print(f"temporal nodes K = {K}")
print(f"t_eval points    = {len(t_eval)}")


# ============================================================
# Reservoir
# ============================================================

reservoir = PTSingleReservoir(
    eta=1.0,
    alpha=ALPHA #0.0 #2.78 #8.33
)

sol = reservoir.simulate(
    input_func=input_func,
    t_span=(0.0, t_end),
    t_eval=t_eval
)

if not sol.success:
    raise RuntimeError(
        f"Reservoir simulation failed: {sol.message}"
    )

if not np.all(np.isfinite(sol.y)):
    raise RuntimeError(
        "Reservoir simulation contains NaN or Inf"
    )


# ============================================================
# Feature extraction
# ============================================================

X = extract_temporal_features(
    sol,
    n_symbols=n_symbols,
    symbol_period=T_SYMBOL,
    k=K
)

print("u shape:", u.shape)
print("y shape:", y.shape)
print("X shape:", X.shape)


print("Input range:")
print(np.min(u), np.max(u))

print("Target range:")
print(np.min(y), np.max(y))

print("V1 range:")
print(np.min(sol.y[0]), np.max(sol.y[0]))

print("V2 range:")
print(np.min(sol.y[1]), np.max(sol.y[1]))



# ============================================================
# Common train/test split
# ============================================================

valid_indices = np.arange(
    max(task_data.washout, channel_radius),
    n_symbols - channel_radius,
)

n_train = int(
    task_data.train_ratio * len(valid_indices)
)

train_indices = valid_indices[:n_train]
test_indices = valid_indices[n_train:]

assert len(X) == len(y) == len(u)

print(f"Train samples : {len(train_indices)}")
print(f"Test samples  : {len(test_indices)}")


# ============================================================
# Reservoir evaluation
# ============================================================

results_pt = evaluate_pam4(
    name="PT Single",
    X=X,
    y=y,
    train_indices=train_indices,
    test_indices=test_indices,
    ridge=RIDGE,
)


# ============================================================
# Input-only baseline
# ============================================================

X_input = u.reshape(-1, 1)

results_input = evaluate_pam4(
    name="Input only",
    X=X_input,
    y=y,
    train_indices=train_indices,
    test_indices=test_indices,
    ridge=RIDGE,
)


# ============================================================
# Channel-window baseline
# ============================================================

X_window = build_channel_window(
    u,
    radius=channel_radius,
)

results_window = evaluate_pam4(
    name="5-tap channel window",
    X=X_window,
    y=y,
    train_indices=train_indices,
    test_indices=test_indices,
    ridge=RIDGE,
)





















'''X_eval = X[WASHOUT:]
y_eval = y[WASHOUT:]


# ============================================================
# Sequential train / test split
# ============================================================

TRAIN_RATIO = 0.8

n_train = int(TRAIN_RATIO * len(X_eval))

X_train = X_eval[:n_train]
y_train = y_eval[:n_train]

X_test = X_eval[n_train:]
y_test = y_eval[n_train:]


# ============================================================
# Normalize using training set only
# ============================================================

mu, std = fit_standardizer(X_train)

X_train_n = transform_standardizer(X_train, mu, std)
X_test_n = transform_standardizer(X_test, mu, std)


# ============================================================
# Ridge regression
# ============================================================

RIDGE = 1e-6

W = train_ridge_regression(
    X_train_n,
    y_train,
    ridge=RIDGE
)

y_train_pred = predict_ridge_regression(
    X_train_n,
    W
)

y_test_pred = predict_ridge_regression(
    X_test_n,
    W
)


# ============================================================
# Evaluation
# ============================================================

train_nrmse = nrmse(
    y_train,
    y_train_pred
)

test_nrmse = nrmse(
    y_test,
    y_test_pred
)

print("\n=== PAM4 benchmark ===")
print(f"Reservoir     : PT Single")
print(f"Samples       : {len(y)}")
print(f"Washout       : {WASHOUT}")
print(f"Feature dim   : {X.shape[1]}")
print(f"Train samples : {len(y_train)}")
print(f"Test samples  : {len(y_test)}")
print(f"Ridge         : {RIDGE:.1e}")
print(f"Train NRMSE   : {train_nrmse:.6f}")
print(f"Test NRMSE    : {test_nrmse:.6f}")



X_input = u[WASHOUT:].reshape(-1, 1)

n_train = int(0.8 * len(X_input))

X_input_train = X_input[:n_train]
X_input_test  = X_input[n_train:]

y_train = y[WASHOUT:][:n_train]
y_test  = y[WASHOUT:][n_train:]

W_input = train_ridge_regression(
    X_input_train,
    y_train,
    ridge=1e-6
)

y_input_train_pred = predict_ridge_regression(
    X_input_train,
    W_input
)

y_input_test_pred = predict_ridge_regression(
    X_input_test,
    W_input
)



y_train_hat = quantize_pam4_class(
    y_train_pred
)

y_test_hat = quantize_pam4_class(
    y_test_pred
)


train_acc = accuracy(y_train, y_train_hat)
test_acc = accuracy(y_test, y_test_hat)

train_ser = symbol_error_rate(y_train, y_train_hat)
test_ser = symbol_error_rate(y_test, y_test_hat)

print(f"Train accuracy : {train_acc:.4%}")
print(f"Test accuracy  : {test_acc:.4%}")
print(f"Train SER      : {train_ser:.4e}")
print(f"Test SER       : {test_ser:.4e}")'''