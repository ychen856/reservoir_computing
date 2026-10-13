import numpy as np

from src.circuits.pt_single import PTSingleReservoir
from src.tasks.benchmarks.sine_cube import generate_sine_cube
from src.utils.encoding import sample_hold_input, carrier_encoded_input
from src.utils.features import extract_temporal_features
from src.readout.ridge import train_ridge_regression, predict_ridge_regression
from src.readout.metrics import nrmse
from src.readout.preprocessing import fit_standardizer, transform_standardizer

# ============================================================
# Configuration
# ============================================================

N_SAMPLES = 2000

FC = 48e6

Tc = 1.0 / FC

N_CYCLES_PER_SYMBOL = 1/40

T_SYMBOL = N_CYCLES_PER_SYMBOL * Tc

I_SCALE = 1e-2

K = 8

POINTS_PER_SYMBOL = 16



# ============================================================
# Generate task
# ============================================================

'''_, u, y = generate_sine_cube(
    n_samples=N_SAMPLES,
    n_periods=20
)'''

task_data = generate_sine_cube(
    n_samples=N_SAMPLES,
    washout=100,
    train_ratio=0.8
)
u = task_data.u
y = task_data.y

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

n_symbols = len(task_data.u)
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
    alpha=8.33 #0.0 #2.78 #8.33
)

sol = reservoir.simulate(
    input_func=input_func,
    t_span=(0.0, t_end),
    t_eval=t_eval
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
# Washout
# ============================================================

WASHOUT = task_data.washout

X_eval = X[WASHOUT:]
y_eval = task_data.y[WASHOUT:]


# ============================================================
# Sequential train / test split
# ============================================================

TRAIN_RATIO = task_data.train_ratio

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

print("\n=== Sine -> Cube benchmark ===")
print(f"Reservoir     : PT Single")
print(f"Samples       : {len(y)}")
print(f"Washout       : {WASHOUT}")
print(f"Feature dim   : {X.shape[1]}")
print(f"Train samples : {len(y_train)}")
print(f"Test samples  : {len(y_test)}")
print(f"Ridge         : {RIDGE:.1e}")
print(f"Train NRMSE   : {train_nrmse:.6f}")
print(f"Test NRMSE    : {test_nrmse:.6f}")



# ============================================================
# Input-only baseline
# ============================================================

X_input = task_data.u[WASHOUT:].reshape(-1, 1)

X_input_train = X_input[:n_train]
X_input_test = X_input[n_train:]

y_input_train = y_eval[:n_train]
y_input_test = y_eval[n_train:]

W_input = train_ridge_regression(
    X_input_train,
    y_input_train,
    ridge=RIDGE,
)

y_input_train_pred = predict_ridge_regression(
    X_input_train,
    W_input,
)

y_input_test_pred = predict_ridge_regression(
    X_input_test,
    W_input,
)

print("\n=== Input-only baseline ===")
print(
    "Train NRMSE:",
    nrmse(y_input_train, y_input_train_pred)
)
print(
    "Test NRMSE :",
    nrmse(y_input_test, y_input_test_pred)
)