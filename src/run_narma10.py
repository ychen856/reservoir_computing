import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from src.circuits.pt_single import PTSingleReservoir
from src.tasks.benchmarks.narma10 import generate_narma10
from src.utils.encoding import sample_hold_input, carrier_encoded_input, bipolar_input, masked_zoh_input
from src.utils.features import extract_temporal_features
from src.readout.ridge import train_ridge_regression, predict_ridge_regression
from src.readout.metrics import nrmse
from src.readout.preprocessing import fit_standardizer, transform_standardizer

def evaluate_regression(
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

    mu, std = fit_standardizer(X_train)

    X_train_n = transform_standardizer(
        X_train, mu, std
    )

    X_test_n = transform_standardizer(
        X_test, mu, std
    )

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

    return {
        "train_nrmse": nrmse(y_train, y_train_pred),
        "test_nrmse": nrmse(y_test, y_test_pred),
        "W": W,
    }

def build_input_history_features(u, order=10):

    X = np.zeros((len(u), order))

    for n in range(len(u)):
        for k in range(order):

            idx = n - k

            if idx >= 0:
                X[n, k] = u[idx]

    return X


# ============================================================
# Configuration
# ============================================================

N_SAMPLES = 3000

FC = 48e6

Tc = 1.0 / FC

N_CYCLES_PER_SYMBOL = 2.0 #1/2

T_SYMBOL = N_CYCLES_PER_SYMBOL * Tc

I_SCALE = 1e-3

K = 8

POINTS_PER_SYMBOL = 16

READOUT_DELAY = 0

# ============================================================
# Generate task
# ============================================================


'''u, y = generate_narma10(
    n_samples=N_SAMPLES,
    seed=42
)'''
task_data = generate_narma10(
    n_samples=N_SAMPLES,
    washout=200,
    train_ratio=0.8,
    seed=42
)
u = task_data.u
y = task_data.y
n_symbols = len(u)


# ============================================================
# Encode input
# ============================================================

'''input_func = bipolar_input(
    u,
    symbol_period=T_SYMBOL,
    amplitude_scale=I_SCALE,
    mode='peak_matched'
)'''
'''input_func = sample_hold_input(
    u,
    symbol_period=T_SYMBOL,
    amplitude_scale=I_SCALE
)'''
#MASK = [1, -1]
#MASK = [-1, 1]
MASK = [1, -1, 1, -1]
#MASK = [1, 1, -1, -1]
M = len(MASK)

input_func = masked_zoh_input(
    sequence=u,
    symbol_period=T_SYMBOL,
    amplitude_scale=I_SCALE,
    mask=MASK,
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
    alpha=8.33 #0.0 #2.78 #8.33
)


'''sol = reservoir.simulate(
    input_func=input_func,
    t_span=(0.0, t_end),
    t_eval=t_eval,
    solver='exact_zoh',
    symbol_period=T_SYMBOL,
)'''
sol = reservoir.simulate(
    input_func=input_func,
    t_span=(0.0, t_end),
    t_eval=t_eval,
    solver="exact_zoh",
    symbol_period=T_SYMBOL / M,
)


# ============================================================
# Feature extraction
# ============================================================

X = extract_temporal_features(
    sol,
    n_symbols=n_symbols,
    symbol_period=T_SYMBOL,
    k=K,
    delay_symbols=READOUT_DELAY
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

valid_indices = np.arange(
    task_data.washout,
    len(task_data.u),
)


# ============================================================
# Sequential train / test split
# ============================================================

TRAIN_RATIO = task_data.train_ratio

n_train = int(TRAIN_RATIO * len(valid_indices))


train_indices = valid_indices[:n_train]
test_indices = valid_indices[n_train:]



X_hist = build_input_history_features(
    task_data.u,
    order=10
)

poly = PolynomialFeatures(
    degree=2,
    include_bias=False
)


X_res_poly = poly.fit_transform(X)
X_hist_poly = poly.fit_transform(X_hist)

results = {
    "PT reservoir": evaluate_regression(
        X, y, train_indices, test_indices
    ),

    "Current input": evaluate_regression(
        u.reshape(-1, 1),
        y,
        train_indices,
        test_indices,
    ),

    "10-tap history": evaluate_regression(
        X_hist,
        y,
        train_indices,
        test_indices,
    ),

    "10-tap polynomial": evaluate_regression(
        X_hist_poly,
        y,
        train_indices,
        test_indices,
    ),

    "PT polynomial": evaluate_regression(
        X_res_poly,
        y,
        train_indices,
        test_indices,
    ),
}


for name, result in results.items():
    print(
        f"{name:25s} "
        f"Train={result['train_nrmse']:.6f} "
        f"Test={result['test_nrmse']:.6f}"
    )

