import numpy as np

from src.circuits.pt_single import PTSingleReservoir
from src.tasks.narma10 import generate_narma10
from src.utils.task_encoding import sample_hold_input
from src.utils.task_features import normalize_train_test
from src.readout.ridge import train_ridge_regression, predict_ridge_regression, nrmse


def extract_temporal_features_delayed(
    sol,
    n_symbols,
    symbol_period,
    k=8,
    delay_symbols=0,
):
    """
    Extract K temporal nodes from V1/V2.

    X[n] is logically associated with input u[n], while the physical
    sampling window is shifted by delay_symbols symbol intervals.
    delay_symbols=0: sample during symbol n
    delay_symbols=1: sample during symbol n+1
    """
    n_valid = n_symbols - delay_symbols
    X = np.zeros((n_valid, 2 * k))

    for n in range(n_valid):
        t_start = (n + delay_symbols) * symbol_period
        t_stop = (n + delay_symbols + 1) * symbol_period
        sample_times = np.linspace(t_start, t_stop, k + 2)[1:-1]

        X[n, 0::2] = np.interp(sample_times, sol.t, sol.y[0])
        X[n, 1::2] = np.interp(sample_times, sol.t, sol.y[1])

    return X


# ============================================================
# Configuration
# ============================================================

N_SAMPLES = 3000

FC = 48e6

Tc = 1.0 / FC

N_CYCLES_PER_SYMBOL = 1/2

T_SYMBOL = N_CYCLES_PER_SYMBOL * Tc

I_SCALE = 1e-2

K = 8

POINTS_PER_SYMBOL = 16

READOUT_DELAY = 1

# ============================================================
# Generate task
# ============================================================


u, y = generate_narma10(
    n_samples=N_SAMPLES,
    seed=42
)



# ============================================================
# Encode input
# ============================================================

input_func = sample_hold_input(
    u,
    symbol_period=T_SYMBOL,
    amplitude_scale=I_SCALE
)

# ============================================================
# Physical simulation time
# ============================================================

dt = T_SYMBOL / POINTS_PER_SYMBOL

t_end = N_SAMPLES * T_SYMBOL

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
    alpha=2.78 #0.0 #2.78 #8.33
)

sol = reservoir.simulate(
    input_func=input_func,
    t_span=(0.0, t_end),
    t_eval=t_eval
)


# ============================================================
# Feature extraction
# ============================================================

X = extract_temporal_features_delayed(
    sol,
    n_symbols=N_SAMPLES,
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

WASHOUT = 200

'''X_eval = X[WASHOUT:-1]
y_eval = y[WASHOUT + 1:]'''

X_eval = X[WASHOUT:]

# X[n] remains logically associated with u[n], even when its physical
# readout window is delayed.  NARMA target alignment therefore stays
# X[n] -> y[n+1].
y_start = WASHOUT + 1
n_valid_eval = min(len(X_eval), len(y) - y_start)
X_eval = X_eval[:n_valid_eval]
y_eval = y[y_start:y_start + n_valid_eval]

print(f"Readout delay    = {READOUT_DELAY} symbol(s)")
print(f"X_eval shape     = {X_eval.shape}")
print(f"y_eval shape     = {y_eval.shape}")


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

X_train_n, X_test_n = normalize_train_test(
    X_train,
    X_test
)

'''mu = X_train.mean(axis=0)
std = X_train.std(axis=0)

std[std < 1e-12] = 1.0

X_train_n = (X_train - mu) / std
X_test_n = (X_test - mu) / std'''


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

for d in range(11):
    # X[n] is logically tied to u[n], irrespective of READOUT_DELAY.
    logical_start = max(WASHOUT, d)
    logical_stop = len(X)

    X_now = X[logical_start:logical_stop]
    u_delayed = u[logical_start - d:logical_stop - d]

    corr = []
    for j in range(X_now.shape[1]):
        c = np.corrcoef(X_now[:, j], u_delayed)[0, 1]
        corr.append(abs(c))

    print(
        f"delay={d:2d}: "
        f"max |corr| = {max(corr):.4f}"
    )



def memory_curve(X, u, washout, max_delay=20, ridge=1e-6):

    results = []

    for d in range(max_delay + 1):

        start = max(washout, d)

        logical_stop = len(X)
        X_d = X[start:logical_stop]
        y_d = u[start - d:logical_stop - d]

        n_train = int(0.8 * len(X_d))

        X_train = X_d[:n_train]
        X_test  = X_d[n_train:]

        y_train = y_d[:n_train]
        y_test  = y_d[n_train:]

        # normalize X using training data
        X_train_n, X_test_n = normalize_train_test(
            X_train,
            X_test
        )
        '''mu = X_train.mean(axis=0)
        std = X_train.std(axis=0)

        std[std < 1e-12] = 1.0

        X_train_n = (X_train - mu) / std
        X_test_n = (X_test - mu) / std'''

        W = train_ridge_regression(
            X_train_n,
            y_train,
            ridge=ridge
        )

        y_pred = predict_ridge_regression(
            X_test_n,
            W
        )

        ss_res = np.sum(
            (y_test - y_pred)**2
        )

        ss_tot = np.sum(
            (y_test - np.mean(y_test))**2
        )

        r2 = 1.0 - ss_res / ss_tot

        results.append(r2)

    return np.asarray(results)


memory_r2 = memory_curve(
    X,
    u,
    washout=WASHOUT,
    max_delay=20,
    ridge=RIDGE
)

print("\n=== Linear memory curve ===")

for d, r2 in enumerate(memory_r2):
    print(
        f"delay={d:2d}: "
        f"R^2={r2:.4f}"
    )


print("\n=== NARMA-10 benchmark ===")
print(f"Reservoir     : PT Single")
print(f"Samples       : {len(y)}")
print(f"Washout       : {WASHOUT}")
print(f"Feature dim   : {X.shape[1]}")
print(f"Train samples : {len(y_train)}")
print(f"Test samples  : {len(y_test)}")
print(f"Ridge         : {RIDGE:.1e}")
print(f"Train NRMSE   : {train_nrmse:.6f}")
print(f"Test NRMSE    : {test_nrmse:.6f}")


# A
X_input = u[WASHOUT:-1].reshape(-1, 1)

n_train = int(0.8 * len(X_input))

X_input_train = X_input[:n_train]
X_input_test  = X_input[n_train:]

X_input_train, X_input_test = normalize_train_test(
    X_input_train,
    X_input_test
)


y_train = y[WASHOUT + 1:][:n_train]
y_test  = y[WASHOUT + 1:][n_train:]



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

print("\n=== Current-input baseline ===")
print(
    "Train NRMSE:",
    nrmse(y_train, y_input_train_pred)
)
print(
    "Test NRMSE :",
    nrmse(y_test, y_input_test_pred)
)

# B
def build_input_history_features(u, order=10):

    X = np.zeros((len(u), order))

    for n in range(len(u)):
        for k in range(order):

            idx = n - k

            if idx >= 0:
                X[n, k] = u[idx]

    return X

X_input = build_input_history_features(u, order=10)[WASHOUT:-1]

n_train = int(0.8 * len(X_input))

X_input_train = X_input[:n_train]
X_input_test  = X_input[n_train:]

y_train = y[WASHOUT + 1:][:n_train]
y_test  = y[WASHOUT + 1:][n_train:]

X_input_train, X_input_test = normalize_train_test(
    X_input_train,
    X_input_test
)

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

print("\n=== 10-tap input-history baseline ===")
print(
    "Train NRMSE:",
    nrmse(y_train, y_input_train_pred)
)
print(
    "Test NRMSE :",
    nrmse(y_test, y_input_test_pred)
)






print("\n=== 10-tap input + 2nd-order polynomial baseline ===")

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(
    degree=2,
    include_bias=False
)

X_hist = build_input_history_features(
    u,
    order=10
)

X_poly = poly.fit_transform(X_hist)

X_input = X_poly[WASHOUT:-1]
y_input = y[WASHOUT + 1:]


n_train = int(0.8 * len(X_input))

X_input_train = X_input[:n_train]
X_input_test  = X_input[n_train:]

y_train = y[WASHOUT + 1:][:n_train]
y_test  = y[WASHOUT + 1:][n_train:]

X_input_train, X_input_test = normalize_train_test(
    X_input_train,
    X_input_test
)

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

print("\n=== 10-tap input-history baseline ===")
print(
    "Train NRMSE:",
    nrmse(y_train, y_input_train_pred)
)
print(
    "Test NRMSE :",
    nrmse(y_test, y_input_test_pred)
)


print("\n=== PT reservoir + 2nd-order polynomial readout ===")
poly = PolynomialFeatures(
    degree=2,
    include_bias=False
)

X_poly = poly.fit_transform(X)

X_input = X_poly[WASHOUT:]
y_start = WASHOUT + 1
n_valid_poly = min(len(X_input), len(y) - y_start)
X_input = X_input[:n_valid_poly]
y_input = y[y_start:y_start + n_valid_poly]

n_train = int(TRAIN_RATIO * len(X_input))

X_input_train = X_input[:n_train]
X_input_test  = X_input[n_train:]
y_train = y_input[:n_train]
y_test  = y_input[n_train:]

X_input_train, X_input_test = normalize_train_test(
            X_input_train,
            X_input_test
        )

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

print("\n=== PT reservoir + 2nd-order polynomial readout ===")
print(
    "Train NRMSE:",
    nrmse(y_train, y_input_train_pred)
)
print(
    "Test NRMSE :",
    nrmse(y_test, y_input_test_pred)
)