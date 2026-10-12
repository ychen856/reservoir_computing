import numpy as np

from src.circuits.pt_single import PTSingleReservoir
from src.tasks.pam4 import apply_linear_channel, generate_pam4, quantize_pam4_class
from src.utils.task_encoding import sample_hold_input
from src.utils.task_features import extract_temporal_features
from src.readout.ridge import train_ridge_regression, predict_ridge_regression, nrmse

# ============================================================
# Configuration
# ============================================================

N_SYMBOLS = 3000
WASHOUT = 200
TRAIN_RATIO = 0.8

FC = 48e6
Tc = 1.0 / FC

N_CYCLES_PER_SYMBOL = 1 / 40
T_SYMBOL = N_CYCLES_PER_SYMBOL * Tc

I_SCALE = 1e-2
K = 8
POINTS_PER_SYMBOL = 16

RIDGE = 1e-6
ALPHA = 2.78
SEED = 42

# Nonlinear channel / noise controls
ADD_AWGN = False
SNR_DB = 40.0
NOISE_SEED = 123

NL_QUAD = 0.036
NL_CUBIC = -0.011

# Linear channel uses n-2 ... n+2, so exclude the final two edge samples
# from ALL three methods for a strictly fair comparison.
CHANNEL_RADIUS = 2


# ============================================================
# Helpers
# ============================================================

def normalize_train_test(X_train, X_test):
    mu = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-12] = 1.0
    return (X_train - mu) / std, (X_test - mu) / std


def build_centered_window(sequence, radius=2):
    """Return centered windows [q[n-r], ..., q[n], ..., q[n+r]]."""
    sequence = np.asarray(sequence, dtype=float)
    n_valid = len(sequence) - 2 * radius
    Xw = np.zeros((n_valid, 2 * radius + 1))

    for i, n in enumerate(range(radius, len(sequence) - radius)):
        Xw[i] = sequence[n - radius:n + radius + 1]

    return Xw


def evaluate_ridge(X_train, X_test, y_train, y_test, ridge=1e-6):
    X_train_n, X_test_n = normalize_train_test(X_train, X_test)

    W = train_ridge_regression(X_train_n, y_train, ridge=ridge)
    y_train_pred = predict_ridge_regression(X_train_n, W)
    y_test_pred = predict_ridge_regression(X_test_n, W)

    y_train_hat = quantize_pam4_class(y_train_pred)
    y_test_hat = quantize_pam4_class(y_test_pred)

    train_acc = np.mean(y_train_hat == y_train)
    test_acc = np.mean(y_test_hat == y_test)

    return {
        "train_nrmse": nrmse(y_train, y_train_pred),
        "test_nrmse": nrmse(y_test, y_test_pred),
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_ser": 1.0 - train_acc,
        "test_ser": 1.0 - test_acc,
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
# Generate PAM4 + linear ISI + nonlinear distortion (+ optional AWGN)
# ============================================================

symbols, y = generate_pam4(
    n_symbols=N_SYMBOLS,
    seed=SEED,
)

q_raw = apply_linear_channel(symbols)

# Nonlinear channel distortion:
# d[n] = q[n] + 0.036 q[n]^2 - 0.011 q[n]^3
d_raw = q_raw + NL_QUAD * q_raw**2 + NL_CUBIC * q_raw**3

# Optional AWGN is added BEFORE the final normalization, so SNR is defined
# relative to the distorted received waveform.
u_raw = d_raw.copy()
measured_snr_db = None
noise_power = 0.0

if ADD_AWGN:
    rng_noise = np.random.default_rng(NOISE_SEED)

    # Exclude the channel edge samples from the power estimate.
    valid_power = d_raw[CHANNEL_RADIUS:-CHANNEL_RADIUS]
    signal_power = np.mean(valid_power**2)
    noise_power = signal_power / (10.0 ** (SNR_DB / 10.0))

    noise = rng_noise.normal(
        loc=0.0,
        scale=np.sqrt(noise_power),
        size=len(d_raw),
    )
    u_raw = d_raw + noise

    measured_noise_power = np.mean(noise[CHANNEL_RADIUS:-CHANNEL_RADIUS]**2)
    measured_snr_db = 10.0 * np.log10(signal_power / measured_noise_power)

# Normalize the final received waveform to unit peak before current injection.
# All baselines use this SAME normalized waveform.
u_scale = np.max(np.abs(u_raw))
u = u_raw / u_scale


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
print(f"received scale   = {u_scale:.6f}")
print(f"nonlinear model  = q + {NL_QUAD} q^2 {NL_CUBIC:+.3f} q^3")
print(f"AWGN             = {ADD_AWGN}")
if ADD_AWGN:
    print(f"requested SNR    = {SNR_DB:.2f} dB")
    print(f"measured SNR     = {measured_snr_db:.2f} dB")


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

# Build [u[n-2], u[n-1], u[n], u[n+1], u[n+2]] directly at the same n.
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
# 1) Current received-sample baseline: u[n] only
# ============================================================

result_current = evaluate_ridge(
    X_current[:n_train],
    X_current[n_train:],
    y_train,
    y_test,
    ridge=RIDGE,
)


# ============================================================
# 2) 5-tap centered received-waveform equalizer baseline
#    [u[n-2], u[n-1], u[n], u[n+1], u[n+2]] -> class[n]
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
print("PAM4 NONLINEAR-CHANNEL BENCHMARK")
print("=" * 60)
print(f"Reservoir          : PT Single")
print(f"alpha              : {ALPHA}")
print(f"T_symbol / Tc      : {N_CYCLES_PER_SYMBOL}")
print(f"I_scale            : {I_SCALE:.3e} A")
print(f"K                  : {K}")
print(f"nonlinear channel  : ON")
print(f"AWGN               : {'ON' if ADD_AWGN else 'OFF'}")
if ADD_AWGN:
    print(f"SNR                 : {SNR_DB:.2f} dB")

print_result("Current received sample u[n]", X_current.shape[1], result_current)
print_result("5-tap linear equalizer", X_5tap.shape[1], result_5tap)
print_result("PT Single reservoir", X_pt.shape[1], result_pt)

print("\n=== Test summary ===")
print(f"{'Method':<28} {'Dim':>5} {'NRMSE':>10} {'Accuracy':>12} {'SER':>12}")
print("-" * 72)
print(
    f"{'u[n] only':<28} {X_current.shape[1]:>5d} "
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
