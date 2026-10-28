import numpy as np

from src.circuits.pt_single import PTSingleReservoir
from src.tasks.applications.pam4_nonlinear import generate_pam4_nonlinear_task
from src.simulation.runner import simulate_continuous_task
from src.simulation.evaluation import evaluate_scalar_classification
from src.simulation.experiment import ExperimentConfig, ReadoutExperiment

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

task_data = generate_pam4_nonlinear_task(
    n_symbols=N_SYMBOLS,
    seed=SEED,
    washout=WASHOUT,
    train_ratio=TRAIN_RATIO,
    quadratic=NL_QUAD,
    cubic=NL_CUBIC,
    add_awgn=ADD_AWGN,
    snr_db=SNR_DB,
    noise_seed=NOISE_SEED,
    channel_radius=CHANNEL_RADIUS,
)

u = task_data.u
y = task_data.y

n_symbols = len(u)

u_scale = task_data.metadata["input_scale"]
measured_snr_db = task_data.metadata["measured_snr_db"]
channel_radius = task_data.metadata["channel_radius"]



# ============================================================
# Physical simulation time
# ============================================================

dt = T_SYMBOL / POINTS_PER_SYMBOL
t_end = n_symbols * T_SYMBOL



# ============================================================
# Reservoir
# ============================================================

reservoir = PTSingleReservoir(
    eta=1.0,
    alpha=ALPHA,
)

sim_result = simulate_continuous_task(
    reservoir=reservoir,
    task_data=task_data,
    symbol_period=T_SYMBOL,
    points_per_symbol=POINTS_PER_SYMBOL,
    k=K,
    amplitude_scale=I_SCALE,
    encoding="zoh",
)

X = sim_result.X
y = sim_result.y

sol = sim_result.solutions
t_eval = sim_result.t_eval


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
# Common valid symbol range for all three methods
# ============================================================
#
# apply_linear_channel() uses a centered n-2...n+2 channel and leaves
# edge values undefined/zero.  Washout already removes the first edge;
# remove the final two symbols so PT, q[n], and 5-tap are evaluated on
# exactly the same labels and chronological train/test split.
# ============================================================

start = max(task_data.washout, channel_radius)
stop = n_symbols - channel_radius       # Python-exclusive
indices = np.arange(start, stop)



# ============================================================
# 1) Current received-sample baseline: u[n] only
# ============================================================

X_current = u.reshape(-1, 1)

result_current = evaluate_scalar_classification(
    X=X_current,
    y=y,
    washout=task_data.washout,
    train_ratio=task_data.train_ratio,
    ridge=RIDGE,
    normalize=True,
    valid_start=start,
    valid_stop=stop,
)



# ============================================================
# 2) 5-tap centered received-waveform equalizer baseline
#    [u[n-2], u[n-1], u[n], u[n+1], u[n+2]] -> class[n]
# ============================================================


X_5tap = np.zeros(
    (n_symbols, 5),
    dtype=float,
)

X_5tap[2:-2] = np.column_stack([
    u[:-4],
    u[1:-3],
    u[2:-2],
    u[3:-1],
    u[4:],
])

result_5tap = evaluate_scalar_classification(
    X=X_5tap,
    y=y,
    washout=task_data.washout,
    train_ratio=task_data.train_ratio,
    ridge=RIDGE,
    normalize=True,
    valid_start=start,
    valid_stop=stop,
)

# ============================================================
# 3) PT Single reservoir: 16D temporal state
# ============================================================


result_pt = evaluate_scalar_classification(
    X=X,
    y=y,
    washout=task_data.washout,
    train_ratio=task_data.train_ratio,
    ridge=RIDGE,
    normalize=True,
    valid_start=start,
    valid_stop=stop,
)


# ============================================================
# Results
# ============================================================

def print_result(name, result):

    print(f"\n=== {name} ===")
    print(f"Feature dim    : {result.feature_dim}")

    print(f"Train NRMSE    : {result.train_nrmse:.6f}")
    print(f"Test NRMSE     : {result.test_nrmse:.6f}")

    print(f"Train accuracy : {result.train_accuracy:.4%}")
    print(f"Test accuracy  : {result.test_accuracy:.4%}")

    print(f"Train SER      : {result.train_ser:.4e}")
    print(f"Test SER       : {result.test_ser:.4e}")


print_result(
    "Current received sample u[n]",
    result_current,
)

print_result(
    "5-tap linear equalizer",
    result_5tap,
)

print_result(
    "PT Single reservoir",
    result_pt,
)

print("\n=== Test summary ===")

print(
    f"{'Method':<28} "
    f"{'Dim':>5} "
    f"{'NRMSE':>10} "
    f"{'Accuracy':>12} "
    f"{'SER':>12}"
)

print("-" * 72)

for name, result in [
    ("u[n] only", result_current),
    ("5-tap equalizer", result_5tap),
    ("PT Single", result_pt),
]:

    print(
        f"{name:<28} "
        f"{result.feature_dim:>5d} "
        f"{result.test_nrmse:>10.6f} "
        f"{result.test_accuracy:>11.2%} "
        f"{result.test_ser:>12.4e}"
    )








# ============================================================
# Common Experiment Configuration
# ============================================================

config = ExperimentConfig(
    symbol_period=T_SYMBOL,
    points_per_symbol=POINTS_PER_SYMBOL,
    k=K,
    amplitude_scale=I_SCALE,
    encoding="zoh",

    ridge=RIDGE,
    normalize=True,

    evaluation_type="scalar_classification",

    valid_start=WASHOUT,
    valid_stop=N_SYMBOLS - CHANNEL_RADIUS,

    n_classes=4,
)


experiment = ReadoutExperiment(
    reservoir=reservoir,
    task_data=task_data,
    config=config,
    reservoir_name="PT Single",
)

# ============================================================
# Reservoir simulation
# ============================================================

#simulation = experiment.simulate()
simulation = sim_result

# ============================================================
# Readout evaluation
# ============================================================

result = experiment.evaluate(
    simulation=simulation,
    k=K,
    ridge=RIDGE,
)

ev = result.evaluation

print("\n=== PAM4 Nonlinear: Common Experiment ===")

print(f"Feature dimension: {ev.feature_dim}")
print(f"Train samples    : {ev.n_train}")
print(f"Test samples     : {ev.n_test}")

print(f"Train NRMSE      : {ev.train_nrmse:.6f}")
print(f"Test NRMSE       : {ev.test_nrmse:.6f}")

print(f"Train accuracy   : {ev.train_accuracy:.4%}")
print(f"Test accuracy    : {ev.test_accuracy:.4%}")

print(f"Train SER        : {ev.train_ser:.6f}")
print(f"Test SER         : {ev.test_ser:.6f}")

print("\nConfusion matrix:")
print(ev.confusion_matrix)


# ============================================================
# Regression check
# ============================================================


np.testing.assert_allclose(
    [
        ev.train_nrmse,
        ev.test_nrmse,
        ev.train_accuracy,
        ev.test_accuracy,
    ],
    [
        result_pt.train_nrmse,
        result_pt.test_nrmse,
        result_pt.train_accuracy,
        result_pt.test_accuracy,
    ],
    rtol=1e-10,
    atol=1e-10,
)

assert ev.feature_dim == 16
assert ev.n_train == 2238
assert ev.n_test == 560

print("\nRegression check: PASS")


# ============================================================
# K sweep
# ============================================================

K_LIST = [4, 8, 16]

for K_eval in K_LIST:

    result = experiment.evaluate(
        simulation=simulation,
        k=K_eval,
        ridge=RIDGE,
    )

    ev = result.evaluation

    assert ev.feature_dim == 2 * K_eval

    print(
        f"K={K_eval:2d} | "
        f"dim={ev.feature_dim:2d} | "
        f"Train acc={ev.train_accuracy:.4%} | "
        f"Test acc={ev.test_accuracy:.4%} | "
        f"Test NRMSE={ev.test_nrmse:.6f}"
    )


