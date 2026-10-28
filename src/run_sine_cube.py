import numpy as np

from src.circuits.pt_single import PTSingleReservoir
from src.tasks.benchmarks.sine_cube import generate_sine_cube
from src.utils.encoding import sample_hold_input, carrier_encoded_input
from src.utils.features import extract_temporal_features
from src.readout.ridge import train_ridge_regression, predict_ridge_regression
from src.readout.metrics import nrmse
from src.readout.preprocessing import fit_standardizer, transform_standardizer
from src.simulation.runner import simulate_continuous_task
from src.simulation.evaluation import evaluate_continuous_regression

# ============================================================
# Configuration
# ============================================================

N_SAMPLES = 2000

FC = 48e6

Tc = 1.0 / FC

N_CYCLES_PER_SYMBOL = 3.05 #1/2

T_SYMBOL = N_CYCLES_PER_SYMBOL * Tc

I_SCALE = 1e-4

K = 8

POINTS_PER_SYMBOL = 16

task_data = generate_sine_cube(
    n_samples=N_SAMPLES,
    washout=100,
    train_ratio=0.8
)

# ============================================================
# Common simulation
# ============================================================

reservoir = PTSingleReservoir(
    eta=1.0,
    alpha=8.33,
)

result = simulate_continuous_task(
    reservoir=reservoir,
    task_data=task_data,
    symbol_period=T_SYMBOL,
    points_per_symbol=POINTS_PER_SYMBOL,
    k=K,
    amplitude_scale=I_SCALE,
    encoding="zoh",
    solver='exact_zoh',
)

X = result.X
y = result.y

sol = result.solutions
t_eval = result.t_eval




print("Target range:")
print(np.min(y), np.max(y))

print("V1 range:")
print(np.min(sol.y[0]), np.max(sol.y[0]))

print("V2 range:")
print(np.min(sol.y[1]), np.max(sol.y[1]))




# ============================================================
# Common regression evaluation
# ============================================================

RIDGE = 1e-6

print("\n=== Debug ===")
print("Ts/Tc =", T_SYMBOL / Tc)
print("I_SCALE =", I_SCALE)
print("alpha =", reservoir.alpha)
print("X shape =", result.X.shape)
print("X min/max =", np.min(result.X), np.max(result.X))
print("X std =", np.std(result.X, axis=0))
print("y shape =", result.y.shape)
print("y min/max =", np.min(result.y), np.max(result.y))

eval_result = evaluate_continuous_regression(
    X=result.X,
    y=result.y,
    washout=task_data.washout,
    train_ratio=task_data.train_ratio,
    ridge=RIDGE,
    normalize=True,
)

train_nrmse = eval_result.train_nrmse
test_nrmse = eval_result.test_nrmse

print("\n=== Sine -> Cube benchmark ===")
print(f"Reservoir     : PT Single")
print(f"Samples       : {len(result.y)}")
print(f"Washout       : {task_data.washout}")
print(f"Feature dim   : {eval_result.feature_dim}")
print(f"Train samples : {eval_result.n_train}")
print(f"Test samples  : {eval_result.n_test}")
print(f"Ridge         : {eval_result.ridge:.1e}")
print(f"Train NRMSE   : {train_nrmse:.6f}")
print(f"Test NRMSE    : {test_nrmse:.6f}")



# ============================================================
# Input-only baseline
# ============================================================

baseline_result = evaluate_continuous_regression(
    X=task_data.u.reshape(-1, 1),
    y=task_data.y,
    washout=task_data.washout,
    train_ratio=task_data.train_ratio,
    ridge=RIDGE,
    normalize=False,
)

print("\n=== Input-only baseline ===")
print(f"Train NRMSE: {baseline_result.train_nrmse:.6f}")
print(f"Test NRMSE : {baseline_result.test_nrmse:.6f}")