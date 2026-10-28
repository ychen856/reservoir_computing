"""Run: python3 -m src.run_input_reconstruction"""
from pathlib import Path
import numpy as np

from src.circuits.pt_single import PTSingleReservoir
from src.tasks.benchmarks.sine_cube import generate_sine_cube
from src.tasks.benchmarks.narma10 import generate_narma10
from src.simulation.runner import simulate_continuous_task
from src.diagnostics.input_reconstruction import evaluate_input_reconstruction, save_diagnostic

FC = 48e6
T_SYMBOL = 0.5 / FC
I_SCALE = 1e-4
K = 8
POINTS_PER_SYMBOL = 16
ETA = 1.0
ALPHA = 8.33
MAX_DELAY = 20
RIDGE = 1e-6
OUTPUT_ROOT = Path('results/input_reconstruction')


def run_one(name, task_data):
    print(f'\n=== {name}: input reconstruction ===')
    reservoir = PTSingleReservoir(eta=ETA, alpha=ALPHA)
    result = simulate_continuous_task(
        reservoir=reservoir,
        task_data=task_data,
        symbol_period=T_SYMBOL,
        points_per_symbol=POINTS_PER_SYMBOL,
        k=K,
        amplitude_scale=I_SCALE,
        encoding='zoh',
        solver='exact_zoh',
    )
    X = result.X
    u = np.asarray(task_data.u)
    print('X shape:', X.shape)
    print('Input range:', float(u.min()), float(u.max()))
    print('V1 range:', float(result.solutions.y[0].min()), float(result.solutions.y[0].max()))
    print('V2 range:', float(result.solutions.y[1].min()), float(result.solutions.y[1].max()))

    rows, meta = evaluate_input_reconstruction(
        X, u, washout=task_data.washout,
        train_ratio=task_data.train_ratio,
        max_delay=MAX_DELAY, ridge=RIDGE,
    )
    save_diagnostic(rows, OUTPUT_ROOT / name, f'{name}: input reconstruction')
    print('Split:', meta)
    print(f"{'delay':>5} {'PT train':>10} {'PT test':>10} {'Input test':>12}")
    for r in rows:
        print(f"{r['delay']:5d} {r['reservoir_train_r2']:10.5f} "
              f"{r['reservoir_test_r2']:10.5f} {r['current_input_test_r2']:12.5f}")
    print('Saved to:', OUTPUT_ROOT / name)


def main():
    sine = generate_sine_cube(n_samples=2000, washout=100, train_ratio=0.8)
    narma = generate_narma10(n_samples=3000, washout=200, train_ratio=0.8, seed=42)
    run_one('sine_cube', sine)
    run_one('narma10', narma)


if __name__ == '__main__':
    main()
