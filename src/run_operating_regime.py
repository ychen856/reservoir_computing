"""PT operating-regime diagnostic; run: python3 -m src.run_operating_regime --task both

Uses the existing circuit, ZOH runner, task generators, and regression evaluator.
Parameter selection must use a validation set, not the final test set.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from src.circuits.pt_single import PTSingleReservoir
from src.simulation.runner import simulate_continuous_task
from src.simulation.evaluation import evaluate_continuous_regression
from src.tasks.benchmarks.sine_cube import generate_sine_cube
from src.tasks.benchmarks.narma10 import generate_narma10

FC = 48e6
ETA = 1.0
ALPHA = 8.33
RIDGE = 1e-6
POINTS_PER_SYMBOL = 16


def current_r2(X, u, washout, train_ratio):
    """Chronological split; standardization and ridge fitted on training only."""
    X = np.asarray(X, dtype=float)[washout:]
    u = np.asarray(u, dtype=float)[washout:]
    cut = int(len(u) * train_ratio)
    scaler = StandardScaler().fit(X[:cut])
    model = Ridge(alpha=RIDGE).fit(scaler.transform(X[:cut]), u[:cut])
    return float(r2_score(u[cut:], model.predict(scaler.transform(X[cut:]))))


def one_run(task_name, task, scale, cycles, k, outdir):
    ts = cycles / FC
    reservoir = PTSingleReservoir(eta=ETA, alpha=ALPHA)
    result = simulate_continuous_task(
        reservoir=reservoir, task_data=task,
        symbol_period=ts, points_per_symbol=POINTS_PER_SYMBOL,
        k=k, amplitude_scale=scale, encoding="zoh", solver="exact_zoh",
    )
    voltages = np.asarray(result.solutions.y[:2], dtype=float)
    if not np.all(np.isfinite(voltages)):
        raise RuntimeError("Non-finite voltage states")
    v1 = voltages[0]
    v2 = voltages[1]
    gain_reduction = (ALPHA * v1**2) / (1 + ALPHA * v1**2)
    reg = evaluate_continuous_regression(
        X=result.X, y=result.y, washout=task.washout,
        train_ratio=task.train_ratio, ridge=RIDGE, normalize=True,
    )
    row = {
        "task": task_name, "input_scale_A": scale,
        "cycles_per_symbol": cycles, "symbol_period_ns": ts * 1e9,
        "points_per_symbol": POINTS_PER_SYMBOL, "K": k,
        "max_abs_V1_V": float(np.max(np.abs(v1))),
        "rms_V1_V": float(np.sqrt(np.mean(v1**2))),
        "mean_V1_V": float(np.mean(v1)),
        "max_abs_V2_V": float(np.max(np.abs(v2))),
        "gain_reduction_mean": float(np.mean(gain_reduction)),
        "gain_reduction_p95": float(np.quantile(gain_reduction, .95)),
        "gain_reduction_max": float(np.max(gain_reduction)),
        "current_input_test_R2": current_r2(result.X, task.u, task.washout, task.train_ratio),
        "train_NRMSE": float(reg.train_nrmse),
        "test_NRMSE": float(reg.test_nrmse),
    }
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["sine_cube", "narma10", "both"], default="both")
    p.add_argument("--scales", type=float, nargs="+", default=[1e-5, 1e-4, 1e-3])
    p.add_argument("--cycles", type=float, nargs="+", default=[0.5, 2.0])
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--out", default="results/operating_regime")
    args = p.parse_args()
    if any(v <= 0 for v in args.scales + args.cycles) or args.k < 1:
        p.error("scales, cycles, and K must be positive")
    if POINTS_PER_SYMBOL < args.k + 2:
        p.error("Increase POINTS_PER_SYMBOL for this K")
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tasks = {}
    if args.task in ("sine_cube", "both"):
        tasks["sine_cube"] = generate_sine_cube(n_samples=2000, washout=100, train_ratio=0.8)
    if args.task in ("narma10", "both"):
        tasks["narma10"] = generate_narma10(n_samples=3000, washout=200, train_ratio=0.8, seed=42)
    rows = []
    for name, task in tasks.items():
        for cycles in args.cycles:
            for scale in args.scales:
                print(f"\n[{name}] I_SCALE={scale:.2e} A, Ts/Tc={cycles:g}, K={args.k}", flush=True)
                try:
                    row = one_run(name, task, scale, cycles, args.k, outdir)
                    print(f"  V1max={row['max_abs_V1_V']:.4g} V; gain reduction p95={row['gain_reduction_p95']:.3g}; "
                          f"R2(u_n)={row['current_input_test_R2']:.4f}; test NRMSE={row['test_NRMSE']:.4f}", flush=True)
                except Exception as exc:
                    print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
                    row = {"task": name, "input_scale_A": scale, "cycles_per_symbol": cycles,
                           "K": args.k, "error": f"{type(exc).__name__}: {exc}"}
                rows.append(row)
                keys = sorted({key for item in rows for key in item})
                with (outdir / "sweep.csv").open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(rows)
    for name in tasks:
        good = [r for r in rows if r["task"] == name and "test_NRMSE" in r]
        if not good:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for cycles in args.cycles:
            group = sorted((r for r in good if r["cycles_per_symbol"] == cycles),
                           key=lambda r: r["input_scale_A"])
            if group:
                ax.plot([r["input_scale_A"] for r in group],
                        [r["test_NRMSE"] for r in group], marker="o", label=f"Ts/Tc={cycles:g}")
        ax.set_xscale("log")
        ax.set_xlabel("Input current scale (A)")
        ax.set_ylabel("Test NRMSE (diagnostic only)")
        ax.set_title(f"{name}: operating-regime sweep")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"{name}_nrmse.png", dpi=180)
        plt.close(fig)
    print(f"\nSaved: {outdir / 'sweep.csv'}")
    print("CAUTION: Do not select parameters using final test NRMSE; use a validation split for selection.")


if __name__ == "__main__":
    main()
