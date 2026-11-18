#!/usr/bin/env python3
"""Compare held-out 32-dimer A/B results with homogeneous controls."""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_gqc_amplitude_scaling import ROOT, read_job

DUAL_ROOT = Path("results/pass_gqc/amplitude_dual32_d5_xi2_0p5_2")
LABELS = {
    "dual_AB": "A16 + B16",
    "matched_AA": "A16 + A16",
    "matched_BB": "B16 + B16",
    "homogeneous_A": "One 32-dimer bank, ξ₂ = 0.5",
    "homogeneous_xi4": "One 32-dimer bank, ξ₂ = 4",
}
PANELS = (
    ("integral_order1", "test_error", "First-order integral NMSE", .01),
    ("integral_order2", "test_error", "Second-order integral NMSE", .01),
    ("fourier", "test_error", "Overall complex NRMSE", .1),
    ("fourier", "worst_bin_nrmse", "Worst-bin NRMSE", .1),
)


def read_dual(path, topology):
    records = []
    with (path / "summary.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["encoding"] != "amplitude":
                continue
            variant = row["configuration"]
            window = int(row["window"])
            worst = ""
            if row["task"] == "fourier":
                with np.load(path / f"{variant}_fourier_{window}.npz") as data:
                    target = data["target"]
                    prediction = data["prediction"]
                variance = np.mean(np.abs(target - target.mean(axis=0)) ** 2, axis=0)
                error = np.mean(np.abs(prediction - target) ** 2, axis=0)
                valid_bins = variance > 0
                if not valid_bins.any():
                    raise ValueError(f"All Fourier bins have zero variance in {path}, window {window}")
                worst = float(np.sqrt(np.max(error[valid_bins] / variance[valid_bins])))
            records.append(dict(
                topology=topology, configuration=variant, task=row["task"],
                window_symbols=window, test_error=float(row["test"]), worst_bin_nrmse=worst,
            ))
    return records


def plot(topology, records, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    available = {row["configuration"] for row in records}
    if "dual_AB" not in available:
        raise ValueError(f"Missing A/B results for {topology}")
    for ax, (task, metric, title, threshold) in zip(axes.flat, PANELS):
        reference_grid = None
        for variant, label in LABELS.items():
            if variant not in available:
                continue
            selected = sorted((r for r in records if r["task"] == task and r["configuration"] == variant),
                              key=lambda r: r["window_symbols"])
            windows = [r["window_symbols"] for r in selected]
            if not windows or len(windows) != len(set(windows)):
                raise ValueError(f"Missing or duplicate {variant} windows for {task}")
            if reference_grid is not None and windows != reference_grid:
                raise ValueError(f"Window grid mismatch for {variant}, {task}")
            reference_grid = windows
            values = np.array([r[metric] for r in selected], dtype=float)
            if not np.all(np.isfinite(values) & (values >= 0)):
                raise ValueError(f"Invalid {variant} values for {task}")
            ax.plot(windows, np.maximum(values, 1e-15), "o-", markersize=3, label=label)
        ax.axhline(threshold, color="black", linestyle=":")
        ax.set(title=title, xlabel="Window (symbols)", ylabel="Held-out error", yscale="log")
        ax.grid(True, which="both", alpha=.22)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(f"Design 5 graphene, {topology}, 32 dimers, seed 44\n"
                 "A: ξ₂ = 0.5 · B: ξ₂ = 2 · Coupled links: 2 nH / 1 Ω")
    for ext in ("png", "pdf"):
        fig.savefig(output / f"dual32_{topology}.{ext}", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=ROOT)
    parser.add_argument("--dual-root", type=Path)
    parser.add_argument("--ab-only", action="store_true")
    parser.add_argument("--topologies", nargs="+", choices=["open", "coupled"], default=["open", "coupled"])
    args = parser.parse_args()
    args.dual_root = args.dual_root or (DUAL_ROOT.with_name(DUAL_ROOT.name + "_ab_only")
                                       if args.ab_only else DUAL_ROOT)
    records = []
    for topology in dict.fromkeys(args.topologies):
        run = args.dual_root / topology
        if not (run / ".complete").is_file():
            raise ValueError(f"Incomplete dual run: {run}")
        baseline, _ = ([], None) if args.ab_only else read_job(args.baseline_root, topology, 32)
        data = read_dual(run, topology)
        for row in baseline:
            row.pop("dimers")
            row["configuration"] = "homogeneous_xi4"
            data.append(row)
        plot(topology, data, args.dual_root)
        records.extend(data)
    with (args.dual_root / "dual32_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
