#!/usr/bin/env python3
"""Plot held-out amplitude integration and Fourier results for both topologies."""
import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("results/pass_gqc/amplitude_scaling_d5_2nH_xi2_4")
TASKS = {"integral_order1", "integral_order2", "fourier"}
KEYS = ["topology", "dimers", "window_symbols", "task", "test_error", "worst_bin_nrmse"]


def read_job(root, topology, count):
    job = root / topology / f"{count}dimers"
    paths = list(job.rglob("summary.csv"))
    if not (job / ".complete").is_file() or len(paths) != 1:
        raise ValueError(f"Incomplete or ambiguous output: {job}")
    summary = paths[0]
    grid = {task: set() for task in TASKS}
    out = []
    with summary.open(newline="") as stream:
        for row in csv.DictReader(stream):
            task = row["task"]
            if task not in TASKS or (row["source"], row["readout"], row["encoding"]) != (
                "reservoir", "linear", "amplitude"
            ):
                continue
            expected = {
                "design_index": 5, "xi2": 4, "seed": 44, "n_dimers": count,
            }
            if topology == "coupled":
                expected.update(coupling_l_nh=2, coupling_r_ohm=1)
            if (
                any(float(row[key]) != value for key, value in expected.items())
                or row["circuit_type"] != topology or row["mode"] != "graphene"
            ):
                raise ValueError(f"Settings mismatch in {summary}")
            window = int(row["window"])
            if window in grid[task]:
                raise ValueError(f"Duplicate {task} window {window} in {summary}")
            grid[task].add(window)
            worst = ""
            if task == "fourier":
                error = float(row["test_complex_nrmse"])
                with np.load(summary.parent / f"fourier_reservoir_L{window}.npz") as data:
                    bins = data["per_bin_nrmse"]
                if bins.size == 0 or not np.all(np.isfinite(bins) & (bins >= 0)):
                    raise ValueError(f"Invalid per-bin errors for {job}, window {window}")
                worst = float(bins.max())
            else:
                error = float(row["test_nmse"])
            if not math.isfinite(error) or error < 0:
                raise ValueError(f"Invalid test error for {job}, {task}, window {window}")
            out.append(dict(zip(KEYS, (topology, count, window, task, error, worst))))
    if not grid["fourier"] or any(windows != grid["fourier"] for windows in grid.values()):
        raise ValueError(f"Missing tasks or different window grids in {summary}")
    return out, grid["fourier"]


def draw(root, rows, counts, topologies, title, filename, panels):
    fig, axes = plt.subplots(2, len(topologies), squeeze=False, figsize=(6.5 * len(topologies), 8),
                             constrained_layout=True)
    for col, topology in enumerate(topologies):
        for axis, (task, metric, ylabel) in zip(axes[:, col], panels):
            for count in counts:
                if topology == "coupled" and count == 1:
                    continue
                records = sorted(
                    (row for row in rows if (row["topology"], row["dimers"], row["task"])
                     == (topology, count, task)), key=lambda row: row["window_symbols"]
                )
                axis.plot(
                    [row["window_symbols"] for row in records],
                    np.maximum([row[metric] for row in records], 1e-15),
                    "o-", markersize=3, color=f"C{counts.index(count)}", label=f"{count} dimers",
                )
            axis.set(title=f"{topology.capitalize()} · {ylabel}", xlabel="Window (symbols)",
                     ylabel="Held-out error", yscale="log")
            axis.axhline(0.1 if task == "fourier" else 0.01, color="black", linestyle=":", linewidth=1.4)
            axis.grid(True, which="both", alpha=0.22)
    for upper, lower in zip(axes[0], axes[1]):
        upper.legend()
    for row in axes:
        y_limits = [ax.get_ylim() for ax in row]
        for ax in row:
            ax.set_ylim(min(bounds[0] for bounds in y_limits), max(bounds[1] for bounds in y_limits))
    fig.suptitle(f"{title}\nDesign 5 graphene, ξ₂ = 4, seed 44 · Coupled links: 2 nH / 1 Ω")
    for ext in ("png", "pdf"):
        fig.savefig(root / f"{filename}.{ext}", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--counts", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--topologies", nargs="+", choices=["open", "coupled"], default=["open", "coupled"])
    args = parser.parse_args()
    counts = sorted(set(args.counts))
    topologies = list(dict.fromkeys(args.topologies))
    rows = []
    grids = []
    for topology in topologies:
        for count in counts:
            if topology == "coupled" and count == 1:
                continue
            records, grid = read_job(args.root, topology, count)
            rows.extend(records)
            grids.append(grid)
    if any(grid != grids[0] for grid in grids[1:]):
        raise ValueError("The requested runs have different window grids.")
    draw(args.root, rows, counts, topologies, "Amplitude-encoded integration", "integral_scaling", [
        ("integral_order1", "test_error", "First-order NMSE"),
        ("integral_order2", "test_error", "Second-order NMSE"),
    ])
    draw(args.root, rows, counts, topologies, "Fourier scaling", "fourier_scaling", [
        ("fourier", "test_error", "Overall complex NRMSE"),
        ("fourier", "worst_bin_nrmse", "Worst-bin NRMSE"),
    ])
    with (args.root / "amplitude_scaling.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=KEYS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
