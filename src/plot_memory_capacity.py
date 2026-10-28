import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# Configuration
# ============================================================

INPUT_CSV = Path("results/memory_capacity_validation/memory_windows.csv")

OUTPUT_DIR = Path("results/memory_capacity_validation/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TS_RATIOS = [0.025, 0.5]

N_VALUES = [3000, 10000, 30000]

N_PLOT = 30000

CONFIGS = [
    (0.0, 0.0),
    (1.0, 0.0),
    (1.0, 2.78),
    (1.0, 8.33),
]

WINDOWS = [
    "1:10",
    "11:50",
    "51:100",
    "101:200",
]

COLORS = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
]

LABELS = [
    r"$\eta=0,\ \alpha=0$",
    r"$\eta=1,\ \alpha=0$",
    r"$\eta=1,\ \alpha=2.78$",
    r"$\eta=1,\ \alpha=8.33$",
]


# ============================================================
# Plot style
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "axes.linewidth": 1.5,
    "xtick.major.width": 1.3,
    "ytick.major.width": 1.3,
    "legend.fontsize": 10,
    "figure.dpi": 120,
    "savefig.dpi": 300,
})


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(INPUT_CSV)

required_columns = {
    "eta",
    "alpha",
    "ts_ratio",
    "n_symbols",
    "window",
    "real_mc",
    "null_mc",
    "excess_mc",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

print(f"Loaded {len(df)} rows from {INPUT_CSV}")


# ============================================================
# Helpers
# ============================================================

def select_config(data, ts_ratio, eta, alpha, n_symbols=None):

    mask = (
        np.isclose(data["ts_ratio"], ts_ratio)
        & np.isclose(data["eta"], eta)
        & np.isclose(data["alpha"], alpha)
    )

    if n_symbols is not None:
        mask &= data["n_symbols"] == n_symbols

    return data.loc[mask].copy()


def save_figure(fig, filename):

    path = OUTPUT_DIR / filename

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Saved: {path}")

    plt.close(fig)


def format_axes(ax):

    ax.grid(
        True,
        axis="y",
        alpha=0.25,
        linewidth=0.8,
    )

    ax.set_axisbelow(True)

    ax.tick_params(
        direction="out",
        length=5,
        width=1.3,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


# ============================================================
# Figure 1: Total MC vs. N
# ============================================================

def plot_mc_convergence(data):

    summary = (
        data.groupby(
            ["ts_ratio", "eta", "alpha", "n_symbols"],
            as_index=False,
        )[["real_mc", "null_mc", "excess_mc"]]
        .sum()
    )

    for ts_ratio in TS_RATIOS:

        fig, ax = plt.subplots(figsize=(7, 4.8))

        for i, (eta, alpha) in enumerate(CONFIGS):

            sub = select_config(
                summary,
                ts_ratio,
                eta,
                alpha,
            ).sort_values("n_symbols")

            if sub.empty:
                print(
                    f"Missing convergence data: "
                    f"Ts/Tc={ts_ratio}, eta={eta}, alpha={alpha}"
                )
                continue

            ax.plot(
                sub["n_symbols"],
                sub["excess_mc"],
                marker="o",
                markersize=7,
                linewidth=2.2,
                color=COLORS[i],
                label=LABELS[i],
            )

        ax.set_xlabel("Number of symbols")
        ax.set_ylabel("Excess total MC")

        ax.set_title(
            rf"$T_s/T_c={ts_ratio}$"
        )

        ax.set_xticks(N_VALUES)

        ax.legend(
            loc="best",
            frameon=True,
            edgecolor="black",
        )

        format_axes(ax)
        fig.tight_layout()

        save_figure(
            fig,
            f"mc_convergence_ts_{ts_ratio}.png",
        )


# ============================================================
# Figure 2: Memory distribution
# ============================================================

def plot_memory_distribution(data):

    for ts_ratio in TS_RATIOS:

        fig, ax = plt.subplots(figsize=(8, 5))

        x = np.arange(len(WINDOWS))

        bar_width = 0.19

        for i, (eta, alpha) in enumerate(CONFIGS):

            sub = select_config(
                data,
                ts_ratio,
                eta,
                alpha,
                n_symbols=N_PLOT,
            )

            # Preserve the defined window order
            sub = (
                sub.set_index("window")
                .reindex(WINDOWS)
            )

            if sub["excess_mc"].isna().any():
                print(
                    f"Missing distribution data: "
                    f"Ts/Tc={ts_ratio}, eta={eta}, alpha={alpha}"
                )
                continue

            values = sub["excess_mc"].to_numpy()

            offset = (
                i - (len(CONFIGS) - 1) / 2
            ) * bar_width

            ax.bar(
                x + offset,
                values,
                width=bar_width,
                color=COLORS[i],
                edgecolor="black",
                linewidth=0.8,
                label=LABELS[i],
            )

        ax.set_xticks(x)
        ax.set_xticklabels(WINDOWS)

        ax.set_xlabel("Delay window (symbols)")
        ax.set_ylabel("Excess memory capacity")

        ax.set_title(
            rf"$T_s/T_c={ts_ratio}$, $N={N_PLOT}$"
        )

        ax.legend(
            loc="best",
            frameon=True,
            edgecolor="black",
        )

        format_axes(ax)
        fig.tight_layout()

        save_figure(
            fig,
            f"mc_distribution_ts_{ts_ratio}.png",
        )


# ============================================================
# Figure 3: Real MC vs. Null MC
# ============================================================

def plot_null_validation(data):

    summary = (
        data.groupby(
            ["ts_ratio", "eta", "alpha", "n_symbols"],
            as_index=False,
        )[["real_mc", "null_mc"]]
        .sum()
    )

    # Choose one representative configuration
    eta = 1.0
    alpha = 8.33

    for ts_ratio in TS_RATIOS:

        sub = select_config(
            summary,
            ts_ratio,
            eta,
            alpha,
        ).sort_values("n_symbols")

        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(7, 4.8))

        ax.plot(
            sub["n_symbols"],
            sub["real_mc"],
            marker="o",
            markersize=7,
            linewidth=2.2,
            color=COLORS[0],
            label="Real MC",
        )

        ax.plot(
            sub["n_symbols"],
            sub["null_mc"],
            marker="s",
            markersize=7,
            linewidth=2.2,
            linestyle="--",
            color=COLORS[3],
            label="Null MC",
        )

        ax.set_xlabel("Number of symbols")
        ax.set_ylabel("Total memory capacity")

        ax.set_title(
            rf"$T_s/T_c={ts_ratio}$, "
            rf"$\eta={eta}$, $\alpha={alpha}$"
        )

        ax.set_xticks(N_VALUES)

        ax.legend(
            frameon=True,
            edgecolor="black",
        )

        format_axes(ax)
        fig.tight_layout()

        save_figure(
            fig,
            f"mc_null_validation_ts_{ts_ratio}.png",
        )


# ============================================================
# Main
# ============================================================

def main():

    plot_mc_convergence(df)

    plot_memory_distribution(df)

    plot_null_validation(df)

    print("\nAll figures generated successfully.")


if __name__ == "__main__":
    main()