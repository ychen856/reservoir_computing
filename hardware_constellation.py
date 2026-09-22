from linear_readout import predict_linear_readout, train_linear_readout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Configuration
# ============================================================

INPUT_CSV = "PT_RC_QPSK_32Gbaud_RRC_OSR4_200Symbols_AllSamples.csv"
OUTPUT_CSV = "PT_RC_QPSK_200symbol_2.csv"

# QPSK configuration
symbol_rate = 32.0e9          # 32 Gbaud
Ts = 1.0 / symbol_rate       # 31.25 ps

# Reference frequency used for I/Q projection
f_ref = 32.0e9               # 32 GHz

print(f"Symbol rate = {symbol_rate / 1e9:.3f} Gbaud")
print(f"Symbol period = {Ts * 1e12:.3f} ps")
print(f"I/Q reference frequency = {f_ref / 1e9:.3f} GHz")


# ============================================================
# 2. Load CSV files
# ============================================================

df_in = pd.read_csv(INPUT_CSV)
df_out = pd.read_csv(OUTPUT_CSV)

print("\n=== Input CSV ===")
print(df_in.columns.tolist())
print(f"Number of rows = {len(df_in)}")

print("\n=== Output CSV ===")
print(df_out.columns.tolist())
print(f"Number of rows = {len(df_out)}")


def parse_ads_value(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip()

    unit_scale = {
        "psec": 1e-12,
        "nsec": 1e-9,
        "usec": 1e-6,
        "msec": 1e-3,
        "sec":  1.0,

        'aV': 1e-18,
        'fV': 1e-15,
        'pV': 1e-12,
        'nV': 1e-9,
        "uV": 1e-6,
        "mV": 1e-3,
        "V":  1.0,

        "uA": 1e-6,
        "mA": 1e-3,
        "A":  1.0,
    }

    parts = s.split()

    # No unit

    if len(parts) == 1:
        return float(parts[0])
    if len(parts) != 2:
        raise ValueError(
            f"Unexpected ADS value format: {repr(s)}"
        )

    value_str, unit = parts

    if unit not in unit_scale:

        raise ValueError(
            f"Unknown ADS unit {repr(unit)} "
            f"in {repr(s)}"
        )

    return float(value_str) * unit_scale[unit]

def remove_ads_unit(series):
    """
    Remove the unit string appended by ADS.
    Examples:
        '5.2500 nsec' -> 5.2500
        '0.0123 V'    -> 0.0123
    """
    return (
        series.astype(str)
        .str.strip()
        .str.split()
        .str[0]
        .astype(float)
    )

for col in ["time", "Cc1", "Cc2", "Vctrl"]:
    df_out["time"] = df_out["time"].apply(parse_ads_value)
    df_out["Cc1"] = df_out["Cc1"].apply(parse_ads_value)
    df_out["Cc2"] = df_out["Cc2"].apply(parse_ads_value)
    df_out["Vctrl"] = df_out["Vctrl"].apply(parse_ads_value)



print("Columns:")
for i, col in enumerate(df_out.columns):
    print(i, repr(col))

print("\nShape:", df_out.shape)

print("\nFirst 10 rows:")
print(df_out.head(10).to_string())

print("\n=== Time diagnostics ===")

t = df_out["time"].to_numpy()

print("Number of output samples:", len(t))
print("t min =", t.min(), "s")
print("t max =", t.max(), "s")

dt = np.diff(t)

print("dt mean =", np.mean(dt), "s")
print("dt median =", np.median(dt), "s")
print("dt min =", np.min(dt), "s")
print("dt max =", np.max(dt), "s")

print("Equivalent sampling rate:",
      1 / np.median(dt) / 1e9, "GHz")

fc = 32e9
Tc = 1 / fc

print("32 GHz carrier period:",
      Tc * 1e12, "ps")

print("Samples per 32-GHz cycle:",
      Tc / np.median(dt))

print("\n=== Input timing ===")

print(df_in.head(20))
print(df_in.tail(20))
print("Number of input rows:", len(df_in))
    
# ============================================================
# 3. Extract valid symbol information
# ============================================================

# Only use valid input samples
df_valid = df_in[df_in["region"] == "valid"].copy()

# One row per symbol is enough for label/timing information.
# sampleWithinSymbol == 0 gives the first input sample of each symbol.
symbols = (
    df_valid[df_valid["sampleWithinSymbol"] == 0]
    .sort_values("validSymbolIndex")
    .reset_index(drop=True)
)

print("\n=== Symbol information ===")
print(f"Number of valid symbols = {len(symbols)}")

print(
    symbols[
        [
            "validSymbolIndex",
            "time_s",
            "classIndex",
            "bits",
            "phaseDeg"
        ]
    ].head()
)

# Symbol start times
symbol_times = symbols["time_s"].to_numpy(dtype=float)

# Labels
class_labels = symbols["classIndex"].to_numpy(dtype=int)
phase_labels = symbols["phaseDeg"].to_numpy(dtype=float)

print(
    f"\nFirst valid symbol time = "
    f"{symbol_times[0] * 1e9:.6f} ns"
)

print(
    f"Last valid symbol start = "
    f"{symbol_times[-1] * 1e9:.6f} ns"
)


# ============================================================
# 4. Load continuous reservoir waveform
# ============================================================

t_out = df_out["time"].to_numpy(dtype=float)
Cc1 = df_out["Cc1"].to_numpy(dtype=float)
Cc2 = df_out["Cc2"].to_numpy(dtype=float)

print("\n=== Continuous output ===")
print(f"Start time = {t_out[0] * 1e9:.6f} ns")
print(f"End time   = {t_out[-1] * 1e9:.6f} ns")

dt_out = np.diff(t_out)

print(
    f"Median output dt = "
    f"{np.median(dt_out) * 1e12:.3f} ps"
)


# ============================================================
# 5. I/Q projection function
# ============================================================

def iq_project_symbol(
    t,
    v,
    t_start,
    Ts,
    f_ref
):
    """
    Project one symbol window onto cosine/sine basis.

    Parameters
    ----------
    t : ndarray
        Continuous waveform timestamps.
    v : ndarray
        Continuous waveform voltage.
    t_start : float
        Symbol start time.
    Ts : float
        Symbol duration.
    f_ref : float
        Reference/carrier frequency.

    Returns
    -------
    I, Q
    """

    t_end = t_start + Ts

    # Samples belonging to this symbol
    mask = (t >= t_start) & (t < t_end)

    t_seg = t[mask]
    v_seg = v[mask]

    if len(t_seg) < 2:
        return np.nan, np.nan

    # --------------------------------------------------------
    # Local time reference:
    #
    # tau = 0 at the beginning of every symbol.
    # --------------------------------------------------------

    tau = t_seg - t_start

    cos_ref = np.cos(
        2.0 * np.pi * f_ref * tau
    )

    sin_ref = np.sin(
        2.0 * np.pi * f_ref * tau
    )

    # Numerical integration
    #
    # Scaling by 2/Ts gives approximately the sinusoidal
    # amplitude of the corresponding quadrature component.
    I = (2.0 / Ts) * np.trapz(
        v_seg * cos_ref,
        t_seg
    )

    Q = (2.0 / Ts) * np.trapz(
        v_seg * sin_ref,
        t_seg
    )

    return I, Q


# ============================================================
# 6. Project all 200 symbols
# ============================================================

I1 = []
Q1 = []

I2 = []
Q2 = []

for t_start in symbol_times:

    # Cc1
    i1, q1 = iq_project_symbol(
        t_out,
        Cc1,
        t_start,
        Ts,
        f_ref
    )

    # Cc2
    i2, q2 = iq_project_symbol(
        t_out,
        Cc2,
        t_start,
        Ts,
        f_ref
    )

    I1.append(i1)
    Q1.append(q1)

    I2.append(i2)
    Q2.append(q2)


I1 = np.asarray(I1)
Q1 = np.asarray(Q1)

I2 = np.asarray(I2)
Q2 = np.asarray(Q2)


# ============================================================
# 7. Sanity check
# ============================================================

print("\n=== I/Q projection sanity check ===")

print(f"I1 shape = {I1.shape}")
print(f"Q1 shape = {Q1.shape}")
print(f"I2 shape = {I2.shape}")
print(f"Q2 shape = {Q2.shape}")

print(f"Cc1 NaN count = {np.sum(~np.isfinite(I1) | ~np.isfinite(Q1))}")
print(f"Cc2 NaN count = {np.sum(~np.isfinite(I2) | ~np.isfinite(Q2))}")

print("\nPhase distribution:")
for phase in np.unique(phase_labels):
    print(
        f"{phase:6.1f} deg : "
        f"{np.sum(phase_labels == phase)} symbols"
    )


# ============================================================
# 8. Plot function
# ============================================================

def plot_constellation(
    I,
    Q,
    phase_labels,
    node_name
):

    plt.figure(figsize=(7, 7))

    phases = np.sort(
        np.unique(phase_labels)
    )

    for phase in phases:

        mask = phase_labels == phase

        plt.scatter(
            I[mask],
            Q[mask],
            s=35,
            alpha=0.75,
            label=rf"$\phi={phase:.0f}^\circ$"
        )

    # Zero axes
    plt.axhline(
        0,
        linewidth=1,
        alpha=0.5
    )

    plt.axvline(
        0,
        linewidth=1,
        alpha=0.5
    )

    plt.xlabel(
        "I: 32 GHz cosine projection (V)"
    )

    plt.ylabel(
        "Q: 32 GHz sine projection (V)"
    )

    plt.title(
        f"{node_name}: QPSK Phase Projection"
    )

    plt.grid(True)
    plt.legend()

    # Equal scaling is important for constellation geometry
    plt.axis("equal")

    plt.tight_layout()
    plt.show()


# ============================================================
# 9. Plot Cc1 and Cc2 constellation
# ============================================================

plot_constellation(
    I1,
    Q1,
    phase_labels,
    "Cc1"
)

plot_constellation(
    I2,
    Q2,
    phase_labels,
    "Cc2"
)


###########################################
# K=4
###########################################
sample_times = df_valid["time_s"].to_numpy()
sample_times = sample_times.reshape(200, 4)

K=4
# get input class label
y = df_valid["classIndex"].to_numpy().reshape(-1, K)[:, 0].astype(int)
print("sample_times:", sample_times.shape)
print("y:", y.shape)
print("classes:", np.unique(y, return_counts=True))

def sample_output_at_times(t_out, v_out, sample_times):
    """
    t_out: continuous ADS output timestamps, shape (N,)
    v_out: continuous output voltage, shape (N,)
    sample_times: desired timestamps, shape (n_symbol, K)
    """
    return np.interp(
        sample_times.ravel(),
        t_out,
        v_out
    ).reshape(sample_times.shape)

V1 = sample_output_at_times(
    t_out,
    Cc1,
    sample_times
)

V2 = sample_output_at_times(
    t_out,
    Cc2,
    sample_times
)

print(V1.shape)
print(V2.shape)


X = np.empty((len(V1), 2 * V1.shape[1]))

X[:, 0::2] = V1
X[:, 1::2] = V2

print("X shape:", X.shape)
print("First sample:")
print(X[0])


i = 0

ts = sample_times[i]

mask = (
    (t_out >= ts[0] - 0.2e-9) &
    (t_out <= ts[-1] + 0.2e-9)
)

plt.figure(figsize=(10, 5))

plt.plot(
    t_out[mask] * 1e9,
    Cc1[mask],
    label="Cc1"
)

plt.plot(
    t_out[mask] * 1e9,
    Cc2[mask],
    label="Cc2"
)

for t in ts:
    plt.axvline(
        t * 1e9,
        linestyle="--",
        alpha=0.5
    )

plt.xlabel("Time (ns)")
plt.ylabel("Voltage (V)")
plt.title("Reservoir waveform and K=4 sampling locations")
plt.legend()
plt.grid(True)
plt.show()

# ===================================
# plot reservoir constellation
# ===================================
mu = X.mean(axis=0)
std = X.std(axis=0)

std[std < 1e-12] = 1.0

X_n = (X - mu) / std

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
Z = pca.fit_transform(X_n)

print(
    "Explained variance:",
    pca.explained_variance_ratio_
)


phase_deg = np.array([45, 135, 225, 315])

plt.figure(figsize=(7, 7))

for c in range(4):
    mask = (y == c)

    plt.scatter(
        Z[mask, 0],
        Z[mask, 1],
        alpha=0.7,
        label=rf"$\phi={phase_deg[c]}^\circ$"
    )

plt.xlabel(
    f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)"
)
plt.ylabel(
    f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)"
)

plt.title("Hardware Reservoir Feature Clustering, K=4")
plt.legend()
plt.grid(True)
plt.show()



# ==================================
# W training
# ==================================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

mu = X_train.mean(axis=0)
std = X_train.std(axis=0)

std = np.where(std < 1e-12, 1.0, std)

X_train_n = (X_train - mu) / std
X_test_n  = (X_test  - mu) / std

Wout = train_linear_readout(
    X_train_n,
    y_train,
    n_classes=4,
    ridge=1e-6
)

train_pred, train_scores = predict_linear_readout(
    X_train_n,
    Wout
)

test_pred, test_scores = predict_linear_readout(
    X_test_n,
    Wout
)

print(
    f"Train accuracy = "
    f"{np.mean(train_pred == y_train)*100:.2f}%"
)

print(
    f"Test accuracy = "
    f"{np.mean(test_pred == y_test)*100:.2f}%"
)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, test_pred)

print(cm)