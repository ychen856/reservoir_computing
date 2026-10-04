from linear_readout import predict_linear_readout, train_linear_readout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from utils import calculate_ber, normalize_bits


plt.rcParams.update({
    "font.size": 28,
    "axes.labelsize": 28,
    "axes.titlesize": 28,
    "legend.fontsize": 28,
    #"figure.dpi": 100,
    'font.sans-serif': 'Arial',
    'axes.unicode_minus': 0,
    'font.weight': 'bold',
    "axes.labelweight": 'bold',
    "axes.titleweight": 'bold',
})


def rebuild_fixed_timestamps(df, dt=100e-12, t0=0.0, time_col="time"):
    """
    Rebuild timestamp column assuming a fixed sampling interval.

    Parameters
    ----------
    df : pd.DataFrame
        Output dataframe.
    dt : float
        Fixed sampling interval in seconds.
        Default = 100 ps.
    t0 : float
        Starting time in seconds.
    time_col : str
        Name of timestamp column.

    Returns
    -------
    df : pd.DataFrame
        Dataframe with reconstructed timestamps.
    """
    df = df.copy()

    df[time_col] = t0 + np.arange(len(df), dtype=float) * dt

    return df

# =================================
# special charactor util
# =================================
def unit_parser(x):
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
# 8. Plot function
# ============================================================
# plot time alignment
def plot_time_alignment(
    df_valid,
    t_out,
    V1,
    V2,
    n_symbols=10,
):
    """
    Plot a short section of input timing / class labels together
    with hardware reservoir V1 and V2.

    Used only as a time-alignment sanity check.
    """

    # --------------------------------------------------
    # One row per symbol
    # --------------------------------------------------
    symbols = (
        #df_valid[df_valid["sampleWithinSymbol"] == 0]
        #.sort_values("validSymbolIndex")
        df_valid
        .reset_index(drop=True)
    )

    symbols = symbols.iloc[:n_symbols]

    #symbol_starts = symbols["time_s"].to_numpy(dtype=float)
    symbol_starts = symbols['startTime_s'].to_numpy(dtype=float)
    class_labels = symbols["classIndex"].to_numpy(dtype=int)

    # Estimate symbol period from input file itself
    Ts_check = np.median(np.diff(symbol_starts))

    t_begin = symbol_starts[0]
    t_end = symbol_starts[-1] + Ts_check

    # --------------------------------------------------
    # Output waveform in the same time range
    # --------------------------------------------------
    mask_out = (
        (t_out >= t_begin) &
        (t_out <= t_end)
    )

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    plt.figure(figsize=(14, 6))

    plt.plot(
        t_out[mask_out] * 1e9,
        V1[mask_out],
        label="V1"
    )

    plt.plot(
        t_out[mask_out] * 1e9,
        V2[mask_out],
        label="V2"
    )

    # Symbol boundaries + class labels
    y_top = max(
        np.max(V1[mask_out]),
        np.max(V2[mask_out])
    )

    for i, (t0, c) in enumerate(
        zip(symbol_starts, class_labels)
    ):
        plt.axvline(
            t0 * 1e9,
            linestyle="--",
            alpha=0.4
        )

        plt.text(
            (t0 + 0.5 * Ts_check) * 1e9,
            y_top * 0.92,
            f"C{c}",
            ha="center",
            va="top"
        )

    # Final boundary
    plt.axvline(
        t_end * 1e9,
        linestyle="--",
        alpha=0.4
    )

    plt.xlabel("Time (ns)")
    plt.ylabel("Reservoir voltage (V)")
    plt.title(
        f"Input / Reservoir Time Alignment "
        f"({n_symbols} symbols)"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # Diagnostics
    # --------------------------------------------------
    print("\n=== Time alignment sanity check ===")
    print(
        f"Input first symbol start : "
        f"{symbol_starts[0]*1e9:.6f} ns"
    )
    print(
        f"Estimated symbol period  : "
        f"{Ts_check*1e9:.6f} ns"
    )
    print(
        f"Output start time        : "
        f"{t_out[0]*1e9:.6f} ns"
    )

    print("\nFirst symbols:")
    for i in range(min(n_symbols, len(symbols))):
        print(
            f"symbol {int(symbols.iloc[i]['validSymbolIndex']):5d} | "
            f"t = {symbol_starts[i]*1e9:10.6f} ns | "
            f"class = {class_labels[i]}"
        )

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



def main():
    # 1. Configuration
    INPUT_CSV = 'PT_RC_16QAM_40000Symbols_Labels.csv'
    INPUT_CSV_TEST = 'PT_RC_16QAM_500Mbaud_RRC_OSR4_200Symbols_Labels.csv'
    
    #OUTPUT_CSV = 'PT_RC_16QAM_2G_40000symbol_Res100ps.csv'
    #OUTPUT_CSV_TEST = 'PT_RC_16QAM_2G_200symbol_Res100ps.csv'

    OUTPUT_CSV = 'PT_RC_16QAM_2G_40000symbol_ISI3P_Res100ps.csv'
    OUTPUT_CSV_TEST = 'PT_RC_16QAM_2G_200symbol_ISI3P_Res100ps.csv'
    
    #OUTPUT_CSV = 'PT_RC_16QAM_2G_40000symbol_AWGN10dB_Res100ps.csv'
    #OUTPUT_CSV_TEST = 'PT_RC_16QAM_2G_200symbol_AWGN10dB_Res100ps.csv'

    # QPSK configuration
    #symbol_rate = 32.0e9          # 32 Gbaud
    symbol_rate = 500.0e6
    Ts = 1.0 / symbol_rate       # 31.25 ps

    # Reference frequency used for I/Q projection
    #f_ref = 32.0e9               # 32 GHz
    f_ref = 500.0e6
    print(f"Symbol rate = {symbol_rate / 1e9:.3f} Gbaud")
    print(f"Symbol period = {Ts * 1e12:.3f} ps")
    print(f"I/Q reference frequency = {f_ref / 1e9:.3f} GHz")


    # 2. Load CSV files
    df_in = pd.read_csv(INPUT_CSV)
    df_out = pd.read_csv(OUTPUT_CSV)

    print("\n=== Input CSV ===")
    print(df_in.columns.tolist())
    print(f"Number of rows = {len(df_in)}")

    print("\n=== Output CSV ===")
    print(df_out.columns.tolist())
    print(f"Number of rows = {len(df_out)}")


    # unit parsing
    for col in ["time", "Cc1", "Cc2"]:
        df_out[col] = df_out[col].apply(unit_parser)

    # Rebuild output timestamps
    OUTPUT_DT = 100e-12  # 100 ps

    df_out = rebuild_fixed_timestamps(
        df_out,
        dt=OUTPUT_DT,
        t0=0.0,
        time_col="time"
    )
        

    # data verification
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
    print(f"Output sampling rate = {1/dt/1e9} GHz")

    print("Equivalent sampling rate:",
          1 / np.median(dt) / 1e9, "GHz")


    #fc = 32e9
    fc = 500.0e6
    Tc = 1 / fc

    print("Carrier period:", Tc * 1e12, "ps")
    print("Samples per cycle:", Tc / np.median(dt))

    print("\n=== Input timing ===")

    print(df_in.head(20))
    print(df_in.tail(20))
    print("Number of input rows:", len(df_in))

    
    # 3. Extract valid symbol information
    # Only use valid input samples
    #df_valid = df_in[df_in["region"] == "valid"].copy()
    df_valid = df_in.copy()

    # One row per symbol is enough for label/timing information.
    # sampleWithinSymbol == 0 gives the first input sample of each symbol.
    symbols = (
        #df_valid[df_valid["sampleWithinSymbol"] == 0]
        #.sort_values("validSymbolIndex")
        df_valid
        .reset_index(drop=True)
    )

    print("\n=== Symbol information ===")
    print(f"Number of valid symbols = {len(symbols)}")

    print(
        symbols[
            [
                #"validSymbolIndex",
                #"time_s",
                #"classIndex",
                #"bits",
                #"phaseDeg"
                'startTime_s',
                "classIndex",
                'bits',
                #'phaseDeg'
            ]
        ].head()
    )

    # Symbol start times
    #symbol_times = symbols["time_s"].to_numpy(dtype=float)
    symbol_times = symbols["startTime_s"].to_numpy(dtype=float)

    Ts_check = np.median(np.diff(symbol_times))
    dt_out = np.median(dt)
    print(f"Symbol period          = {Ts_check*1e9} ns")
    print(f"Output pts/symbol      = {Ts_check/dt}")


    # Labels
    class_labels = symbols["classIndex"].to_numpy(dtype=int)
    #phase_labels = symbols["phaseDeg"].to_numpy(dtype=float)

    print(
        f"\nFirst valid symbol time = "
        f"{symbol_times[0] * 1e9:.6f} ns"
    )

    print(
        f"Last valid symbol start = "
        f"{symbol_times[-1] * 1e9:.6f} ns"
    )


    # 4. Load continuous reservoir waveform
    t_out = df_out["time"].to_numpy(dtype=float)
    Cc1 = df_out["Cc1"].to_numpy(dtype=float)
    Cc2 = df_out["Cc2"].to_numpy(dtype=float)

    # plot input/output alignment
    plot_time_alignment(
        df_valid,
        t_out,
        Cc1,
        Cc2,
        n_symbols=10,
    )

    print("\n=== Continuous output ===")
    print(f"Start time = {t_out[0] * 1e9:.6f} ns")
    print(f"End time   = {t_out[-1] * 1e9:.6f} ns")

    dt_out = np.diff(t_out)

    print(
        f"Median output dt = "
        f"{np.median(dt_out) * 1e12:.3f} ps"
    )







    print("\n=== Output data integrity ===")

    print("Total rows:", len(df_out))
    print("Unique timestamps:", df_out["time"].nunique())
    print("Duplicated timestamps:", df_out["time"].duplicated().sum())

    print("Cc1 NaN:", df_out["Cc1"].isna().sum())
    print("Cc2 NaN:", df_out["Cc2"].isna().sum())
    print("time NaN:", df_out["time"].isna().sum())

    bad = (
        df_out["Cc1"].isna()
        | df_out["Cc2"].isna()
        | df_out["time"].isna()
    )

    print("\nFirst bad rows:")
    print(df_out.loc[bad].head(20))

    print("\nRows around first bad row:")
    if bad.any():
        i = np.flatnonzero(bad.to_numpy())[0]
        print(df_out.iloc[max(0, i-5):i+6])


    t = df_out["time"].to_numpy()

    dt_all = np.diff(t)

    print("\n=== dt structure ===")
    print("dt == 0:", np.sum(dt_all == 0))
    print("dt < 0 :", np.sum(dt_all < 0))
    print("dt > 0 :", np.sum(dt_all > 0))

    positive_dt = dt_all[dt_all > 0]

    print("Median positive dt:",
          np.median(positive_dt) * 1e12, "ps")

    vals, counts = np.unique(np.round(dt_all * 1e12, 6),
                         return_counts=True)

    idx = np.argsort(counts)[::-1][:10]

    print("\nMost common dt values:")
    for v, c in zip(vals[idx], counts[idx]):
        print(f"{v:12.6f} ps : {c}")

















    # 6. Project all 200 symbols
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


    # 7. Sanity check
    print("\n=== I/Q projection sanity check ===")

    print(f"I1 shape = {I1.shape}")
    print(f"Q1 shape = {Q1.shape}")
    print(f"I2 shape = {I2.shape}")
    print(f"Q2 shape = {Q2.shape}")

    print(f"Cc1 NaN count = {np.sum(~np.isfinite(I1) | ~np.isfinite(Q1))}")
    print(f"Cc2 NaN count = {np.sum(~np.isfinite(I2) | ~np.isfinite(Q2))}")

    print("\nPhase distribution:")
    '''for phase in np.unique(phase_labels):
        print(
            f"{phase:6.1f} deg : "
            f"{np.sum(phase_labels == phase)} symbols"
        )'''
    for c in np.unique(class_labels):
            print(
                #f"{phase:6.1f} deg : "
                f"{np.sum(class_labels == c)} symbols"
            )


    # 9. Plot Cc1 and Cc2 constellation
    plot_constellation(
        I1,
        Q1,
        #phase_labels,
        class_labels,
        "Cc1"   
    )

    plot_constellation(
        I2,
        Q2,
        #phase_labels,
        class_labels,
        "Cc2"
    )





    ###########################################
    # for given K
    ###########################################
    #sample_times = df_valid["time_s"].to_numpy()
    '''sample_times = df_valid["startTime_s"].to_numpy()
    sample_times = sample_times.reshape(200, 4)'''

    ###########################################
    # for given K
    ###########################################

    K = 8

    symbol_starts = df_valid["startTime_s"].to_numpy(dtype=float)
    symbol_ends   = df_valid["endTime_s"].to_numpy(dtype=float)

    # Avoid sampling exactly at the next symbol boundary.
    # For K=4, fractions = [0, 0.25, 0.5, 0.75]
    fractions = np.arange(K) / K

    sample_times = (
        symbol_starts[:, None]
        + fractions[None, :]
        * (symbol_ends - symbol_starts)[:, None]
    )

    # Labels: one label per symbol
    y = df_valid["classIndex"].to_numpy(dtype=int)

    print("sample_times:", sample_times.shape)
    print("y:", y.shape)
    print("classes:", np.unique(y, return_counts=True))

    print("\nFirst symbol:")
    print("start =", symbol_starts[0] * 1e9, "ns")
    print("end   =", symbol_ends[0] * 1e9, "ns")
    print("sample times =", sample_times[0] * 1e9, "ns")




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


    # plot K sample time
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


    # plot reservoir constellation
    mu = X.mean(axis=0)
    std = X.std(axis=0)

    std[std < 1e-12] = 1.0

    X_n = (X - mu) / std


    pca = PCA(n_components=2)
    Z = pca.fit_transform(X_n)

    print(
        "Explained variance:",
        pca.explained_variance_ratio_
    )


    #phase_deg = np.array([45, 135, 225, 315])
    # rotation
    def rotate_2d(X, angle_deg):
        theta = np.deg2rad(angle_deg)

        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

        return X @ R.T

    Z = rotate_2d(Z, 50)
    
    '''colors_16qam = [
        "#1F77B4",  # blue
        "#FF7F0E",  # orange
        "#2CA02C",  # green
        "#D62728",  # red
        "#9467BD",  # purple
        "#8C564B",  # brown
        "#E377C2",  # pink
        "#17BECF",  # cyan

        "#393B79",  # dark indigo
        "#637939",  # olive green
        "#8C6D31",  # ochre
        "#843C39",  # dark red
        "#7B4173",  # plum
        "#3182BD",  # medium blue
        "#E6550D",  # burnt orange
        "#31A354",  # emerald green
    ]'''
    colors_16qam = [
        "#4E79A7",  # blue
        "#A0CBE8",  # light blue
        "#59A14F",  # green
        "#8CD17D",  # light green

        "#F28E2B",  # orange
        "#FFBE7D",  # light orange
        "#E15759",  # red
        "#FF9D9A",  # light red

        "#B07AA1",  # purple
        "#D4A6C8",  # light purple
        "#9C755F",  # brown
        "#D7B5A6",  # light brown

        "#76B7B2",  # teal
        "#86BCB6",  # light teal
        "#79706E",  # gray-brown
        "#BAB0AC",  # warm gray
    ]
    plt.figure(figsize=(7, 7))

    for c in range(16):
        mask = (y == c)

        plt.scatter(
            Z[mask, 0],
            Z[mask, 1],
            alpha=0.6,
            s=5,
            color=colors_16qam[c],
            label=rf"$class={class_labels[c]}$"
        )

        center = Z[mask].mean(axis=0)
        
        plt.scatter(
            center[0],
            center[1],
            marker='X',
            s=150,
            color=colors_16qam[c],
            edgecolors='black',
            linewidths=2.0,
            zorder=10
        )

    plt.xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)"
    )
    plt.ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)"
    )

    #plt.title("16 QAM - Ideal, K=8")
    plt.title("16 QAM - 3-Path ISI, K=8")
    #plt.title("16 QAM - AWGN 10 dB, K=8")
    #plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(
        #'16qam_awgn_v4.png',
        '16pam_isi_v4.png',
        #'16qam_ideal_v4.png',
        dpi=300
        )
    
    plt.show()


    poly = PolynomialFeatures(
        degree=4,
        include_bias=False
    )

    X_poly = poly.fit_transform(X)
    print("Original:", X.shape)
    print("Poly:", X_poly.shape)



    # W training
    X_train, X_test, y_train, y_test = train_test_split(
        X_poly,
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
        n_classes=16,
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


    #cm = confusion_matrix(y_test, test_pred)
    #print(cm)


    '''pca = PCA(n_components=3)
    Z = pca.fit_transform(X_n)

    print("Explained variance:")
    print(pca.explained_variance_ratio_)
    print("Total:", pca.explained_variance_ratio_.sum())

    pairs = [
        (0, 1),
        (0, 2),
        (1, 2),
    ]

    for a, b in pairs:

        plt.figure(figsize=(7, 7))

        for c in range(16):
            mask = (y == c)

            plt.scatter(
                Z[mask, a],
                Z[mask, b],
                s=15,
                alpha=0.5,
                label=rf"$class={class_labels[c]}^\circ$"
            )

        plt.xlabel(f"PC{a+1}")
        plt.ylabel(f"PC{b+1}")
        plt.title(f"PC{a+1} vs PC{b+1}")
        plt.legend()
        plt.grid(True)
        plt.show()'''

















    # ========================================
    # ========================================
    # ========================================

    df_in_test = pd.read_csv(INPUT_CSV_TEST)
    df_out_test = pd.read_csv(OUTPUT_CSV_TEST)

    print("\n=== Input CSV ===")
    print(df_in_test.columns.tolist())
    print(f"Number of rows = {len(df_in_test)}")

    print("\n=== Output CSV ===")
    print(df_out_test.columns.tolist())
    print(f"Number of rows = {len(df_out_test)}")


    # unit parsing
    for col in ["time", "Cc1", "Cc2", "Vctrl"]:
        df_out_test[col] = df_out_test[col].apply(unit_parser)

    # Rebuild output timestamps
    OUTPUT_DT = 100e-12  # 100 ps

    df_out_test = rebuild_fixed_timestamps(
        df_out_test,
        dt=OUTPUT_DT,
        t0=0.0,
        time_col="time"
    )
        

    # data verification
    print("Columns:")
    for i, col in enumerate(df_out_test.columns):
        print(i, repr(col))

    print("\nShape:", df_out_test.shape)

    print("\nFirst 10 rows:")
    print(df_out_test.head(10).to_string())

    print("\n=== Time diagnostics ===")

    t = df_out_test["time"].to_numpy()

    print("Number of output samples:", len(t))
    print("t min =", t.min(), "s")
    print("t max =", t.max(), "s")

    dt = np.diff(t)

    print("dt mean =", np.mean(dt), "s")
    print("dt median =", np.median(dt), "s")
    print("dt min =", np.min(dt), "s")
    print("dt max =", np.max(dt), "s")
    print(f"Output sampling rate = {1/dt/1e9} GHz")

    print("Equivalent sampling rate:",
          1 / np.median(dt) / 1e9, "GHz")


    #fc = 32e9
    fc = 500.0e6
    Tc = 1 / fc

    print("Carrier period:", Tc * 1e12, "ps")
    print("Samples per cycle:", Tc / np.median(dt))

    print("\n=== Input timing ===")

    print(df_in_test.head(20))
    print(df_in_test.tail(20))
    print("Number of input rows:", len(df_in_test))

    
    # 3. Extract valid symbol information
    # Only use valid input samples
    #df_valid = df_in[df_in["region"] == "valid"].copy()
    df_valid = df_in_test.copy()

    # One row per symbol is enough for label/timing information.
    # sampleWithinSymbol == 0 gives the first input sample of each symbol.
    symbols = (
        #df_valid[df_valid["sampleWithinSymbol"] == 0]
        #.sort_values("validSymbolIndex")
        df_valid
        .reset_index(drop=True)
    )

    print("\n=== Symbol information ===")
    print(f"Number of valid symbols = {len(symbols)}")

    print(
        symbols[
            [
                #"validSymbolIndex",
                #"time_s",
                #"classIndex",
                #"bits",
                #"phaseDeg"
                'startTime_s',
                "classIndex",
                'bits',
                #'phaseDeg'
            ]
        ].head()
    )

    # Symbol start times
    #symbol_times = symbols["time_s"].to_numpy(dtype=float)
    symbol_times = symbols["startTime_s"].to_numpy(dtype=float)

    Ts_check = np.median(np.diff(symbol_times))
    dt_out = np.median(dt)
    print(f"Symbol period          = {Ts_check*1e9} ns")
    print(f"Output pts/symbol      = {Ts_check/dt}")


    # Labels
    class_labels = symbols["classIndex"].to_numpy(dtype=int)
    #phase_labels = symbols["phaseDeg"].to_numpy(dtype=float)

    print(
        f"\nFirst valid symbol time = "
        f"{symbol_times[0] * 1e9:.6f} ns"
    )

    print(
        f"Last valid symbol start = "
        f"{symbol_times[-1] * 1e9:.6f} ns"
    )


    # 4. Load continuous reservoir waveform
    t_out = df_out_test["time"].to_numpy(dtype=float)
    Cc1 = df_out_test["Cc1"].to_numpy(dtype=float)
    Cc2 = df_out_test["Cc2"].to_numpy(dtype=float)

    # plot input/output alignment
    plot_time_alignment(
        df_valid,
        t_out,
        Cc1,
        Cc2,
        n_symbols=10,
    )

    print("\n=== Continuous output ===")
    print(f"Start time = {t_out[0] * 1e9:.6f} ns")
    print(f"End time   = {t_out[-1] * 1e9:.6f} ns")

    dt_out = np.diff(t_out)

    print(
        f"Median output dt = "
        f"{np.median(dt_out) * 1e12:.3f} ps"
    )







    print("\n=== Output data integrity ===")

    print("Total rows:", len(df_out_test))
    print("Unique timestamps:", df_out_test["time"].nunique())
    print("Duplicated timestamps:", df_out_test["time"].duplicated().sum())

    print("Cc1 NaN:", df_out_test["Cc1"].isna().sum())
    print("Cc2 NaN:", df_out_test["Cc2"].isna().sum())
    print("time NaN:", df_out_test["time"].isna().sum())

    bad = (
        df_out_test["Cc1"].isna()
        | df_out_test["Cc2"].isna()
        | df_out_test["time"].isna()
    )

    print("\nFirst bad rows:")
    print(df_out_test.loc[bad].head(20))

    print("\nRows around first bad row:")
    if bad.any():
        i = np.flatnonzero(bad.to_numpy())[0]
        print(df_out_test.iloc[max(0, i-5):i+6])


    t = df_out_test["time"].to_numpy()

    dt_all = np.diff(t)

    print("\n=== dt structure ===")
    print("dt == 0:", np.sum(dt_all == 0))
    print("dt < 0 :", np.sum(dt_all < 0))
    print("dt > 0 :", np.sum(dt_all > 0))

    positive_dt = dt_all[dt_all > 0]

    print("Median positive dt:",
          np.median(positive_dt) * 1e12, "ps")

    vals, counts = np.unique(np.round(dt_all * 1e12, 6),
                         return_counts=True)

    idx = np.argsort(counts)[::-1][:10]

    print("\nMost common dt values:")
    for v, c in zip(vals[idx], counts[idx]):
        print(f"{v:12.6f} ps : {c}")

















    # 6. Project all 200 symbols
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


    # 7. Sanity check
    print("\n=== I/Q projection sanity check ===")

    print(f"I1 shape = {I1.shape}")
    print(f"Q1 shape = {Q1.shape}")
    print(f"I2 shape = {I2.shape}")
    print(f"Q2 shape = {Q2.shape}")

    print(f"Cc1 NaN count = {np.sum(~np.isfinite(I1) | ~np.isfinite(Q1))}")
    print(f"Cc2 NaN count = {np.sum(~np.isfinite(I2) | ~np.isfinite(Q2))}")

    print("\nPhase distribution:")
    for c in np.unique(class_labels):
        print(
            #f"{phase:6.1f} deg : "
            f"{np.sum(class_labels == c)} symbols"
        )


    # 9. Plot Cc1 and Cc2 constellation
    plot_constellation(
        I1,
        Q1,
        class_labels,
        "Cc1"   
    )

    plot_constellation(
        I2,
        Q2,
        class_labels,
        "Cc2"
    )




    ###########################################
    # for given K
    ###########################################

    #K = 8

    symbol_starts = df_valid["startTime_s"].to_numpy(dtype=float)
    symbol_ends   = df_valid["endTime_s"].to_numpy(dtype=float)

    # Avoid sampling exactly at the next symbol boundary.
    # For K=4, fractions = [0, 0.25, 0.5, 0.75]
    fractions = np.arange(K) / K

    sample_times = (
        symbol_starts[:, None]
        + fractions[None, :]
        * (symbol_ends - symbol_starts)[:, None]
    )

    # Labels: one label per symbol
    y = df_valid["classIndex"].to_numpy(dtype=int)

    print("sample_times:", sample_times.shape)
    print("y:", y.shape)
    print("classes:", np.unique(y, return_counts=True))

    print("\nFirst symbol:")
    print("start =", symbol_starts[0] * 1e9, "ns")
    print("end   =", symbol_ends[0] * 1e9, "ns")
    print("sample times =", sample_times[0] * 1e9, "ns")


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


    # plot K sample time
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


    # plot reservoir constellation
    mu_test = X.mean(axis=0)
    std_test = X.std(axis=0)

    std_test[std_test < 1e-12] = 1.0

    X_n = (X - mu_test) / std_test


    pca = PCA(n_components=2)
    Z = pca.fit_transform(X_n)

    print(
        "Explained variance:",
        pca.explained_variance_ratio_
    )


    #phase_deg = np.array([45, 135, 225, 315])

    plt.figure(figsize=(7, 7))

    for c in range(16):
        mask = (y == c)

        plt.scatter(
            Z[mask, 0],
            Z[mask, 1],
            alpha=0.7,
            label=rf"$class={class_labels[c]}^\circ$"
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

    poly = PolynomialFeatures(
        degree=4,
        include_bias=False
    )
        
    X_poly = poly.fit_transform(X)
    

    X_test = X_poly
    y_test = y


    #std_test = np.where(std_test < 1e-12, 1.0, std_test)

    X_test_n  = (X_test  - mu) / std


    test_pred, test_scores = predict_linear_readout(
        X_test_n,
        Wout
    )

    print(
        f"Test accuracy = "
        f"{np.mean(test_pred == y_test)*100:.2f}%"
    )


    #cm = confusion_matrix(y_test, test_pred)
    #print(cm)


    '''pca = PCA(n_components=3)
    Z = pca.fit_transform(X_n)

    print("Explained variance:")
    print(pca.explained_variance_ratio_)
    print("Total:", pca.explained_variance_ratio_.sum())

    pairs = [
        (0, 1),
        (0, 2),
        (1, 2),
    ]

    for a, b in pairs:

        plt.figure(figsize=(7, 7))

        for c in range(16):
            mask = (y == c)

            plt.scatter(
                Z[mask, a],
                Z[mask, b],
                s=15,
                alpha=0.5,
                label=rf"$class={class_labels[c]}^\circ$"
            )

        plt.xlabel(f"PC{a+1}")
        plt.ylabel(f"PC{b+1}")
        plt.title(f"PC{a+1} vs PC{b+1}")
        plt.legend()
        plt.grid(True)
        plt.show()'''

    class_to_bits = {}

    for c, b in zip(df_in_test['classIndex'], df_in_test['bits']):
        b = normalize_bits(b, 16)

        if c in class_to_bits:
            assert class_to_bits[c] == b, \
                f"Inconsistent bit mapping for class {c}"
        else:
            class_to_bits[c] = b

    print(class_to_bits)

    ber, bit_errors, total_bits = calculate_ber(
        y_test,
        test_pred,
        class_to_bits
    )

    acc = np.mean(y_test == test_pred)

    print(f"Accuracy   = {100*acc:.2f}%")
    print(f"BER        = {ber:.6e}")
    print(f"Bit errors = {bit_errors}/{total_bits}")
    


if __name__=="__main__":
    main()
