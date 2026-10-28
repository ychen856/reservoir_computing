"""Estimate a voltage-envelope decay time during unforced relaxation."""
from dataclasses import dataclass

import numpy as np
from scipy.signal import hilbert


@dataclass
class VoltageDecayResult:
    tau_seconds: float
    slope: float
    r_squared: float
    n_fit: int


def evaluate_voltage_decay(t, v, fit_start, fit_end, floor_fraction=0.02):
    """Fit log(|Hilbert(v)|) = intercept - t/tau in a free-decay interval.

    The caller must ensure that the drive is exactly zero throughout the fit.
    A nonnegative slope yields tau=nan rather than a negative decay time.
    """
    t = np.asarray(t, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    if (t.size != v.size or t.size < 30 or
            not np.all(np.isfinite(t)) or not np.all(np.isfinite(v)) or
            np.any(np.diff(t) <= 0)):
        raise ValueError("Invalid waveform: expected finite, increasing t and matching v")
    if not (t[0] <= fit_start < fit_end <= t[-1]):
        raise ValueError("Fit interval must lie inside the simulated time range")
    if not (0 <= floor_fraction < 1):
        raise ValueError("floor_fraction must be in [0, 1)")

    envelope = np.abs(hilbert(v))
    interval = (t >= fit_start) & (t <= fit_end)
    if np.count_nonzero(interval) < 20:
        raise ValueError("Too few points in fit interval")
    floor = max(1e-15, floor_fraction * envelope[interval].max())
    mask = interval & (envelope > floor)
    if np.count_nonzero(mask) < 20:
        raise ValueError("Too few envelope points above fit floor")

    xx = t[mask] - t[mask][0]
    yy = np.log(envelope[mask])
    slope, intercept = np.polyfit(xx, yy, 1)
    predicted = slope * xx + intercept
    ss_total = np.sum((yy - yy.mean()) ** 2)
    r_squared = (1 - np.sum((yy - predicted) ** 2) / ss_total
                 if ss_total > 0 else float("nan"))
    tau = -1 / slope if slope < 0 else float("nan")
    return VoltageDecayResult(float(tau), float(slope),
                              float(r_squared), int(np.count_nonzero(mask)))
