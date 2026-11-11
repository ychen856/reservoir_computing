"""Trial waveforms and per-dimer features for Passive-GQC applications."""

import numpy as np

from src.tasks.applications.mpsk import get_mpsk_phases, phase_modulated_pulse
from src.tasks.applications.waveform_classification import (
    EXPERIMENTS, generate_waveform_classification,
)


def mpsk_trials(a, order, condition, frequency, seed):
    rng = np.random.default_rng(seed)
    period = 1 / frequency
    t = np.linspace(0, 24 * period, 24 * a.samples_per_cycle + 1)
    center = 12 * period
    sigma = (8 / 6) * period
    phases = get_mpsk_phases(order)
    labels = np.repeat(np.arange(order), a.repeats)
    waves = phase_modulated_pulse(
        t[None, :], center, sigma, a.amplitude, frequency, phases[labels, None])
    if condition == "isi":
        # Observe the current symbol in the presence of one prior symbol,
        # delayed by the symbol spacing and scaled by the channel tap.
        previous = rng.integers(order, size=len(labels))
        waves += a.isi_tap * phase_modulated_pulse(
            t[None, :], center - a.isi_spacing_cycles * period,
            sigma, a.amplitude, frequency, phases[previous, None])
    elif condition == "awgn":
        mask = np.abs(t - center) <= 3 * sigma
        power = np.mean(waves[:, mask] ** 2, axis=1, keepdims=True)
        waves += rng.normal(size=waves.shape) * np.sqrt(power * 10**(-a.snr_db / 10))
    return waves, labels, t, (center, t[-1])


def shape_trials(a, experiment, frequency, seed):
    data = generate_waveform_classification(
        n_per_class=a.repeats, n_points=24 * a.samples_per_cycle + 1,
        duration=24 / frequency, seed=seed, **EXPERIMENTS[experiment])
    t = data.metadata["t_eval"]
    return a.amplitude * data.u, data.y, t, (t[0], t[-1])


def waveform_features(case, waves, t, observation, a):
    """Reset each trial, evolving every lane continuously between switch edges."""
    reservoir = case.reservoir()
    dimers = reservoir.dimers if case.n_dimers > 1 else (reservoir,)
    times = np.linspace(*observation, a.k)
    switch_period = a.cycles_per_symbol / case.frequency
    interleaved = case.n_dimers > 1 and case.routing == "interleaved"
    max_step = np.min(np.diff(t))
    features = []
    cache = {}

    for wave in waves:
        key = wave.tobytes()
        if key in cache:
            features.append(cache[key])
            continue
        def drive(time):
            return np.interp(time, t, wave)

        if interleaved:
            # The integer slot selects the lane before integration starts.
            edges = t[0] + switch_period * np.arange(1, int(np.ceil((t[-1]-t[0])/switch_period)))
            edges = np.r_[t[0], edges[edges < t[-1]], t[-1]]
            blocks = np.empty((a.k, len(dimers), 2))
            initial = [np.zeros(dimer.state_size) for dimer in dimers]
            for slot, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
                lo = np.searchsorted(times, left)
                hi = np.searchsorted(times, right, side="right" if right == t[-1] else "left")
                evaluation = np.unique(np.r_[times[lo:hi], right])
                for lane, dimer in enumerate(dimers):
                    current = drive if lane == slot % len(dimers) else zero_current
                    sol = dimer.simulate(current, (left, right), evaluation,
                                         x0=initial[lane], max_step=max_step)
                    initial[lane] = sol.y[:, -1]
                    blocks[lo:hi, lane] = sol.y[:2, :hi-lo].T
                if case.ordering == "age":
                    order = (slot - np.arange(len(dimers))) % len(dimers)
                    blocks[lo:hi] = blocks[lo:hi, order, :]
        else:
            # Broadcast lanes share parameters, input, and initial state.
            sol = dimers[0].simulate(drive, (t[0], t[-1]), times, max_step=max_step)
            blocks = np.repeat(sol.y[:2].T[:, None, :], len(dimers), axis=1)
        feature = blocks.reshape(-1)
        if not np.isfinite(feature).all():
            raise RuntimeError("Nonfinite waveform features")
        cache[key] = feature
        features.append(feature)
    raw = np.asarray([np.interp(times, t, wave) for wave in waves])
    return np.asarray(features), raw, times


def zero_current(_time):
    return 0.0
