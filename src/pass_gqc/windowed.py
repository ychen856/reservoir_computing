"""RF window tasks ported from the PT parallel integral and Fourier runners."""

from dataclasses import replace

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def encode(a, levels):
    """Uniform bins over [-1,1], bin-centre reconstruction; no duplicated 2pi."""
    indices = np.minimum(((np.clip(a, -1, 1)+1)*levels/2).astype(int), levels-1)
    return -1+(indices+0.5)*2/levels, indices*2*np.pi/levels


def modulation_values(a, encoding, levels, amplitude_values, current, depth):
    """Return encoded numerical values, RF phase and peak current."""
    q, phases = encode(a, levels)
    if encoding == 'phase':
        return q, phases, np.full(len(a), current)
    values = a.copy() if amplitude_values == 'continuous' else q
    return values, np.zeros(len(a)), current*(1+depth*values)


def targets(q, window, ts):
    """Row n is read at (n+1)Ts and includes q[n]. ZOH-exact integrals.

    I2 = integral_a^b (b-t)q(t)dt, a=max(0,b-window*Ts).
    This is NOT two moving sums (which would have support 2L-1).
    """
    first = np.convolve(q, np.ones(window), mode='full')[:len(q)]*ts
    second = np.convolve(q, np.arange(window)+0.5, mode='full')[:len(q)]*ts**2
    return np.column_stack((first, second))


def fit_predict(X, Y, train, val, test, lambdas):
    """Training-only scaling; validation-only ridge selection; frozen test readout."""
    mean, scale = X[train].mean(0), X[train].std(0)
    scale = np.where(scale > 1e-14, scale, 1)
    Z = (X-mean)/scale
    ym, ys = Y[train].mean(0), Y[train].std(0)
    ys = np.where(ys > 0, ys, 1)
    target = (Y-ym)/ys
    zm = Z[train].mean(0)
    A, B = Z[train]-zm, target[train]-target[train].mean(0)
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    projected = U.T @ B
    validation_features = Z[val] - zm
    target_mean = target[train].mean(0)
    best_score = np.inf
    for ridge in lambdas:
        weight = (Vh.T*(s/(s*s+ridge))) @ projected
        residual = validation_features @ weight + target_mean - target[val]
        score = np.mean(residual**2)
        if score < best_score:
            best_score, lam, w = score, ridge, weight
    bias = target_mean - zm @ w
    pred = (Z @ w + bias)*ys + ym
    metrics = {}
    for label, sl in [('train', train), ('validation', val), ('test', test)]:
        mse = np.mean((pred[sl]-Y[sl])**2, axis=0)
        variance = np.var(Y[sl], axis=0)
        metrics[label+'_nmse'] = (mse/np.maximum(variance, 1e-300)).tolist()
    model = dict(mean=mean, scale=scale, y_mean=ym, y_scale=ys, w=w, bias=bias)
    return pred, lam, metrics, model


def sliding_dft(values, length):
    """Row n contains the rectangular window ending at symbol n."""
    history = sliding_window_view(np.pad(values, (length - 1, 0)), length)
    return np.fft.rfft(history, axis=1) / length


def complex_nrmse(target, prediction, length):
    """Return per-bin error and Parseval-weighted full-spectrum error."""
    error = np.mean(np.abs(prediction - target) ** 2, axis=0)
    variance = np.mean(np.abs(target - target.mean(axis=0)) ** 2, axis=0)
    ratio = np.full_like(error, np.nan)
    np.divide(error, variance, out=ratio, where=variance > 0)
    weights = np.full(error.shape, 2.0)
    weights[0] = 1
    if length % 2 == 0:
        weights[-1] = 1
    denominator = weights @ variance
    total = np.sqrt((weights @ error) / denominator) if denominator > 0 else np.nan
    return np.sqrt(ratio), total


def rf_features(case, phases, amplitudes, ts, k, steps_per_cycle):
    """Integrate a continuous RF carrier, keeping all states at switch edges."""
    reservoir = case.reservoir()
    coupled = case.coupling_l_nh is not None
    count = case.n_dimers
    dimers = reservoir.dimers if count > 1 else (reservoir,)
    # Each coupled pair is electrically independent of the other pairs.
    branches = ([replace(reservoir, dimers=dimers[i:i+2]) for i in range(0, count, 2)]
                if coupled else dimers)
    lanes_per_branch = 2 if coupled else 1
    sample_times = ts*np.arange(1, k+1)/(k+1)
    evaluation = np.r_[sample_times, ts]
    omega = 2*np.pi*case.frequency
    angle = np.remainder(omega*ts*np.arange(len(phases)) + phases, 2*np.pi)
    raw = amplitudes[:, None]*np.cos(angle[:, None] + omega*sample_times)
    blocks = np.empty((len(phases), count, k, 2))
    max_step = min(ts/8, 1/(case.frequency*steps_per_cycle))

    def zero_current(_):
        return 0.0

    for index, branch in enumerate(branches):
        start = index*lanes_per_branch
        stop = start+lanes_per_branch
        voltage_indices = list(branch.voltage_state_indices) if coupled else [0, 1]
        state = np.zeros(branch.state_size)
        symbol = 0
        while symbol < len(phases):
            lane = symbol % count
            active = start <= lane < stop
            if active:
                phase, amplitude = angle[symbol], amplitudes[symbol]

                def carrier(t):
                    return amplitude*np.cos(omega*t + phase)

                kwargs = {'active_lane': lane-start} if coupled else {}
                result = branch.simulate(carrier, (0, ts), evaluation,
                                         x0=state, max_step=max_step, **kwargs)
                end = symbol+1
            else:
                # During an idle interval there is no RF drive or switch discontinuity.
                idle = min((start-lane) % count, len(phases)-symbol)
                end = symbol+idle
                times = (np.arange(idle)[:, None]*ts + sample_times).ravel()
                boundary = idle*ts
                kwargs = {'active_lane': 0} if coupled else {}
                result = branch.simulate(zero_current, (0, boundary), np.r_[times, boundary],
                                         x0=state, **kwargs)
            state = result.y[:, -1]
            voltages = result.y[voltage_indices, :-1]
            blocks[symbol:end, start:stop] = voltages.reshape(
                lanes_per_branch, 2, end-symbol, k).transpose(2, 0, 3, 1)
            symbol = end

    ages = (np.arange(len(phases))[:, None]-np.arange(count)) % count
    blocks = blocks[np.arange(len(phases))[:, None], ages]
    if not np.isfinite(blocks).all():
        raise RuntimeError("Nonfinite RF reservoir features")
    return blocks.reshape(len(phases), -1), raw
