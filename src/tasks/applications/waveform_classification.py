"""Four-class waveform data with controlled nuisance-factor experiments."""
import numpy as np
from scipy.signal import sawtooth
from src.tasks.base import TaskData

LABELS = ('sine', 'square', 'triangle', 'sawtooth')
EXPERIMENTS = {
    'A': dict(random_phase=False, random_amplitude=False, random_frequency=False, noise_std=0.0),
    'B': dict(random_phase=True, random_amplitude=False, random_frequency=False, noise_std=0.0),
    'C': dict(random_phase=True, random_amplitude=True, random_frequency=False, noise_std=0.0),
    'D': dict(random_phase=True, random_amplitude=True, random_frequency=True, noise_std=0.0),
    'E': dict(random_phase=True, random_amplitude=True, random_frequency=True, noise_std=0.02),
}


def generate_waveform_classification(n_per_class=40, n_points=128, cycles=3,
                                     duration=0.5e-6, noise_std=0.02,
                                     seed=42, train_ratio=0.8,
                                     random_phase=True, random_amplitude=True,
                                     random_frequency=False, phase_fixed=0.35,
                                     amplitude_range=(0.8, 1.2),
                                     frequency_range=(0.8, 1.2)):
    """Return independent trials (n_trials, n_points), one class per trial.

    The *same nuisance-factor draws* are used across all four classes, so
    amplitude/phase/frequency distributions cannot reveal the label.
    Noise is drawn independently for each trial. The physical input current
    scale is applied by the runner, not here.
    """
    if n_per_class < 5 or n_points < 32 or cycles <= 0 or duration <= 0:
        raise ValueError('Invalid trial count, sample count, cycles, or duration')
    if not 0 < train_ratio < 1 or noise_std < 0:
        raise ValueError('Invalid train_ratio or noise_std')
    if not (0 < amplitude_range[0] <= amplitude_range[1]):
        raise ValueError('Invalid amplitude_range')
    if not (0 < frequency_range[0] <= frequency_range[1]):
        raise ValueError('Invalid frequency_range')

    rng = np.random.default_rng(seed)
    t = np.linspace(0., duration, n_points, endpoint=True)
    # Matched nuisance distributions: one set of nuisance draws per trial index.
    phases = rng.uniform(0, 2*np.pi, n_per_class) if random_phase else np.full(n_per_class, phase_fixed)
    amps = rng.uniform(*amplitude_range, size=n_per_class) if random_amplitude else np.ones(n_per_class)
    freq_scales = (rng.uniform(*frequency_range, size=n_per_class)
                   if random_frequency else np.ones(n_per_class))
    waves, labels = [], []
    for c in range(len(LABELS)):
        for j in range(n_per_class):
            theta = 2*np.pi*cycles*freq_scales[j]*(t/duration) + phases[j]
            if c == 0:
                v = np.sin(theta)
            elif c == 1:
                v = np.where(np.sin(theta) >= 0., 1., -1.)
            elif c == 2:
                v = sawtooth(theta, width=0.5)
            else:
                v = sawtooth(theta, width=1.0)
            waves.append(amps[j]*v + rng.normal(0., noise_std, n_points))
            labels.append(c)
    order = rng.permutation(len(labels))
    u = np.asarray(waves, dtype=float)[order]
    y = np.asarray(labels, dtype=int)[order]
    return TaskData('Waveform classification', u, y, 'classification',
                    train_ratio=train_ratio,
                    metadata={'t_eval': t, 'labels': LABELS, 'cycles': cycles,
                              'noise_std': noise_std, 'seed': seed,
                              'random_phase': random_phase,
                              'random_amplitude': random_amplitude,
                              'random_frequency': random_frequency,
                              'frequency_range': frequency_range})
