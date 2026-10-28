import numpy as np
from src.tasks.base import TaskData


def generate_mackey_glass(n_samples=3000, washout=200, train_ratio=0.8,
                          beta=0.2, gamma=0.1, exponent=10, delay=17.0,
                          dt=0.1, sample_interval=1.0, transient=1000.0,
                          horizon=1, initial_value=1.2):
    """Euler-integrated Mackey-Glass DDE, teacher-forced horizon-step prediction.

    Internal integration dt=0.1, output sampled at integer physical time units.
    Target y[k]=x[k+horizon]; input u[k]=x[k]. No test targets fed to readout.
    """
    if n_samples < 100 or horizon < 1 or dt <= 0 or sample_interval <= 0:
        raise ValueError('Invalid simulation parameters')
    lag = delay / dt
    stride = sample_interval / dt
    if abs(lag-round(lag)) > 1e-8 or abs(stride-round(stride)) > 1e-8:
        raise ValueError('delay/dt and sample_interval/dt must be integers')
    lag, stride = int(round(lag)), int(round(stride))
    transient_steps = int(round(transient/dt))
    steps = transient_steps + (n_samples + horizon + 2)*stride
    state = np.empty(steps+lag+1, dtype=float)
    state[:lag+1] = initial_value
    for i in range(lag, len(state)-1):
        delayed = state[i-lag]
        dx = beta*delayed/(1+delayed**exponent) - gamma*state[i]
        state[i+1] = state[i] + dt*dx
    start = lag + transient_steps
    series = state[start:start+(n_samples+horizon)*stride:stride]
    u = series[:n_samples].copy()
    y = series[horizon:horizon+n_samples].copy()
    return TaskData('Mackey-Glass', u, y, 'regression', washout=washout,
                    train_ratio=train_ratio, target_offset=horizon,
                    metadata={'horizon': horizon, 'delay': delay, 'dt': dt,
                              'sample_interval': sample_interval, 'transient': transient})
