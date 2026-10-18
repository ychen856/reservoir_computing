import numpy as np

def adc_zoh_input(t, t_start, dt_adc, u_adc_quant):
    idx = np.floor(
        (t - t_start) / dt_adc
    ).astype(int)

    idx = np.clip(
        idx,
        0,
        len(u_adc_quant) - 1
    )

    return u_adc_quant[idx]  


def sample_reservoir_states(
    sol,
    sample_times,
    state_indices=(0, 1),
):
    """
    Sample selected reservoir states.

    Returns
    -------
    feature : ndarray
        Interleaved temporal features.

        [x0(t1), x1(t1), ..., x0(t2), x1(t2), ...]
    """

    sample_times = np.asarray(sample_times)

    sampled = []

    for idx in state_indices:
        sampled.append(
            np.interp(
                sample_times,
                sol.t,
                sol.y[idx],
            )
        )

    return np.column_stack(
        sampled
    ).reshape(-1)