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