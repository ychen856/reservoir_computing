import numpy as np

class input_wave:
    def __init__(self, Iin_peak, sigma, t_start, t_end, t_center, t_eval):
        self.Iin_peak = Iin_peak
        self.sigma = sigma
        self.t_center = t_center
        self.t_prev = 0.30e-6

        self.t_start = t_start
        self.t_end = t_end
        self.t_eval = t_eval

    def gaussian_pulse(self, t, center, sigma, amplitude):
        return amplitude * np.exp(
            -0.5 * ((t - center) / sigma)**2
        )


    def input_history_A(self, t):
        return (
            self.gaussian_pulse(t, self.t_prev, self.sigma, self.Iin_peak)
            + self.gaussian_pulse(t, self.t_center, self.sigma, self.Iin_peak)
        )


    def input_history_B(self, t):
        return self.gaussian_pulse(
            t, self.t_center, self.sigma, self.Iin_peak
        )