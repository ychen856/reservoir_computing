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


    '''def build_mpsk_waveform(
        self,
        phase,
        t_input,
        t_center,
        sigma,
        Iin_peak,
        fc,
        noise_enabled=False,
        snr_db=20.0,
        seed=1234
    ):

        # --------------------------------------------------------
        # clean waveform
        # --------------------------------------------------------

        u_clean = phase_modulated_pulse(
            t=t_input,
            center=t_center,
            sigma=sigma,
            I_peak=Iin_peak,
            fc=fc,
            phase=phase
        )

        # --------------------------------------------------------
        # region used for SNR calculation
        # --------------------------------------------------------

        power_mask = (
            np.abs(t_input - t_center)
            <= 3.0 * sigma
        )

        # --------------------------------------------------------
        # optional AWGN
        # --------------------------------------------------------

        if noise_enabled:

            rng = np.random.default_rng(seed)

            (
                u_noisy,
                noise,
                signal_power,
                noise_power_target
            ) = add_awgn(
                signal=u_clean,
                snr_db=snr_db,
                rng=rng,
                power_mask=power_mask
            )

        else:

            u_noisy = u_clean.copy()

            noise = np.zeros_like(u_clean)

            signal_power = np.mean(
                u_clean[power_mask]**2
            )

            noise_power_target = 0.0

        return {
            "t": t_input,
            "clean": u_clean,
            "noisy": u_noisy,
            "noise": noise,
            "signal_power": signal_power,
            "noise_power_target": noise_power_target,
            "power_mask": power_mask
        }'''