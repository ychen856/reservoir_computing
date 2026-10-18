class BaseReservoir:
    def simulate(
        self,
        input_func,
        t_span,
        t_eval,
        x0=None,
    ):
        raise NotImplementedError