from .common import NoiseScheduler


class ContinuousNoiseScheduler(NoiseScheduler):
    """Base for schedulers whose timestep is a continuous variable, not an index.

    Subclasses interpret ``steps`` however their parameterization requires (a
    [0, 1] progress variable, a raw sigma, ...) and implement ``get_rates``.
    ``timesteps`` is kept for interface compatibility with the discrete
    schedulers — it is the upper end of the timestep domain (1.0 by default,
    i.e. "fully noised") and doubles as the switch that makes
    ``generate_timesteps`` sample a float in [0, timesteps) instead of an
    integer index into a table. Continuous schedulers have no beta table, so
    the discrete-API methods (``get_posterior_mean``/``get_posterior_variance``)
    stay unimplemented; sampling goes through ``get_rates``.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(timesteps=1, *args, **kwargs)
