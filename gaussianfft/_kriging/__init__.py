from gaussianfft._kriging.simple_kriging import SimpleKriging

__all__ = ['predict', 'simulate']


def _parse_grid_and_obs(args, kwargs):
    """Parse positional/keyword args: [ny, dy, [nz, dz,]] obs_locations, obs_values, obs_uncertainties."""
    ny = kwargs.pop('ny', None)
    dy = kwargs.pop('dy', None)
    nz = kwargs.pop('nz', None)
    dz = kwargs.pop('dz', None)

    if ny is not None:
        # Grid dims passed as keyword arguments; all positional args are obs
        if dy is None:
            raise TypeError("dy must be provided when ny is given")
        if nz is None:
            nz = 1
        if dz is None:
            dz = 1.0
        if len(args) != 3:
            raise TypeError(
                "When ny/dy/nz/dz are keyword arguments, exactly 3 positional args "
                "(obs_locations, obs_values, obs_uncertainties) are expected"
            )
        obs_locations, obs_values, obs_uncertainties = args
    else:
        # All positional
        if len(args) == 3:
            ny, dy, nz, dz = 1, 1.0, 1, 1.0
            obs_locations, obs_values, obs_uncertainties = args
        elif len(args) == 5:
            ny, dy = args[0], args[1]
            nz, dz = 1, 1.0
            obs_locations, obs_values, obs_uncertainties = args[2], args[3], args[4]
        elif len(args) == 7:
            ny, dy, nz, dz = args[0], args[1], args[2], args[3]
            obs_locations, obs_values, obs_uncertainties = args[4], args[5], args[6]
        else:
            raise TypeError(
                "predict/simulate expects grid args followed by "
                "obs_locations, obs_values, obs_uncertainties. "
                "Valid patterns: (variogram, nx, dx, obs...), "
                "(variogram, nx, dx, ny, dy, obs...), "
                "(variogram, nx, dx, ny, dy, nz, dz, obs...)"
            )
    return ny, dy, nz, dz, obs_locations, obs_values, obs_uncertainties


def predict(variogram, nx, dx, *args, mean=0.0, **kwargs):
    ny, dy, nz, dz, obs_locations, obs_values, obs_uncertainties = _parse_grid_and_obs(args, kwargs)
    sk = SimpleKriging(variogram, nx, dx, ny, dy, nz, dz,
                       obs_locations, obs_values, obs_uncertainties, mean=mean)
    return sk.predict()


def simulate(variogram, nx, dx, *args, mean=0.0, n_sim=1, **kwargs):
    ny, dy, nz, dz, obs_locations, obs_values, obs_uncertainties = _parse_grid_and_obs(args, kwargs)
    sk = SimpleKriging(variogram, nx, dx, ny, dy, nz, dz,
                       obs_locations, obs_values, obs_uncertainties, mean=mean)
    return sk.simulate(n_sim=n_sim)