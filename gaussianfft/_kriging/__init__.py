from gaussianfft._kriging.simple_kriging import SimpleKriging

__all__ = ['predict', 'simulate']


def predict(variogram, nx, dx, obs_locations, obs_values, obs_uncertainties, *,
             ny=1, dy=1.0, nz=1, dz=1.0, mean=0.0):
    sk = SimpleKriging(variogram, nx, dx, obs_locations, obs_values, obs_uncertainties,
                       ny=ny, dy=dy, nz=nz, dz=dz, mean=mean)
    return sk.predict()


def simulate(variogram, nx, dx, obs_locations, obs_values, obs_uncertainties, *,
             ny=1, dy=1.0, nz=1, dz=1.0, mean=0.0, n=1):
    sk = SimpleKriging(variogram, nx, dx, obs_locations, obs_values, obs_uncertainties,
                       ny=ny, dy=dy, nz=nz, dz=dz, mean=mean)
    return sk.simulate(n=n)
