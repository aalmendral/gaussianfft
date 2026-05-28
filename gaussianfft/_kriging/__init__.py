from gaussianfft._kriging.simple_kriging import SimpleKriging

__all__ = ['predict', 'simulate']


def predict(variogram, nx, dx, ny, dy, nz, dz,
            obs_locations, obs_values, obs_uncertainties, *, mean=0.0):
    sk = SimpleKriging(variogram, nx, dx, ny, dy, nz, dz,
                       obs_locations, obs_values, obs_uncertainties, mean=mean)
    return sk.predict()


def simulate(variogram, nx, dx, ny, dy, nz, dz,
             obs_locations, obs_values, obs_uncertainties, *, mean=0.0, n=1):
    sk = SimpleKriging(variogram, nx, dx, ny, dy, nz, dz,
                       obs_locations, obs_values, obs_uncertainties, mean=mean)
    return sk.simulate(n=n)
