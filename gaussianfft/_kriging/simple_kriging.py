import numpy as np
import scipy.linalg
import scipy.ndimage

import gaussianfft


class SimpleKriging:
    def __init__(self, variogram, nx, dx, ny, dy, nz, dz,
                 obs_locations, obs_values, obs_uncertainties, *, mean=0.0):
        self.variogram = variogram
        self.nx = nx
        self.dx = dx
        self.ny = ny
        self.dy = dy
        self.nz = nz
        self.dz = dz
        self.mean = mean

        self.obs_locations = np.asarray(obs_locations, dtype=float)
        self.obs_values = np.asarray(obs_values, dtype=float)
        self.obs_uncertainties = np.asarray(obs_uncertainties, dtype=float)

        self.n_obs = len(self.obs_locations)
        self.ndims = 3 if nz > 1 else (2 if ny > 1 else 1)
        self._grid_shape = (nx, ny, nz)[:self.ndims]

        self._validate_inputs()

        self._cho_factor = self._build_obs_covariance()
        self._grid_to_obs_cov = self._build_grid_to_obs_covariance()
        self._compute_obs_indices()

    def _validate_inputs(self):
        if self.obs_locations.ndim != 2:
            raise ValueError(
                f"obs_locations must be a 2D array with shape (N, {self.ndims}), got {self.obs_locations.shape}"
            )
        if self.obs_locations.shape[1] != self.ndims:
            raise ValueError(
                f"obs_locations must have {self.ndims} columns, got {self.obs_locations.shape[1]}"
            )
        n = self.obs_locations.shape[0]
        if self.obs_values.shape != (n,):
            raise ValueError(f"obs_values must have shape ({n},), got {self.obs_values.shape}")
        if self.obs_uncertainties.shape != (n,):
            raise ValueError(f"obs_uncertainties must have shape ({n},), got {self.obs_uncertainties.shape}")
        if np.any(self.obs_uncertainties < 0):
            raise ValueError("obs_uncertainties must be non-negative")
        sizes = np.array((self.nx, self.ny, self.nz)[:self.ndims])
        steps = np.array((self.dx, self.dy, self.dz)[:self.ndims])
        grid_max = (sizes - 1) * steps
        if np.any(self.obs_locations < 0) or np.any(self.obs_locations > grid_max):
            raise ValueError("obs_locations has values outside grid bounds")

    def _corr(self, dist):
        if self.ndims == 1:
            return self.variogram.corr(dist[0])
        elif self.ndims == 2:
            return self.variogram.corr(dist[0], dist[1])
        else:
            return self.variogram.corr(dist[0], dist[1], dist[2])

    def _build_obs_covariance(self):
        n = self.n_obs
        cov = np.empty((n, n))
        for i in range(n):
            for j in range(i, n):
                dist = self.obs_locations[i] - self.obs_locations[j]
                c = self._corr(dist)
                cov[i, j] = c
                cov[j, i] = c
        cov[np.diag_indices(n)] += self.obs_uncertainties ** 2 + 1e-10
        try:
            return scipy.linalg.cho_factor(cov)
        except scipy.linalg.LinAlgError as exc:
            raise ValueError(
                "Failed to factorize observation covariance matrix. "
                "The matrix is likely not positive definite. This can happen for "
                "nearly duplicate observation locations with too small observation "
                "uncertainties. Consider increasing obs_uncertainties or removing "
                "duplicate/near-duplicate observations."
            ) from exc

    def _build_grid_to_obs_covariance(self):
        grid_coords = self._grid_coordinates()
        n_grid = grid_coords.shape[0]
        n_obs = self.n_obs
        cov = np.empty((n_grid, n_obs))
        for j in range(n_obs):
            dists = grid_coords - self.obs_locations[j]
            for i in range(n_grid):
                cov[i, j] = self._corr(dists[i])
        return cov

    def _grid_coordinates(self):
        x = np.arange(self.nx) * self.dx
        if self.ndims == 1:
            return x.reshape(-1, 1)
        y = np.arange(self.ny) * self.dy
        if self.ndims == 2:
            xx, yy = np.meshgrid(x, y, indexing='ij')
            return np.column_stack([xx.ravel(), yy.ravel()])
        z = np.arange(self.nz) * self.dz
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    def _compute_obs_indices(self):
        """Precompute observation coordinates in index space for interpolation."""
        steps = np.array((self.dx, self.dy, self.dz)[:self.ndims])
        self._obs_coords = (self.obs_locations / steps).T

    def _interpolate_at_obs(self, field):
        """Linear interpolation of field at observation locations."""
        return scipy.ndimage.map_coordinates(field, self._obs_coords, order=1)

    def predict(self):
        if isinstance(self.mean, np.ndarray):
            mean_at_obs = self._interpolate_at_obs(self.mean)
            mean_flat = self.mean.ravel()
        else:
            mean_at_obs = self.mean
            mean_flat = self.mean

        residuals = self.obs_values - mean_at_obs
        weights = scipy.linalg.cho_solve(self._cho_factor, residuals)

        kriging_mean = mean_flat + self._grid_to_obs_cov @ weights
        kriging_mean = kriging_mean.reshape(self._grid_shape)

        # Kriging variance: C(0) - k^T C^{-1} k for each grid point
        # C(0) = 1 (unit sill from variogram correlation)
        alpha = scipy.linalg.cho_solve(self._cho_factor, self._grid_to_obs_cov.T)
        kriging_variance = 1.0 - np.sum(self._grid_to_obs_cov * alpha.T, axis=1)
        kriging_variance = np.maximum(kriging_variance, 0.0)
        kriging_variance = kriging_variance.reshape(self._grid_shape)
        kriging_stdev = np.sqrt(kriging_variance)

        return kriging_mean, kriging_stdev

    def simulate(self, n_sim=1):
        results = []
        for _ in range(n_sim):
            # Generate unconditional field using gaussianfft
            if self.ndims == 1:
                uncond = gaussianfft.simulate(self.variogram, self.nx, self.dx)
            elif self.ndims == 2:
                uncond = gaussianfft.simulate(self.variogram, self.nx, self.dx, self.ny, self.dy)
            else:
                uncond = gaussianfft.simulate(
                    self.variogram, self.nx, self.dx, self.ny, self.dy, self.nz, self.dz
                )
            uncond = np.array(uncond).reshape(self._grid_shape, order='F')

            # Extract unconditional values at observation locations
            uncond_at_obs = self._interpolate_at_obs(uncond)

            # Krige the residual between observations and unconditional at obs locations
            obs_residuals = (self.obs_values - self.mean) - uncond_at_obs
            weights = scipy.linalg.cho_solve(self._cho_factor, obs_residuals)
            correction = (self._grid_to_obs_cov @ weights).reshape(self._grid_shape)

            conditioned = self.mean + uncond + correction
            results.append(conditioned)
        return results
