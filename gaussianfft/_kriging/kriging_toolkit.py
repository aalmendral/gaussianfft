import numpy as np
import scipy.linalg
import scipy.ndimage

import gaussianfft

_FACTORIZE_ERROR = (
    "Failed to factorize observation covariance matrix. "
    "The matrix is likely not positive definite. This can happen for "
    "nearly duplicate observation locations with too small observation "
    "uncertainties. Consider increasing obs_uncertainties or removing "
    "duplicate/near-duplicate observations."
)


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------

def _ndims(ny, nz):
    return 3 if nz > 1 else (2 if ny > 1 else 1)


def _validate_inputs(obs_locations, obs_values, obs_uncertainties,
                     nx, dx, ny, dy, nz, dz, ndims):
    if obs_locations.ndim != 2:
        raise ValueError(
            f"obs_locations must be a 2D array with shape (N, {ndims}), got {obs_locations.shape}"
        )
    if obs_locations.shape[1] != ndims:
        raise ValueError(
            f"obs_locations must have {ndims} columns, got {obs_locations.shape[1]}"
        )
    n = obs_locations.shape[0]
    if obs_values.shape != (n,):
        raise ValueError(f"obs_values must have shape ({n},), got {obs_values.shape}")
    if obs_uncertainties.shape != (n,):
        raise ValueError(f"obs_uncertainties must have shape ({n},), got {obs_uncertainties.shape}")
    if np.any(obs_uncertainties < 0):
        raise ValueError("obs_uncertainties must be non-negative")
    sizes = np.array((nx, ny, nz)[:ndims])
    steps = np.array((dx, dy, dz)[:ndims])
    grid_max = (sizes - 1) * steps
    if np.any(obs_locations < 0) or np.any(obs_locations > grid_max):
        raise ValueError("obs_locations has values outside grid bounds")


def _corr(variogram, dist, ndims):
    if ndims == 1:
        return variogram.corr(dist[0])
    elif ndims == 2:
        return variogram.corr(dist[0], dist[1])
    else:
        return variogram.corr(dist[0], dist[1], dist[2])


def _grid_coordinates(nx, dx, ny, dy, nz, dz, ndims):
    x = np.arange(nx) * dx
    if ndims == 1:
        return x.reshape(-1, 1)
    y = np.arange(ny) * dy
    if ndims == 2:
        xx, yy = np.meshgrid(x, y, indexing='ij')
        return np.column_stack([xx.ravel(), yy.ravel()])
    z = np.arange(nz) * dz
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def _build_obs_cov_matrix(variogram, obs_locations, obs_uncertainties, ndims):
    n = len(obs_locations)
    cov = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            c = _corr(variogram, obs_locations[i] - obs_locations[j], ndims)
            cov[i, j] = c
            cov[j, i] = c
    cov[np.diag_indices(n)] += obs_uncertainties ** 2 + 1e-10
    return cov


def _build_grid_to_obs_cov(variogram, nx, dx, ny, dy, nz, dz, obs_locations, ndims):
    grid_coords = _grid_coordinates(nx, dx, ny, dy, nz, dz, ndims)
    n_grid = grid_coords.shape[0]
    n_obs = len(obs_locations)
    cov = np.empty((n_grid, n_obs))
    for j in range(n_obs):
        dists = grid_coords - obs_locations[j]
        for i in range(n_grid):
            cov[i, j] = _corr(variogram, dists[i], ndims)
    return cov


def _obs_interp_coords(obs_locations, dx, dy, dz, ndims):
    steps = np.array((dx, dy, dz)[:ndims])
    return (obs_locations / steps).T


def _interpolate_at_obs(field, obs_coords):
    return scipy.ndimage.map_coordinates(field, obs_coords, order=1)


def _simulate_uncond(variogram, nx, dx, ny, dy, nz, dz, grid_shape, ndims):
    if ndims == 1:
        raw = gaussianfft.simulate(variogram, nx, dx)
    elif ndims == 2:
        raw = gaussianfft.simulate(variogram, nx, dx, ny, dy)
    else:
        raw = gaussianfft.simulate(variogram, nx, dx, ny, dy, nz, dz)
    return np.array(raw).reshape(grid_shape, order='F')


# ---------------------------------------------------------------------------
# Kriging classes
# ---------------------------------------------------------------------------

class SimpleKriging:
    def __init__(self, variogram, nx, dx, ny, dy, nz, dz,
                 obs_locations, obs_values, obs_uncertainties, *, mean=0.0):
        self.variogram = variogram
        self.nx, self.dx = nx, dx
        self.ny, self.dy = ny, dy
        self.nz, self.dz = nz, dz
        self.mean = mean

        self.obs_locations = np.asarray(obs_locations, dtype=float)
        self.obs_values = np.asarray(obs_values, dtype=float)
        self.obs_uncertainties = np.asarray(obs_uncertainties, dtype=float)

        self.ndims = _ndims(ny, nz)
        self._grid_shape = (nx, ny, nz)[:self.ndims]
        if isinstance(self.mean, np.ndarray) and self.mean.shape != self._grid_shape:
            raise ValueError(
                f"mean must be a scalar or have shape {self._grid_shape}, got {self.mean.shape}"
            )

        _validate_inputs(self.obs_locations, self.obs_values, self.obs_uncertainties,
                         nx, dx, ny, dy, nz, dz, self.ndims)

        cov = _build_obs_cov_matrix(variogram, self.obs_locations, self.obs_uncertainties, self.ndims)
        try:
            self._cho_factor = scipy.linalg.cho_factor(cov)
        except scipy.linalg.LinAlgError as exc:
            raise ValueError(_FACTORIZE_ERROR) from exc

        self._grid_to_obs_cov = _build_grid_to_obs_cov(
            variogram, nx, dx, ny, dy, nz, dz, self.obs_locations, self.ndims
        )
        self._obs_coords = _obs_interp_coords(self.obs_locations, dx, dy, dz, self.ndims)

    def _mean_components(self):
        if isinstance(self.mean, np.ndarray):
            return _interpolate_at_obs(self.mean, self._obs_coords), self.mean
        return self.mean, self.mean

    def predict(self):
        mean_at_obs, mean_field = self._mean_components()

        residuals = self.obs_values - mean_at_obs
        weights = scipy.linalg.cho_solve(self._cho_factor, residuals)

        kriging_mean = (np.ravel(mean_field) + self._grid_to_obs_cov @ weights).reshape(self._grid_shape)

        alpha = scipy.linalg.cho_solve(self._cho_factor, self._grid_to_obs_cov.T)
        kriging_variance = 1.0 - np.sum(self._grid_to_obs_cov * alpha.T, axis=1)
        kriging_stdev = np.sqrt(np.maximum(kriging_variance, 0.0)).reshape(self._grid_shape)

        return kriging_mean, kriging_stdev

    def simulate(self, n_sim=1):
        results = []
        for _ in range(n_sim):
            mean_at_obs, mean_field = self._mean_components()
            uncond = _simulate_uncond(
                self.variogram, self.nx, self.dx, self.ny, self.dy,
                self.nz, self.dz, self._grid_shape, self.ndims
            )
            uncond_at_obs = _interpolate_at_obs(uncond, self._obs_coords)
            obs_residuals = (self.obs_values - mean_at_obs) - uncond_at_obs
            weights = scipy.linalg.cho_solve(self._cho_factor, obs_residuals)
            correction = (self._grid_to_obs_cov @ weights).reshape(self._grid_shape)
            results.append(mean_field + uncond + correction)
        return results


class OrdinaryKriging:
    """Ordinary Kriging: estimates an unknown constant mean from the data.

    The unbiasedness constraint (weights sum to 1) is enforced via a Lagrange
    multiplier.  Weights and multiplier are computed via the Schur complement
    of C in the augmented system, using only the Cholesky factor of C::

        alpha  = C^{-1} 1                          (n_obs,)
        s      = 1^T alpha                          scalar
        beta   = C^{-1} K^T                         (n_obs, n_grid)
        mu(x)  = (1 - 1^T beta(x)) / s             (n_grid,)
        w(x)   = beta(x) - alpha * mu(x)            (n_obs, n_grid)

    Kriging mean:     z*(x) = w(x)^T z_obs
    Kriging variance: sigma^2(x) = C(0) - k(x)^T w(x) - mu(x)
    """

    def __init__(self, variogram, nx, dx, ny, dy, nz, dz,
                 obs_locations, obs_values, obs_uncertainties):
        self.variogram = variogram
        self.nx, self.dx = nx, dx
        self.ny, self.dy = ny, dy
        self.nz, self.dz = nz, dz

        self.obs_locations = np.asarray(obs_locations, dtype=float)
        self.obs_values = np.asarray(obs_values, dtype=float)
        self.obs_uncertainties = np.asarray(obs_uncertainties, dtype=float)

        self.ndims = _ndims(ny, nz)
        self._grid_shape = (nx, ny, nz)[:self.ndims]
        n_obs = len(self.obs_locations)

        _validate_inputs(self.obs_locations, self.obs_values, self.obs_uncertainties,
                         nx, dx, ny, dy, nz, dz, self.ndims)

        cov = _build_obs_cov_matrix(variogram, self.obs_locations, self.obs_uncertainties, self.ndims)
        try:
            cho = scipy.linalg.cho_factor(cov)
        except scipy.linalg.LinAlgError as exc:
            raise ValueError(_FACTORIZE_ERROR) from exc

        self._grid_to_obs_cov = _build_grid_to_obs_cov(
            variogram, nx, dx, ny, dy, nz, dz, self.obs_locations, self.ndims
        )
        self._obs_coords = _obs_interp_coords(self.obs_locations, dx, dy, dz, self.ndims)

        # Schur complement: solve for weights and Lagrange multipliers without
        # forming the indefinite augmented matrix.
        ones = np.ones(n_obs)
        alpha = scipy.linalg.cho_solve(cho, ones)                    # (n_obs,)
        s = ones @ alpha                                              # scalar: 1^T C^{-1} 1
        beta = scipy.linalg.cho_solve(cho, self._grid_to_obs_cov.T)  # (n_obs, n_grid)
        lagrange = (ones @ beta - 1.0) / s                           # (n_grid,)
        self._weights = beta - np.outer(alpha, lagrange)             # (n_obs, n_grid)
        self._lagrange = lagrange                                     # (n_grid,)

        # BLUE estimate of the unknown constant mean: mu_hat = (1^T C^{-1} z) / (1^T C^{-1} 1)
        gamma = scipy.linalg.cho_solve(cho, self.obs_values)         # C^{-1} z_obs
        self.estimated_mean: float = float(ones @ gamma / s)

    def predict(self):
        kriging_mean = (self._weights.T @ self.obs_values).reshape(self._grid_shape)

        kTw = np.sum(self._grid_to_obs_cov * self._weights.T, axis=1)
        kriging_variance = 1.0 - kTw - self._lagrange
        kriging_stdev = np.sqrt(np.maximum(kriging_variance, 0.0)).reshape(self._grid_shape)

        return kriging_mean, kriging_stdev

    def simulate(self, n_sim=1):
        results = []
        for _ in range(n_sim):
            uncond = _simulate_uncond(
                self.variogram, self.nx, self.dx, self.ny, self.dy,
                self.nz, self.dz, self._grid_shape, self.ndims
            )
            uncond_at_obs = _interpolate_at_obs(uncond, self._obs_coords)
            obs_residuals = self.obs_values - uncond_at_obs
            correction = (self._weights.T @ obs_residuals).reshape(self._grid_shape)
            results.append(uncond + correction)
        return results
