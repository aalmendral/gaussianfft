import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
import gaussianfft as grf

grf.seed(42)

# Grid setup
nx, ny = 100, 100
dx, dy = 10.0, 10.0

variogram = grf.variogram('matern52', main_range=300, perp_range=100, azimuth=30)

# Observation points
obs_locations = np.array([[250.0, 750.0], [755.0, 255.0]])
obs_values = np.array([2.5, -2.0])
obs_uncertainties = np.array([0.00, 0.5])

# Average of 100 conditional simulations
n_realizations = 100
realizations = grf.conditional_simulate(
    variogram, nx, dx, ny, dy, obs_locations, obs_values, obs_uncertainties,
    n=n_realizations,
)
avg_cond = np.mean(realizations, axis=0)

# Kriging prediction
prediction, variance_field = grf.predict(
    variogram, nx, dx, ny, dy, obs_locations, obs_values, obs_uncertainties,
)

def bilinear_sample(field, xs, ys):
    """Bilinearly sample a 2D field of shape (nx, ny) at physical coords (xs, ys)."""
    return map_coordinates(field, [xs / dx, ys / dy], order=1, mode='nearest')


# Cross-section along line through observations, extended 200m in both directions
p0_obs = obs_locations[0]
p1_obs = obs_locations[1]
obs_direction = p1_obs - p0_obs
obs_dist = np.linalg.norm(obs_direction)
obs_direction_unit = obs_direction / obs_dist
p0 = p0_obs - 200.0 * obs_direction_unit
p1 = p1_obs + 200.0 * obs_direction_unit
s = np.linspace(0.0, 1.0, 400)
line_x = p0[0] + s * (p1[0] - p0[0])
line_y = p0[1] + s * (p1[1] - p0[1])
arclen = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
dist = s * arclen

# Observation positions along the extended line
obs_proj_dist = np.array([200.0, 200.0 + obs_dist])

# Plot: 1x2 imshows on top, cross-section spanning the bottom
fig = plt.figure(figsize=(11, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.8])
ax_avg = fig.add_subplot(gs[0, 0])
ax_pred = fig.add_subplot(gs[0, 1])
ax_cs = fig.add_subplot(gs[1, :])
extent = [0, nx * dx, 0, ny * dy]
vmin, vmax = -3, 3

im = ax_avg.imshow(avg_cond.T, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap='RdBu_r')
ax_avg.plot([p0[0], p1[0]], [p0[1], p1[1]], 'k--', lw=1)
ax_avg.plot(obs_locations[:, 0], obs_locations[:, 1], 'ko', markersize=8)
for i, v in enumerate(obs_values):
    ax_avg.annotate(f'{v:.1f}', obs_locations[i], textcoords="offset points", xytext=(5, 5))
ax_avg.set_title(f'Mean of {n_realizations} conditional simulations')
fig.colorbar(im, ax=ax_avg)

im = ax_pred.imshow(prediction.T, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap='RdBu_r')
ax_pred.plot([p0[0], p1[0]], [p0[1], p1[1]], 'k--', lw=1)
ax_pred.plot(obs_locations[:, 0], obs_locations[:, 1], 'ko', markersize=8)
for i, v in enumerate(obs_values):
    ax_pred.annotate(f'{v:.1f}', obs_locations[i], textcoords="offset points", xytext=(5, 5))
ax_pred.set_title('Kriging prediction')
fig.colorbar(im, ax=ax_pred)

# Cross-section panel
pred_cs = bilinear_sample(prediction, line_x, line_y)
std_cs = np.sqrt(bilinear_sample(variance_field, line_x, line_y))
ax_cs.fill_between(dist, pred_cs - std_cs, pred_cs + std_cs, alpha=0.3, label='±1 std dev', color='C1')
ax_cs.plot(dist, bilinear_sample(avg_cond, line_x, line_y),
           label=f'Mean of {n_realizations} conditional sims', lw=1.5)
ax_cs.plot(dist, pred_cs,
           label='Kriging prediction', lw=1.5, ls='--', color='C1')
# Plot observations with their kriging variance (measurement uncertainty)
for i, sx in enumerate(obs_proj_dist):
    ax_cs.axvline(sx, color='k', lw=0.5, alpha=0.3)
    obs_std = obs_uncertainties[i]
    ax_cs.plot(sx, obs_values[i], 'ro', markersize=7, zorder=5, label='Observation' if i == 0 else '')
    if obs_std > 0:
        ax_cs.errorbar(sx, obs_values[i], yerr=obs_std, fmt='none', ecolor='r', capsize=4, capthick=1.5, zorder=5)
ax_cs.set_xlabel('arc length along diagonal cross-section')
ax_cs.set_ylabel('value')
ax_cs.set_title('Cross-section through observations')
ax_cs.legend(loc='best')
ax_cs.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
