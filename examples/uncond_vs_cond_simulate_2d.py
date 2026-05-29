import platform
import matplotlib
if platform.system() == 'Linux':
    matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
import gaussianfft as grf

# Grid setup
nx, ny = 100, 100
dx, dy = 10.0, 10.0

variogram = grf.variogram('matern52', main_range=300, perp_range=100, azimuth=30)

# Observation points
obs_pt = np.array([[250.0, 750.0], [755.0, 255.0]])
obs_val = np.array([2.5, -2.0])
obs_unc = np.array([0.00, 0.5])

# Unconditional simulation with seed=42
grf.seed(42)
uncond = grf.simulate(variogram, nx, dx, ny, dy)
uncond_2d = np.array(uncond).reshape((nx, ny), order='F')

# Conditional simulation with the same seed — shows how conditioning modifies the field
cond = grf.conditional_simulate(variogram, nx, dx, ny, dy, obs_pt, obs_val, obs_unc, n=1, seed=42)[0]

# Second realization with a different seed
cond_2 = grf.conditional_simulate(variogram, nx, dx, ny, dy, obs_pt, obs_val, obs_unc, n=1, seed=123)[0]

# Predict using the unconditional simulation as mean — equivalent to conditional_simulate(seed=42)
pred_with_uncond, variance_field = grf.predict(variogram, nx, dx, ny, dy, obs_pt, obs_val, obs_unc, mean=uncond_2d)

# Predict with zero mean
pred_zero_mean, variance_zero_mean = grf.predict(variogram, nx, dx, ny, dy, obs_pt, obs_val, obs_unc, mean=0.0)

# Plot
fig = plt.figure(figsize=(10, 12))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8])
axes = np.array([[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
                 [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]])
ax_cs = fig.add_subplot(gs[2, :])
vmin, vmax = -3, 3
extent = [0, nx * dx, 0, ny * dy]

# Cross-section along line through observations, extended 200m in both directions
p0_obs = obs_pt[0]
p1_obs = obs_pt[1]
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


def bilinear_sample(field, xs, ys):
    return map_coordinates(field, [xs / dx, ys / dy], order=1, mode='nearest')


ax = axes[0, 0]
im = ax.imshow(uncond_2d.T, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap='RdBu_r')
ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'k--', lw=1)
ax.set_title('Unconditional (seed=42)')
fig.colorbar(im, ax=ax)

ax = axes[0, 1]
im = ax.imshow(cond.T, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap='RdBu_r')
ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'k--', lw=1)
ax.plot(obs_pt[:, 0], obs_pt[:, 1], 'ko', markersize=8)
for i, v in enumerate(obs_val):
    ax.annotate(f'{v:.1f}', obs_pt[i], textcoords="offset points", xytext=(5, 5))
ax.set_title('Conditional (seed=42)')
fig.colorbar(im, ax=ax)

ax = axes[1, 0]
im = ax.imshow(cond_2.T, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap='RdBu_r')
ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'k--', lw=1)
ax.plot(obs_pt[:, 0], obs_pt[:, 1], 'ko', markersize=8)
for i, v in enumerate(obs_val):
    ax.annotate(f'{v:.1f}', obs_pt[i], textcoords="offset points", xytext=(5, 5))
ax.set_title('Conditional (seed=123)')
fig.colorbar(im, ax=ax)

ax = axes[1, 1]
im = ax.imshow(pred_with_uncond.T, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap='RdBu_r')
ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'k--', lw=1)
ax.plot(obs_pt[:, 0], obs_pt[:, 1], 'ko', markersize=8)
for i, v in enumerate(obs_val):
    ax.annotate(f'{v:.1f}', obs_pt[i], textcoords="offset points", xytext=(5, 5))
ax.set_title('Predict(mean=unconditional)')
fig.colorbar(im, ax=ax)

# Cross-section panel
pred_cs = bilinear_sample(pred_with_uncond, line_x, line_y)
std_cs = np.sqrt(bilinear_sample(variance_field, line_x, line_y))
pred_zero_cs = bilinear_sample(pred_zero_mean, line_x, line_y)
std_zero_cs = np.sqrt(bilinear_sample(variance_zero_mean, line_x, line_y))
ax_cs.fill_between(dist, pred_zero_cs - std_zero_cs, pred_zero_cs + std_zero_cs, alpha=0.2, color='C2', label='±1 std dev (mean=0)')
ax_cs.plot(dist, bilinear_sample(uncond_2d, line_x, line_y), label='Unconditional (seed=42)', lw=1.2)
ax_cs.plot(dist, bilinear_sample(cond, line_x, line_y), label='Conditional (seed=42)', lw=1.5)
ax_cs.plot(dist, bilinear_sample(cond_2, line_x, line_y), label='Conditional (seed=123)', lw=1.5)
ax_cs.plot(dist, pred_zero_cs, label='Predict(mean=0)', lw=1.5, ls='--', color='C2')
# Plot observations with their kriging variance (measurement uncertainty)
for i, sx in enumerate(obs_proj_dist):
    ax_cs.axvline(sx, color='k', lw=0.5, alpha=0.3)
    obs_std = obs_unc[i]
    ax_cs.plot(sx, obs_val[i], 'ro', markersize=7, zorder=5, label='Observation' if i == 0 else '')
    if obs_std > 0:
        ax_cs.errorbar(sx, obs_val[i], yerr=obs_std, fmt='none', ecolor='r', capsize=4, capthick=1.5, zorder=5)
ax_cs.set_xlabel('arc length along diagonal cross-section')
ax_cs.set_ylabel('value')
ax_cs.set_title('Cross-section through observations')
ax_cs.legend(loc='best')
ax_cs.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
