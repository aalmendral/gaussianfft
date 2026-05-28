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
pred_with_uncond, _ = grf.predict(variogram, nx, dx, ny, dy, obs_pt, obs_val, obs_unc, mean=uncond_2d)

# Plot
fig = plt.figure(figsize=(10, 12))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8])
axes = np.array([[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
                 [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]])
ax_cs = fig.add_subplot(gs[2, :])
vmin, vmax = -3, 3
extent = [0, nx * dx, 0, ny * dy]

# Cross-section line through both observations
p0, p1 = obs_pt[0], obs_pt[1]
s = np.linspace(0.0, 1.0, 400)
line_x = p0[0] + s * (p1[0] - p0[0])
line_y = p0[1] + s * (p1[1] - p0[1])
arclen = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
dist = s * arclen


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
ax_cs.plot(dist, bilinear_sample(uncond_2d, line_x, line_y), label='Unconditional (seed=42)', lw=1.2)
ax_cs.plot(dist, bilinear_sample(cond, line_x, line_y), label='Conditional (seed=42)', lw=1.5)
ax_cs.plot(dist, bilinear_sample(cond_2, line_x, line_y), label='Conditional (seed=123)', lw=1.5)
for sx, v in zip([0.0, arclen], obs_val):
    ax_cs.axvline(sx, color='k', lw=0.5, alpha=0.5)
    ax_cs.plot(sx, v, 'ko', markersize=8)
    ax_cs.annotate(f' obs={v:.2f}', (sx, v), textcoords='offset points', xytext=(6, 4))
ax_cs.set_xlabel('arc length along cross-section')
ax_cs.set_ylabel('value')
ax_cs.set_title(f'Cross-section through observations')
ax_cs.legend(loc='best')
ax_cs.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
