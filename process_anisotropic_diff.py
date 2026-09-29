
#%%

# Author: Antonella Cutrupi
import numpy as np
import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 20
plt.rcParams['font.style'] = 'normal'

SCALE = 0.9
CAP = np.deg2rad(3.5)
CMAP_COLORS = [
    (30/255, 144/255, 255/255),
    (100/255, 149/255, 237/255),
    (72/255, 61/255, 139/255),
    (205/255, 92/255, 92/255),
    (214/255, 39/255, 40/255),
]
CMAP = LinearSegmentedColormap.from_list('custom_grad', CMAP_COLORS)

def angles_and_diffusivities():
    angle = np.array([70, 80, 90, 85, 110, 175, 105, 95, 120, 135, 115, 100, 190, 50])
    diff_left = np.array([0.055, 0.058, 0.102, 0.089, 0.112, 0.05, 0.13,
                          0.119, 0.087, 0.074, 0.083, 0.147, 0.042, 0.052])
    diff_right = diff_left.copy()
    error_d = np.array([0.005, 0.004, 0.004, 0.005, 0.003, 0.005, 0.008,
                        0.006, 0.007, 0.005, 0.004, 0.008, 0.006, 0.004])
    return angle, diff_left, diff_right, error_d

def find_max(angles, diff_left, diff_right):
    if diff_left.max() > diff_right.max():
        return angles[np.argmax(diff_left)], 'left'
    return angles[np.argmax(diff_right)], 'right'

def compute_colors(angles, diff_left, diff_right):
    max_val = max(diff_left.max(), diff_right.max())
    idx_max = np.argmax(np.maximum(diff_left, diff_right))
    angle_max = angles[idx_max]
    angle_min = angles[np.argmin(diff_right)]
    total_dist = abs(angle_min - angle_max) or 1.0

    colors_final = []
    for a in angles:
        if a == angle_max:
            right_is_max = np.isclose(diff_right[idx_max], max_val)
            left_is_max = np.isclose(diff_left[idx_max], max_val)
            colors_final.append('#1f77b4' if right_is_max else '#d62728')
            colors_final.append('#1f77b4' if left_is_max else '#d62728')
        else:
            t = np.clip(abs(a - angle_max) / total_dist, 0, 1)
            t = t ** 0.3
            colors_final.append(CMAP(t))

    return colors_final

def draw_error_bar(ax, th, r, err, color='gray', lw=1.5):
    band = np.linspace(th - CAP, th + CAP, 20)
    ax.fill_between(band, r - err, r + err, color=color, alpha=0.15, zorder=0)
    ax.plot([th, th], [r - err, r + err], '--', color=color, lw=lw)
    for edge in (r + err, r - err):
        ax.plot([th - CAP, th + CAP], [edge, edge], '--', color=color, lw=lw)

def polar_plot(scan_angle, diff_right, diff_left, error_d, fill=False, fill_alpha=0.4):
    max_val = max(diff_right.max(), diff_left.max())
    r_right = diff_right / max_val * SCALE
    r_left = diff_left / max_val * SCALE

    th_right = np.deg2rad(scan_angle)
    th_left = np.deg2rad(scan_angle - 180)
    colors_final = compute_colors(scan_angle, diff_left, diff_right)

    th_all = np.concatenate([th_right, th_left])
    r_all = np.concatenate([r_right, r_left])
    v_all = np.concatenate([diff_right, diff_left])
    order = np.argsort(th_all)
    th_closed = np.append(th_all[order], th_all[order][0])
    r_closed = np.append(r_all[order], r_all[order][0])

    angle_max = scan_angle[np.argmax(np.maximum(diff_left, diff_right))]
    margin = (r_all.max() - r_all.min()) * 0.03
    r_lo = r_all.min() - margin
    r_hi = r_all.max() + margin

    vmin = min(diff_right.min(), diff_left.min())
    vmax = max(diff_right.max(), diff_left.max())
    norm = Normalize(vmin, vmax)
    cmap_rev = CMAP.reversed()

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection='polar')

    if fill:
        th_f = np.append(th_all[order], th_all[order][0] + 2 * np.pi)
        v_sorted = v_all[order]
        v_f = np.append(v_sorted, v_sorted[0])
        for i in range(len(th_f) - 1):
            color = cmap_rev(norm((v_f[i] + v_f[i + 1]) / 2))
            ax.fill([th_f[i], th_f[i], th_f[i + 1]],
                    [r_lo, r_closed[i], r_closed[i + 1]],
                    color=color, alpha=fill_alpha, linewidth=0, zorder=0)

    ax.plot(th_closed, r_closed, '--', color='black', lw=2, alpha=0.5)

    j = 0
    for thr, rr, thl, rl, e, a in zip(th_right, r_right, th_left, r_left, error_d, scan_angle):
        if a == angle_max:
            c_r, c_l = colors_final[j], colors_final[j + 1]
            c_line = '#1f77b4'
            j += 2
        else:
            c_r = c_l = c_line = colors_final[j]
            j += 1

        ax.plot([thr, thl], [rr, rl], color=c_line, lw=2, alpha=0.5)
        for th, r, c in ((thr, rr, c_r), (thl, rl, c_l)):
            draw_error_bar(ax, th, r, e)
            ax.plot(th, r, 'o', ms=5, color=c, zorder=5)

    ax.set_rlim(r_lo, r_hi)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_yticklabels([])
    sm = plt.cm.ScalarMappable(cmap=cmap_rev, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.15, fraction=0.05)
    ticks = [vmin, (vmin + vmax) / 2, vmax]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{t:.3f}' for t in ticks])
    cbar.set_label('Diffusivity (cm²/s)', fontsize=24)
    cbar.ax.tick_params(labelsize=24)

    plt.tight_layout()
    return fig, ax

#%%


if __name__ == '__main__':
    
    angle, diff_left, diff_right, error_d = angles_and_diffusivities()
    polar_plot(angle, diff_right, diff_left, error_d, fill=False)
    polar_plot(angle, diff_right, diff_left, error_d, fill=True)
    plt.show()

# %%
