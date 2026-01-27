
#%%

# Author: Antonella Cutrupi

import numpy as np
import os
import json
import re
import glob

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

# Set up the environment for plotting

import matplotlib.font_manager as font_manager
from matplotlib import rcParams

font = font_manager.FontProperties(family='Arial',
                                   
                                   style='normal', size=40)

rcParams['font.family'] = font.get_name()
rcParams['font.size'] = 20
rcParams['font.style'] = 'normal'

import warnings
warnings.filterwarnings("ignore")  # Ignore warnings for cleaner output

def angles_and_diffusivities():

        angle = np.array([70, 80, 90, 85, 110, 175, 105, 95, 120, 135, 115, 100, 190, 50]) #scan angles in degrees
        Diff_left = np.array([0.055, 0.058, 0.102, 0.089, 0.112, 0.05, 0.13, 0.119, 0.087, 0.074, 0.083, 0.147, 0.042, 0.052])#cm^2/s
        Diff_right = np.array([0.055, 0.058, 0.102, 0.089, 0.112, 0.05, 0.13, 0.119, 0.087, 0.074, 0.083, 0.147, 0.042, 0.052])#cm^2/s

        return angle, Diff_left, Diff_right

def data_polar_plot(left_diff, right_diff, scan_angle):

    max_val = max(right_diff.max(), left_diff.max())
    D_right = right_diff/ max_val * 0.9 
    D_left  = left_diff / max_val * 0.9

    theta_right = np.deg2rad(scan_angle)
    theta_left  = np.deg2rad(scan_angle - 180)

    idx_max = np.argmax(left_diff)
    idx_min = np.argmin(right_diff)
    angle_max = scan_angle[idx_max]
    angle_min = scan_angle[idx_min]

    cmap_colors = [
        (30/255,144/255,255/255),  
        (100/255,149/255,237/255), 
        (72/255,61/255,139/255),   
        (205/255,92/255,92/255),  
        (188/255,101/255,106/255), 
    ]
    cmap = LinearSegmentedColormap.from_list('custom_grad', cmap_colors)

    colors_final = []
    for a in scan_angle:
        if a == angle_max:
            colors_final.append('#1f77b4')  
            colors_final.append('#d62728')             
        else:
            dist = abs(a - angle_max)
            total_dist = abs(angle_min - angle_max)
            t = np.clip(dist / total_dist, 0, 1)
            t = t**0.3  
            colors_final.append(cmap(t))

    theta_all = np.concatenate([theta_right, theta_left])
    r_all = np.concatenate([D_right, D_left])

    sorted_indices = np.argsort(theta_all)
    theta_sorted = theta_all[sorted_indices]
    r_sorted = r_all[sorted_indices]

    return theta_sorted, r_sorted, D_left, D_right, theta_left, theta_right, cmap, colors_final

def polar_plot(theta,r, theta_r, theta_l, d_right, d_left, cmap, colors_final):

    fig = plt.figure(figsize=(8,8))
    plt.rcParams['font.family'] = 'Arial'
    ax = plt.subplot(111, projection='polar')
    ax.plot(theta, r, color='black', linewidth=2, linestyle='--', alpha=0.7)

    for i, (th_r, r_r, th_l, r_l) in enumerate(zip(theta_r, d_right, theta_l, d_left)):
        c = colors_final[i]
        ax.plot(th_r, r_r, 'o', markersize=18, color=c)
        ax.plot(th_l, r_l, 'o', markersize=18, color=c)
        ax.plot([th_r, th_l], [r_r, r_l], color=c, linewidth=2, alpha=0.5)
        ax.plot([th_r, th_r], [0, r_r], color=c, linestyle='--', alpha=0.4, linewidth=1)
        ax.plot([th_l, th_l], [0, r_l], color=c, linestyle='--', alpha=0.4, linewidth=1)

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_ylim(0, 0.95)
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 30)))
    ax.tick_params(axis='x', labelsize=22, pad=15)

    ax.plot([], [], 'o', color='#1f77b4', label='$c$ - axis')
    ax.plot([], [], 'o', color='#d62728', label='$a$ - axis')
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(0.8, 1.1),
        frameon=False,
        prop={'family':'Arial', 'size':25},
        markerscale=3,
        handlelength=3
    )
    ax.set_yticklabels([])
    norm = Normalize(vmin=min(d_right.min(), d_left.min()), vmax=max(d_right.max(), d_left.max()))
    sm = plt.cm.ScalarMappable(cmap=cmap.reversed(), norm=norm) 
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.15, fraction=0.05)
    cbar.set_label('Diffusivity (cm²/s)', fontsize=24)
    cbar.ax.tick_params(labelsize=24)
    plt.tight_layout()

#%%

def main():

    angle, Diff_left, Diff_right = angles_and_diffusivities()

    theta_sorted, r_sorted, D_left, D_right, theta_l, theta_r, cmap, color_plot = data_polar_plot(Diff_left, Diff_right, angle)

    #Figure 4d)
    polar_plot(theta_sorted,r_sorted, theta_r, theta_l, D_right, D_left, cmap, color_plot)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()
    plt.show()  # Show the plots interactively


# %%
