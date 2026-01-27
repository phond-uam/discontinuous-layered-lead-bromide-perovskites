
#%%

# Author: Antonella Cutrupi

import numpy as np
import re
import pandas as pd
import glob
import os 
from matplotlib.lines import Line2D
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter

from matplotlib.colors import LinearSegmentedColormap


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

def read_spectra_filtered(files, median_size=3):
    spectra_dict = {}
    for file in files:
        spectra_df = pd.read_csv(file, header=None, skiprows=2)
        spectra_df = spectra_df.iloc[:1024]
        spectra_df = spectra_df.drop(0, axis=1)
        img = spectra_df.iloc[1:].values.astype(float)
        img_filtered = median_filter(img, size=(1, median_size))
        spectra_dict[file] = img_filtered
    return spectra_dict


def process_images(spectra_dict):

    angles_plot = []
    integrated_I = []
    roi_size = 50

    for file, img in spectra_dict.items():
        match = re.search(r'Angle_(\d+)', file)
        if not match:
            continue
        angle = int(match.group(1))
        angles_plot.append(angle + 90)

        max_idx = np.unravel_index(np.argmax(img), img.shape)
        center_y, center_x = max_idx
        y_start = max(0, center_y - roi_size // 2)
        y_end = min(img.shape[0], center_y + roi_size // 2 + 1)
        
        x_start = max(0, center_x - roi_size // 2)
        x_end = min(img.shape[1], center_x + roi_size // 2 + 1)

        roi_image = img[y_start:y_end, x_start:x_end]

        bg_y_start, bg_y_end = 0, 100
        bg_x_start, bg_x_end = 0, img.shape[1] 

        bg_value_area = img[bg_y_start:bg_y_end, bg_x_start:bg_x_end]

        bg_value = np.nanmean(bg_value_area) 

        roi_corrected = roi_image - bg_value
        roi_corrected[roi_corrected < 0] = 0
        
        I_total = np.sum(roi_corrected) 
        integrated_I.append(I_total)

    angles= np.array(angles_plot)
    integrated_I = np.array(integrated_I)
    min_intensity = np.min(integrated_I)
    min_index = np.argmin(integrated_I)
    print(f"Min intensity: {min_intensity} at angle {angles[min_index]} degrees")
    max_intensity = np.max(integrated_I)
    max_index = np.argmax(integrated_I)
    print(f"Max intensity: {max_intensity} at angle {angles[max_index]+90} degrees")

    order = np.argsort(angles)
    angles = angles[order]
    integrated_I = integrated_I[order]

    angles_rad = np.deg2rad(angles)

    return angles_rad, integrated_I

def polar_scatter_plot(angles_rad, integrated_I):
    r = integrated_I * 0.99

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})

    intensity = integrated_I/np.max(integrated_I)
    rose_brown_cmap = LinearSegmentedColormap.from_list(
        "rose_brown",
        ["#C97A6D",  "#A3504A",     
        "#6A3024"]   
    )
    sc = ax.scatter(
        angles_rad,
        r,                         
        c=intensity,          
        cmap=rose_brown_cmap,
        s=200,
    edgecolors='#5C1F14',       
        linewidths=2.5,
        alpha=0.7
    )
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rmax(integrated_I.max()*1.06)
    ax.set_yticklabels([])
    ax.tick_params(axis='x', labelsize=16)

    cbar = plt.colorbar(
        sc,
        ax=ax,
        orientation='horizontal',
        pad=0.09                   
    )

    cbar.set_label("Norm Intensity (a.u.)", fontsize=18)
    cbar.ax.tick_params(labelsize=16)
    plt.tight_layout()

#%%

def main():

    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '/Users/antonellacutrupi/Desktop/code_github_adv_mat/data/polarization/')

    files = glob.glob(folder + 'Angle_*.csv')
    spectra_dict = read_spectra_filtered(files, median_size=5)
    angles_rad, integrated_I = process_images(spectra_dict)
    
    #Figure 3c)
    polar_scatter_plot(angles_rad, integrated_I)
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()
    plt.show()  # Show the plots interactively
#%%