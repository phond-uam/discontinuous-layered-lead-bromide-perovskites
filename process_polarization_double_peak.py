
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

def bin_spectrum(signal, wavelength, bin_size):
        n_bins = len(signal) // bin_size
        signal_binned = signal[:n_bins*bin_size].reshape(n_bins, bin_size).mean(axis=1)
        wavelength_binned = wavelength[:n_bins*bin_size].reshape(n_bins, bin_size).mean(axis=1)
        return wavelength_binned, signal_binned


def process_spectra(spectra_dict):

        pixel_y = 440
        pixel_center = 529
        dispersion = 0.144 # nm/pixel

        pattern = re.compile(r'(\d+)_(\d+)_(\w+)_degree', re.IGNORECASE)

        groups = {}  # {label: {lambda: img}}

        for file, img in spectra_dict.items():
            fname = file.split("/")[-1]
            m = pattern.search(fname)
            if not m:
                continue

            lam = float(m.group(1))      
            idx = int(m.group(2))        
            typ = m.group(3)             

            label = f"{idx}_{typ}"

            if label not in groups:
                groups[label] = {}

            groups[label][lam] = img

        available_colors = plt.cm.tab10.colors
        color_map = {label: available_colors[i % len(available_colors)]
                    for i, label in enumerate(sorted(groups.keys()))}

        sigma = 5  # smoothing

        for label, group in sorted(groups.items()):
            for lam in sorted(group.keys()):
                img = group[lam]
                spectrum = img[pixel_y, :].astype(float)
                spectrum -= np.min(spectrum)
                spectrum = gaussian_filter1d(spectrum, sigma=sigma)

                px = np.arange(len(spectrum))
                wavelength = lam + (px - pixel_center) * dispersion

                #plt.scatter(wavelength, spectrum, color=color_map[label], s=10)

        sigma = 1
        lambda_cross = 540
        bin_size = 5
        max_intensity = 0

        spectra_data = {}
        integrated_480 = []
        integrated_570 = []
        labels_list = []


        for label, group in sorted(groups.items()):
            if 480 not in group or 570 not in group:
                continue

            labels_list.append(label)

            #480 nm
            img480 = group[480]
            sp480 = img480[pixel_y, :].astype(float)
            sp480 -= np.min(sp480)

            px = np.arange(len(sp480))
            wavelength480 = 480 + (px - pixel_center) * dispersion

            wavelength480, sp480 = bin_spectrum(sp480, wavelength480, bin_size)

            mask480 = wavelength480 < lambda_cross
            sp480_cut = sp480[mask480]
            wl480_cut = wavelength480[mask480]

            integrated_480.append(np.trapz(sp480_cut, wl480_cut))

            # 570 nm 
            img570 = group[570]
            sp570 = img570[pixel_y, :].astype(float)
            sp570 -= np.min(sp570)

            px = np.arange(len(sp570))
            wavelength570 = 570 + (px - pixel_center) * dispersion

            wavelength570, sp570 = bin_spectrum(sp570, wavelength570, bin_size)

            mask570 = wavelength570 > lambda_cross
            sp570_cut = sp570[mask570]
            wl570_cut = wavelength570[mask570]

            integrated_570.append(np.trapz(sp570_cut, wl570_cut))

            N = min(2, len(sp480_cut), len(sp570_cut))
            mean_480 = np.mean(sp480_cut[-N:])
            mean_570 = np.mean(sp570_cut[:N])

            offset = mean_480 - mean_570
            sp570_aligned = sp570_cut + offset

            full_lambda = np.concatenate([wl480_cut, wl570_cut])
            full_signal = np.concatenate([sp480_cut, sp570_aligned])
            full_signal -= np.min(full_signal)

            spectra_data[label] = (full_lambda, full_signal)
            max_intensity = max(max_intensity, np.max(full_signal))

        return spectra_data
        

def plot_spectra(spectra_data):

    rose_brown_cmap = ["#C97A6D" , "#6A3024"]  

    label = list(spectra_data.keys())[0]
    full_lambda, full_signal = spectra_data[label] 

    plt.figure(figsize=(7, 5))
    for (label, (full_lambda, full_signal)), color in zip(
            spectra_data.items(), rose_brown_cmap):    
        
        plt.scatter(full_lambda,
                full_signal,
                    s=40,alpha = 0.5,
                    color=color)
        
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker='o', color='none',
            markerfacecolor=rose_brown_cmap[1],
            markersize=8, label='0°'),
        Line2D([0], [0], marker='o', color='none',
            markerfacecolor=rose_brown_cmap[0],
            markersize=8, label='90°')
    ]

    plt.legend(handles=legend_elements, fontsize=14)
    plt.xlabel("Wavelength (nm)", fontsize = 18)
    plt.ylabel("Photoluminescence Intensity  (a.u.)", fontsize = 18)
    plt.tick_params(axis='both', which='major', labelsize=16)


#%%

def main():

    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '/Users/antonellacutrupi/Desktop/code_github_adv_mat/data/double_peak/polarization/')

    files = glob.glob(folder + '*.csv')

    spectra_dict = read_spectra_filtered(files, median_size=3)
        
    spectra_data = process_spectra(spectra_dict)

    #Figure S4a)
    plot_spectra(spectra_data)

    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()
    plt.show()  # Show the plots interactively
#%%