
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
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit

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

def gaussian_bg(x, A, mu, sigma, B):
    return A * np.exp(-(x - mu)**2 / (2*sigma**2)) + B

def process_spectra(spectra_dict, pixel_y=501, pixel_center=529, dispersion=0.144,
                    sigma=5, bin_size=5, lambda_cross=510):
    pattern = re.compile(r'(\d+)_(\d+)_(\w+)_degree', re.IGNORECASE)
    groups = {}

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

    spectra_data = {}

    for label, group in sorted(groups.items()):
        if 480 not in group or 570 not in group:
            continue

        # --- 480 nm ---
        img480 = group[480]
        sp480 = img480[pixel_y, :].astype(float)
        sp480 -= np.min(sp480)
        sp480 = gaussian_filter1d(sp480, sigma=sigma)

        px = np.arange(len(sp480))
        wavelength480 = 480 + (px - pixel_center) * dispersion
        wavelength480, sp480 = bin_spectrum(sp480, wavelength480, bin_size)
        mask480 = wavelength480 < lambda_cross
        wl480_cut = wavelength480[mask480]
        sp480_cut = sp480[mask480]

        # --- 570 nm ---
        img570 = group[570]
        sp570 = img570[pixel_y, :].astype(float)
        sp570 -= np.min(sp570)
        sp570 = gaussian_filter1d(sp570, sigma=sigma)

        px = np.arange(len(sp570))
        wavelength570 = 570 + (px - pixel_center) * dispersion
        wavelength570, sp570 = bin_spectrum(sp570, wavelength570, bin_size)
        mask570 = wavelength570 > lambda_cross
        wl570_cut = wavelength570[mask570]
        sp570_cut = sp570[mask570]

        # --- Offset ---
        N = min(2, len(sp480_cut), len(sp570_cut))
        offset = np.mean(sp480_cut[-N:]) - np.mean(sp570_cut[:N])
        sp570_aligned = sp570_cut + offset

        full_lambda = np.concatenate([wl480_cut, wl570_cut])
        full_signal = np.concatenate([sp480_cut, sp570_aligned])
        full_signal -= np.min(full_signal)

        spectra_data[label] = (full_lambda, full_signal)

    return spectra_data

def plot_spectra(spectra_data, color='#156082'):
    plt.figure(figsize=(5,5))
    first = True

    for label, (full_lambda, full_signal) in spectra_data.items():
        A0 = np.max(full_signal)
        mu0 = full_lambda[np.argmax(full_signal)]
        sigma0 = 20
        B0 = np.min(full_signal)

        try:
            popt, _ = curve_fit(gaussian_bg, full_lambda, full_signal,
                                p0=[A0, mu0, sigma0, B0])
            peak_value = gaussian_bg(popt[1], *popt)
        except RuntimeError:
            popt = [A0, mu0, sigma0, B0]
            peak_value = np.max(full_signal)

        y_normalized = full_signal / peak_value

        plt.scatter(full_lambda, y_normalized, s=35, alpha=0.4, color=color)
        mu = popt[1]
        sigma_fit = popt[2]
        fwhm_nm = 2.35482 * sigma_fit
        fwhm_eV = (1240 / mu**2) * fwhm_nm

        print(f"Peak at {mu:.1f} nm")
        print(f"FWHM = {fwhm_nm:.1f} nm, {fwhm_eV*1000:.0f} meV")

    plt.xlabel("Wavelength (nm)", fontsize=16)
    plt.ylabel("Photoluminescence Intensity (a.u.)", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(-0.03,1.1)
    plt.tight_layout()
    plt.show()



#%%

def main():

    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '/Users/antonellacutrupi/Desktop/code_github_adv_mat/data/spectra/')

    files = glob.glob(folder + '*.csv')

    spectra_dict = read_spectra_filtered(files, median_size=1)
    spectra_data = process_spectra(spectra_dict,
                                pixel_y=501,
                                pixel_center=529,
                                dispersion=0.144,
                                sigma=1,
                                bin_size=5,
                                lambda_cross=510)
    #Figure 3b)
    plot_spectra(spectra_data)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()
    plt.show()  # Show the plots interactively
#%%