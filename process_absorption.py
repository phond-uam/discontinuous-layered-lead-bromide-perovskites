
#%%

# Author: Antonella Cutrupi

import numpy as np
import re
import pandas as pd
import os
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

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

def load_rtf_table(path):
    with open(path, "r", encoding="latin-1") as f:
        text = f.read()
    text = re.sub(r"{\\.*?}|\\[A-Za-z]+\d* ?", " ", text)
    text = re.sub(r"[{}]", " ", text)
    lines = text.split("\n")
    data = []
    for line in lines:
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        if len(nums) >= 2:
            wavelength = float(nums[0])
            absorbance = float(nums[1])
            data.append([wavelength, absorbance])
    df = pd.DataFrame(data, columns=["wavelength", "absorbance"])
    return df

def process_data(df):
    df = df[df["wavelength"] >= 250].reset_index(drop=True)
    n_points = 10  #extimation background using last n points
    background = df["absorbance"].values[-n_points:].mean()
    abs_corrected = df["absorbance"].values - background
    return df["wavelength"].values, abs_corrected

def plot_data(wavelength, abs_corrected):
    sigma = 5  
    smoothed = gaussian_filter1d(abs_corrected, sigma=sigma)
    normalized = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())
    derivative = np.gradient(normalized, wavelength)
    zero_cross_idx = np.where(np.diff(np.sign(derivative)))[0]
    if len(zero_cross_idx) > 1:
        zero_cross_idx = zero_cross_idx[1:]
    zero_cross_wl = wavelength[zero_cross_idx]
    zero_cross_abs = normalized[zero_cross_idx]
    for wl, val in zip(zero_cross_wl, zero_cross_abs):
        print(f"{wl:.2f} nm -> {val:.3f}")
    plt.figure(figsize=(6,4))
    plt.plot(wavelength, normalized, linewidth = 2.5, color= 'darkslategray')
    y_point = zero_cross_abs[-2]
    plt.scatter(
        zero_cross_wl[-2],
        y_point,
        color="#00ffff",
        s=100,zorder=3,
        label=f"{zero_cross_wl[-1]:.2f} nm"
    )
    plt.scatter(
        zero_cross_wl[-1],
        y_point,
        color="#11c638",
        s=100,zorder=3,
        label=f"{zero_cross_wl[-1]:.2f} nm"
    )
    plt.xlabel("Wavelength (nm)", fontsize = 16)
    plt.ylabel("Absorption (a.u.)", fontsize = 16)
    plt.tick_params(axis='both', which='major', labelsize=14)


#%%

#Figure S4 c)
def main():

    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '/Users/antonellacutrupi/Desktop/code_github_adv_mat/data/absorption/Absorbance RUM156a4.rtf')

    df = load_rtf_table(folder)

    wavelength, abs_corrected = process_data(df)

    df = df[df["wavelength"] >= 250].reset_index(drop=True)
    
    plot_data(wavelength, abs_corrected)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()
    plt.show()  # Show the plots interactively
#%%