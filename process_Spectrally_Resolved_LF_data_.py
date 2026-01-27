
#%%

# Author: Antonella Cutrupi

import numpy as np
import os
import json
import glob
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from scipy.optimize import curve_fit
from matplotlib.colors import to_rgba

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

def load_data(filename):
    # load the data from the npz file
    data_dict = np.load(filename, allow_pickle=True)['arr_0'].item()
    
    return data_dict

def load_data_and_metadata(folder_path):
    """
    Load npz data files and their corresponding metadata json files from a folder.
    
    Args:
        folder_path (str): Path to the folder containing npz and json files
        
    Returns:
        list: List of dictionaries containing matched data and metadata files
    """
    
    # Get all npz and json files
    data_files = glob.glob(os.path.join(folder_path, '*.npz'))
    metadata_files = glob.glob(os.path.join(folder_path, '*.json'))
    
    # Create a dictionary to store metadata files by timestamp
    metadata_dict = {}
    for meta_file in metadata_files:
        timestamp = os.path.getmtime(meta_file)
        metadata_dict[timestamp] = meta_file
    
    # Match data files with metadata
    matched_pairs = []
    for data_file in data_files:
        data_timestamp = os.path.getmtime(data_file)
        
        # Find closest metadata file timestamp
        closest_timestamp = min(metadata_dict.keys(), 
                              key=lambda x: abs(x - data_timestamp))
        
        # Match if within 60 seconds
        if abs(closest_timestamp - data_timestamp) <= 60:
            with open(metadata_dict[closest_timestamp], 'r') as f:
                metadata = json.load(f)
            
            # Load the data using your existing load_data function
            data_dict = load_data(data_file)
            
            matched_pairs.append({
                'data_file': data_file,
                'metadata_file': metadata_dict[closest_timestamp],
                'data': data_dict,  # This is now a dictionary with 'wavelength', 'data', 'time' keys
                'metadata': metadata,
                'laser_power': metadata.get('laser_pw', None)
            })
            
    return matched_pairs


def param_from_metadata(metadata_json):

    # given a metadata json file, extract the parameters

    TimeResolution = metadata_json['TimeResolution']
    AcquisitionTime = metadata_json['AcquisitionTime']
    laserPower = metadata_json['laserPower']
    laserDivider = metadata_json['laserDivider']
    spotsize = metadata_json['spot_size']
    steps = metadata_json['steps']
    center_w = metadata_json['center_w']
    wavelength_range_start = metadata_json['wavelenght_range_start']
    wavelength_range_end = metadata_json['wavelenth_range_end']
    objective = metadata_json['objective']
    maxRepRate = metadata_json['MaxRepRate']
    sample = metadata_json['sample']

    return TimeResolution, AcquisitionTime, laserPower, laserDivider, spotsize, steps, center_w,wavelength_range_start,wavelength_range_end, objective, maxRepRate, sample

def extract_parameters_from_metadata(metadata_path, folder=None):
    if isinstance(metadata_path, list):
        metadata_path = metadata_path[0]

    if os.path.isdir(metadata_path):
        metadata_path = os.path.join(metadata_path, "metadata.json")

    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return param_from_metadata(metadata)

def biexp(t, A1, tau1, A2, tau2):
    return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2)

def process_measurement_data(folder, time_lim, select=0):

    measurements = load_data_and_metadata(folder)

    print("\nAvailable measurements:")
    for i, meas in enumerate(measurements):
        print(f"[{i}] File: {os.path.basename(meas['data_file'])}")

    selected_measurement = measurements[select]

    results_dict = selected_measurement['data']

    (
        TimeResolution, AcquisitionTime, laser_pw, laser_divider,
        spotsize, steps, center_w, wavelength_range_start,
        wavelength_range_end, objective, MaxRepRate, sample
    ) = extract_parameters_from_metadata(
        [selected_measurement['metadata_file']], folder
    )

    wavelengths = results_dict['wavelength']
    data = results_dict['data']

    N_timepoints = data.shape[1]
    Time_array = np.arange(N_timepoints) * TimeResolution

    time_lim_index = int(time_lim / TimeResolution * 1e03)

    return (
        wavelengths,
        data,
        Time_array,
        TimeResolution,
        MaxRepRate,
        laser_divider,
        time_lim_index
    )

def plot_selected_wavelengths(
    selected_wavelengths,
    wavelengths,
    data,
    Time_array,
    t_start=12,
    t_end=13.5
):

    plt.figure(figsize=(6, 4))
    colors = ['#00ffff', '#11c638']

    Time_ns = Time_array * 1e-3

    for w, c in zip(selected_wavelengths, colors):
        idx = np.argmin(np.abs(wavelengths - w))
        y = data[idx]

        mask = (Time_ns >= t_start) & (Time_ns <= t_end)

        Time_sel = Time_ns[mask] - t_start
        y_sel = y[mask] / np.max(y[mask])

        plt.scatter(
            Time_sel,
            y_sel,
            s=100,
            color=c,
            alpha=0.9,
            edgecolors='none',
            label=f'{wavelengths[idx]:.0f} nm'
        )

    plt.xlabel('Time (ns)', fontsize=16)
    plt.ylabel('Counts (a.u.)', fontsize=16)
    plt.xlim(0, t_end - t_start)
    plt.legend(fontsize=14)
    plt.tight_layout()

    
def perform_biexp(selected_wavelengths, wavelengths, data, Timearr):

    plt.figure(figsize=(6, 4))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(selected_wavelengths)))

    for w, c in zip(selected_wavelengths, colors):
        idx = np.argmin(np.abs(wavelengths - w))
        y = data[idx]

        imax = np.argmax(y)
        y_roll = np.roll(y, -imax)
        y_roll = y_roll / np.max(y_roll)

        t_ns = Timearr * 1e-3

        # ---------- FIT ----------
        mask = t_ns > 0
        p0 = [0.6, 0.2, 0.4, 1.0]  # A1, tau1, A2, tau2 (ns)

        popt, _ = curve_fit(
            biexp,
            t_ns[mask],
            y_roll[mask],
            p0=p0,
            bounds=(0, np.inf),
            maxfev=10000
        )

        A1, tau1, A2, tau2 = popt

        label_nm = '480 nm' if int(round(w)) < 520 else '570 nm'
        legend_text = f'{label_nm}, τ1={tau1:.2f} ns, τ2={tau2:.2f} ns'

        plt.scatter(
            t_ns,
            y_roll,
            s=28,
            color=c,
            alpha=0.9,
            edgecolors='none'
        )

        t_fit = np.linspace(0, 1.5, 500)
        plt.plot(
            t_fit,
            biexp(t_fit, *popt),
            color=c,
            lw=2,
            label=legend_text 
        )

    plt.xlabel('Time (ns)', fontsize=14)
    plt.ylabel('Normalized intensity (a.u.)', fontsize=14)

    plt.xlim(-0.02, 1.5)
    plt.ylim(-0.02, 1.05)

    plt.legend(
        fontsize=11,
        frameon=False,
        labelcolor='black' 
    )

    plt.tick_params(axis='both', labelsize=12)
    plt.tight_layout()

#%%

def main():

    folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '/Users/antonellacutrupi/Desktop/code_github_adv_mat/data/double_peak/sp_res/'
    )

    time_lim = 25  # ns

    wavelengths, data, Time_array, TimeResolution, MaxRepRate, laser_divider, time_lim_index = \
        process_measurement_data(folder, time_lim, select = 0)

    selected_wavelengths = [479, 571]

    #Figure S4d)
    plot_selected_wavelengths(
        selected_wavelengths,
        wavelengths,
        data,
        Time_array
    )

    perform_biexp(
        selected_wavelengths,
        wavelengths,
        data,
        Time_array
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()
    plt.show()  # Show the plots interactively


# %%
