
#%%

# Author: Antonella Cutrupi

import numpy as np
import os
import json
import re
import glob
import time 
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator
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

def pulses_from_rep_rate(Rep_rate):

    if Rep_rate == 80:
        NPulses = 7
    elif Rep_rate == 40:
        NPulses = 4
    elif Rep_rate == 20:
        NPulses = 2
    else:
        NPulses = 1

    return NPulses

def good_bins(metadata):
    try:
        TimeRes = metadata['Time resolution (ps)']
    except:
        TimeRes = 64
    RepRate = metadata['Rep_rate (MHz)']

    picoharp_used = metadata['Picoharp used']

    if picoharp_used == 'PH300':
        NPulses = pulses_from_rep_rate(RepRate)

    elif picoharp_used == 'PH330':
        NPulses = 1
    else:
        raise Exception('Picoharp not recognized')
    
    Time_betPulse = 1/(RepRate)*10**12 # MHz to Hz and then to s, to ps
    GoodBins = NPulses * Time_betPulse / TimeRes

    return int(GoodBins)

def load_trpl_data(folder, select=0):

    npy_files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])
    
    if not npy_files:
        raise FileNotFoundError('No .npy files in folder')
    if select >= len(npy_files):
        raise IndexError(f'Select {select} out of range')

    selected_npy_file = os.path.join(folder, npy_files[select])
    base_name = os.path.splitext(npy_files[select])[0]
    selected_json_file = os.path.join(folder, base_name + '.json')

    if not os.path.exists(selected_json_file):
        raise FileNotFoundError(f"No JSON file found for {npy_files[select]}")

    data_dict = np.load(selected_npy_file, allow_pickle=True).item()
    with open(selected_json_file, 'r') as f:
        metadata = json.load(f)

    MaxRepRate = metadata.get('Rep_rate (MHz)', 80)
    laser_divider = metadata.get('laser_divider', 1)
    RepRate = MaxRepRate / laser_divider
    TimeResolution = metadata.get('Time resolution (ps)', 1)

    datArr = data_dict['datapoints']
    values = data_dict['power_values'] * 90  
    objective_transmission = 0.5
    pw_injection = values * objective_transmission  # W

    Pulse_time = 1 / (RepRate * 1e6)
    pulse_energy = pw_injection * Pulse_time

    spot_size = metadata.get('Spot_size (diam, um)', 10)
    spot_area = spot_size**2 * np.pi / 4 * 1e-8  # cm²
    fluence = pulse_energy / spot_area * 1e6  # uJ/cm²

    Time_arr = data_dict.get('Time (ps)', np.arange(datArr.shape[1]))

    return {
        'datArr': datArr,
        'power_values': pw_injection,  
        'fluence': fluence,
        'Time_arr': Time_arr,
        'metadata': metadata
    }

def plot_lifetimes(ready_dat, time_arr, powers, folder, scale = 'log', size = (10, 6), p0 = [1, 1, 1, 1, 1]):
    '''
    Plot the lifetimes of the data. The data is normalized to the max value.

    IN:
    ready_dat: data to plot
    time_arr: time array
    powers: power array

    OUT:
    plot with the lifetimes
    '''
    
    Time_arr= time_arr[:-50]/1e3

    fig, ax = plt.subplots(figsize=size)
    for i in range(len(ready_dat)):
        # fit each data to a bi-exponential function
        
        data_fit = ready_dat[i][:-50] / ready_dat[i].max() # select the data for the first power
        try:

            #popt, pcov = curve_fit(biexp_fit, Time_arr, data_fit, p0=p0, maxfev=10000)
            ax.plot(Time_arr, data_fit, label='Power: ' + str(powers[i]) + ' uW')

        except:
            print('error in plot')


    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Counts (a.u.)')
    ax.legend()

    ax.minorticks_on()

    ax.tick_params(axis='both', which='major', direction='in', bottom=True, top=True, left=True, right=True,
                   length=3, width=1, colors='k')
    ax.tick_params(axis='both', which='minor', direction='in', bottom=True, top=True, left=True, right=True,
                   length=3, width=1, colors='k')

    if scale == 'log':
        plt.yscale('log')

    else:
        plt.yscale('linear')

def process_power_data(folder):
        
    npy_files = [f for f in os.listdir(folder) if f.endswith('.npy')]
    datArr_all_list = []
    power_all_list = []

    for i in range(len(npy_files)):
        data_dict = load_trpl_data(folder, select=i)
        datArr = np.array(data_dict['datArr'])
        power = np.array(data_dict['power_values'])

        n_rows = min(len(datArr), len(power))
        datArr_all_list.append(datArr[:n_rows])
        power_all_list.append(power[:n_rows])

    datArr_all = np.concatenate(datArr_all_list, axis=0)
    power_all = np.concatenate(power_all_list)

    return datArr_all, power_all
    
def plot_intensities(ready_dat, powers, scale='log', size=(8, 6)):
    """
    Plot the integrated and peak intensities vs power, con fit legge di potenza e fit lineare (colori diversi).
    """
    powers = np.array([float(i) for i in powers])
    peak_counts = np.max(ready_dat, axis=1)

    peak_counts = peak_counts / np.max(peak_counts)

    mask = peak_counts > 1e-10
    powers_f = powers[mask]
    peak_counts_f = peak_counts[mask]

    x_fit = np.linspace(np.min(powers), np.max(powers), 200)

    fig, ax = plt.subplots(figsize=size)
    ax.plot(powers_f, peak_counts_f, '.', markersize = 15, color='tab:orange', label='Peak counts')

    ax.set_title('Power dep total range')

    plt.tight_layout()

def plot_peak_intensities_concatenated(ready_dat, powers, scale='linear',
                                       size=(10,6),
                                       remove_data=True, cut_1=0, cut_2=-1):
   
    if remove_data:
        if cut_2 == -1:
            datArr_plot = ready_dat[cut_1:]
            powers_plot = powers[cut_1:]
        else:
            datArr_plot = ready_dat[cut_1:cut_2]
            powers_plot = powers[cut_1:cut_2]
    else:
        datArr_plot = ready_dat
        powers_plot = powers

    powers_plot = np.array([float(i) for i in powers_plot])

    peak_counts = []
    n_points_baseline = 100

    for trace in datArr_plot:
        baseline = np.mean(trace[-n_points_baseline:])       
        peak = np.max(trace - baseline)                   
        peak_counts.append(max(peak, 0))                     
    peak_counts = np.array(peak_counts)
    peak_counts = peak_counts/np.max(peak_counts)

    print("Peak counts:", peak_counts)

    def pow(x, a, b):
        return a * x**b 

    x_fit = np.linspace(0, np.max(powers_plot), 200)

    fig, ax = plt.subplots(figsize=size)
    ax.plot(powers_plot*1e5, peak_counts, 'o', color='indianred', markersize=10, label='Experimental data', alpha=0.8)
    from matplotlib.ticker import MaxNLocator
    try:
        popt, _ = curve_fit(pow, powers_plot*1e5, peak_counts, p0=[1,1], bounds=([0,0],[np.inf,10]), maxfev=1000000)
        y_fit = pow(x_fit*1e5, *popt)
        ax.plot(x_fit*1e5, y_fit, '-', color='maroon', linewidth=5, label=f'power fit: exp={popt[1]:.3f}')
    except Exception as e:
        print('Linear fit failed:', e)
        popt = [np.nan]

    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlabel('Fluence ($\mu$J/cm²)', fontsize=20, labelpad=12)
    ax.set_ylabel('PL$_{0}$ (a.u.)', fontsize=20, labelpad=12)
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=18)
    ax.tick_params(axis='both', which='minor', direction='in', length=3, width=1.0)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.set_yscale(scale)
    ax.yaxis.set_major_locator(MaxNLocator(5)) 
    ax.set_title('Low power range')

    plt.xlim(np.min(powers_plot*1e5)*0.9, np.max(powers_plot*1e5)*1.05)
    plt.ylim(np.min(peak_counts[peak_counts>0])*0.9, np.max(peak_counts)*1.05)

    plt.tight_layout()

    print('power law exp:', popt[1], 'error exp', np.sqrt(np.diag(_))[1])
    return {'linear_fit_slope': popt[1]}


#%%

def main():

    folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '/Users/antonellacutrupi/Desktop/code_github_adv_mat/data/power/'
    )
    
    datArr_all, power_all = process_power_data(folder)

    plot_intensities(datArr_all, power_all, scale='linear', size=(10, 6))

    #Figure 3d)
    popt_dict = plot_peak_intensities_concatenated(
        ready_dat=datArr_all,
        powers=power_all,
        scale='linear',
        remove_data=True,
        cut_1= 24,
        cut_2=-1
    )

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()
    plt.show()  # Show the plots interactively


# %%
