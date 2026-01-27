
#%%

# Author: Antonella Cutrupi

import numpy as np
import os
import json
import re
import glob
import os
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


def read_trpl_data(file, metadata_file):

    # read metadata
    with open(metadata_file) as f:
        metadata = json.load(f)

    NPulsesSeen = pulses_from_rep_rate(metadata['Rep_rate (MHz)'])

    # assumes its a .csv with the first column being the time array and the rest the data.

    df = pd.read_csv(file)

    Time_arr = df['Time (ps)'].values

    powers = list(df.columns[2:])
    #powers = [float(i) for i in powers]

    # here it is. For each data row, perform the analysis

    ready_data = []

    for pow in powers:

        data_pw = df[pow].values
        BinsBetweenPulse = good_bins(metadata)
        # Here we need differnet analysis functions for PH300 and PH330

        if metadata['Picoharp used'] == 'PH300':
            
            # analysis for PH300. Merge pulses given the Rep rate
            for pulse_nr in range(NPulsesSeen):
                if pulse_nr == 0:
                    '''
                    FIRST POSITION
                    '''
                    counts = data_pw[:BinsBetweenPulse//NPulsesSeen]
                else:
                    '''
                    SECOND TO LAST POSITIONS
                    '''
                    counts += data_pw[BinsBetweenPulse//NPulsesSeen*pulse_nr:BinsBetweenPulse//NPulsesSeen*(pulse_nr+1)]
            
            # roll to get the max position in the first position
            max_pos = np.argmax(counts)
            counts = np.roll(counts, -max_pos)

            ready_data.append(counts) # perfect, now we have the data ready for analysis


        elif metadata['Picoharp used'] == 'PH330':

            # no merge pulses for PH330, but we need to cut a bit of the data
            counts = data_pw[1:BinsBetweenPulse] # chec up to which bin we need to cut

            # roll
            max_pos = np.argmax(counts)
            counts = np.roll(counts, -max_pos)
        
            ready_data.append(counts)
            
        else:
            raise Exception('Picoharp not recognized')
        
    Time_arr = Time_arr[:len(ready_data[0])]
        
    return np.array(ready_data), Time_arr, powers, metadata

def plot_lifetimes(ready_dat, time_arr, powers, folder, scale = 'log', size = (10, 6), save = True, filename = 'lifetimes', p0 = [1, 1, 1, 1, 1]):
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


def extract_wavelength(filename):
    base = os.path.basename(filename)
    match = re.search(r'_(480|570)_', base)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f'Wavelength not found in filename: {filename}')


def plot_intensities(ready_dat, powers, wavelength,
                     scale='linear',
                     size=(6, 4), save=False,
                     filename='intensities'):

    powers = np.array(powers)

    if wavelength == 480:
        power_cutoff = 16
    elif wavelength == 570:
        power_cutoff = 14
    else:
        power_cutoff = np.max(powers)

    mask_power = powers <= power_cutoff
    powers = powers[mask_power]
    ready_dat = np.array(ready_dat)[mask_power]

    peak_counts = np.max(ready_dat, axis=1)
    peak_counts = peak_counts / np.max(peak_counts)

    if wavelength == 480:
        color = 'cyan'
    elif wavelength == 570:
        color = '#11c638'
    else:
        color = 'k'

    fig, ax = plt.subplots(figsize=size)

    ax.scatter(powers, peak_counts,
               s=100, color=color, alpha=1,
               label=f'{wavelength} nm')

    def power_law_shift(x, a, x0, b):
        return a * np.power(np.maximum(x - x0, 0), b)

    p0 = [1, 0.1, 1]  # a, x0, b

    x_fit = np.linspace(np.min(powers), np.max(powers)+2, 400)

    popt, pcov = curve_fit(
        power_law_shift,
        powers,
        peak_counts,
        p0=p0,
        bounds=(0, np.inf),
        maxfev=20000
    )

    ax.plot(
        x_fit,
        power_law_shift(x_fit, *popt),
        '-',
        color='maroon',
        lw=2.5,
        label=rf'pow law exp $\beta= {popt[2]:.2f}$'
    )
    

    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlabel('Fluence ($\mu$J/cm²)', fontsize=16, labelpad=12)
    ax.set_ylabel('PL$_{0}$ (a.u.)', fontsize=16, labelpad=12)
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=18)
    ax.tick_params(axis='both', which='minor', direction='in', length=3, width=1.0)
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.set_yscale(scale)
    ax.yaxis.set_major_locator(MaxNLocator(5)) 

    plt.xlim(np.min(powers)*0.5, np.max(powers)*1.05)
    plt.ylim(np.min(peak_counts[peak_counts>0])*0.5, np.max(peak_counts)*1.05)
    plt.legend()
    plt.tight_layout()

    print(f'Power-law exponent b ({wavelength} nm): {popt[2]:.3f} ± {np.sqrt(np.diag(pcov))[2]:.3f}')

    return popt, pcov

def process_data_and_plot(folder):
            
    files = glob.glob(folder + '*.csv')
    metadata_files = glob.glob(folder + '*.json')

    for file, metadata_file in zip(files, metadata_files):

        print(file)

        wavelength = extract_wavelength(file)

        ready_data, Time_arr, powers_, _ = read_trpl_data(file, metadata_file)

        powers_ = np.array([float(i) for i in powers_])

        order = np.argsort(powers_)
        ready_data = [ready_data[i] for i in order]
        powers_ = powers_[order]

        '''
        plot_lifetimes(
            ready_data, Time_arr, powers_,
            folder, scale='linear',
            size=(7, 5),
            save=False,
            p0=[0.01, 1]
        )
        '''
        popt, pcov = plot_intensities(
        ready_data,
        powers_,
        wavelength=wavelength,
        scale='linear',
        size=(7, 5),
        save=False,
        filename='power_dep_'
    )
        

#%%

def main():

    folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '/Users/antonellacutrupi/Desktop/code_github_adv_mat/data/double_peak/pow_dep/'
    )

    #Figure S4e)- S4f)
    process_data_and_plot(folder)

   
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()
    plt.show()  # Show the plots interactively


# %%
