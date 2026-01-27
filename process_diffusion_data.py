
#%%

# Author: Antonella Cutrupi

import sys
from diff_eval_functions import *
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting

# Set up the environment for plotting

import matplotlib.font_manager as font_manager
from matplotlib.colors import LinearSegmentedColormap

from matplotlib import rcParams

font = font_manager.FontProperties(family='Arial',
                                   
                                   style='normal', size=40)

rcParams['font.family'] = font.get_name()
rcParams['font.size'] = 20
rcParams['font.style'] = 'normal'

import warnings
warnings.filterwarnings("ignore")  # Ignore warnings for cleaner output

#%%

def main():

    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '/Users/antonellacutrupi/Desktop/code_github_adv_mat/data/diffusion/')
    saveFig = False

    path, dataBinned, spatialValues, timeValuesUncut, BinningArrayTimeUncut, cutData_time, SpatialResolution = analysis_parameters(folder)

    #Figure S5c)- 4a) - S5b) - 4b) - 4c)
    Merged_0, Merged_1, timeValuesMerged = plot_diff_map_profiles_MSD(
        dataBinned,
        spatialValues,
        timeValuesUncut,
        BinningArrayTimeUncut,
        cutData_time,
        SpatialResolution,
        path,
        folder,
        saveFig
    )

    #Figure S5d) - S5e)
    plot_lifetime_trace(timeValuesMerged,Merged_0,Merged_1,path,folder,saveFig)
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()
    plt.show()  # Show the plots interactively
#%%