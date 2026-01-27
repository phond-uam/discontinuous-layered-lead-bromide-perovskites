
'''
HELP FUNCTIONS TO BE USED IN THE EVALUATION AND INTERPRETATION OF THE DATA EXTRACTED FROM AN APD IN THE DIFFUSION
MEASUREMENTS.

4 TYPES OF FUNCTIONS. 

0 - IMPORTS

1 - GENERAL USE: LOADING DATA, GENERAL FITTING, BINNING ETC

2 - DATA TREATMENT

3 - MIXING DATA FROM MANY MEASUREMENTS

4 - DIFFUISIVITY FITTING

'''

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import curve_fit #For fitting
from scipy.special import wofz
from scipy import interpolate
from bisect import bisect_left
import statsmodels.api as sm
import pandas as pd
import os 
import glob
import json

'''
LOADING DATA FROM FILE, GETS ALL THE FILES IN A PATH AND PUTS EACH INTO A ARRAY THAT IS INSIDE A BIG LIST
'''
def LoadData(paths):
    '''
    PRELOCATE
    '''
    datas = []
    '''
    LOOPING FOR EACH FILE
    '''
    for name in paths:
        '''
        DATAS IS A N-LENGTH ARRAY WHERE EACH ARRAY HAS ALL THE DATA FROM ONE CSV FILE
        '''
        datas.append(np.loadtxt(name,delimiter = ','))
    return datas
#

def fwhm(p1, xp1 = [-1], averaging = 'average', factor = 0.5):
    def interpolate(x, xp, fp): #interpolation function
        return fp[0]+(x-xp[0])*(fp[1]-fp[0])/(xp[1]-xp[0]) 
    if len(xp1) == 1: xp1 = [i for i in range(len(p1))] #If no xp1 is given we just take the indices of p1
    maximum, max_index = max([p1[i],i] for i in range(len(p1)))
    goal = maximum*factor
    lefts = [] #array to store the values on the left of the maximum where the goal is crossed
    rights = [] #array to store the values on the right of the maximum where the goal is crossed 
    for i, value in enumerate(p1[:-1]): #find all the 0.5 crossings
        if p1[i] <= goal and p1[i+1] > goal: #crossing below 0.5 above 0.5
            if i < max_index: #crossing on left side
                lefts.append(interpolate(goal,[p1[i],p1[i+1]],[xp1[i],xp1[i+1]])) 
            else: #crossing on right side
                rights.append(interpolate(goal,[p1[i],p1[i+1]],[xp1[i],xp1[i+1]]))
        elif p1[i] >= maximum*factor and p1[i+1] < maximum*factor: #crossing above 0.5 to below 0.5
            if i < max_index:
                lefts.append(interpolate(goal,[p1[i],p1[i+1]],[xp1[i],xp1[i+1]]))
            else:
                rights.append(interpolate(goal,[p1[i],p1[i+1]],[xp1[i],xp1[i+1]]))
    #Averaging the found values Here we can choose either the arithmetic 'mean' or the 'median'
    if averaging == 'median':
        left = np.median(lefts)
        right = np.median(rights)
    else:
        left = np.mean(lefts)
        right = np.mean(rights)

    return abs(right-left) #If you want to get back the index of the fwhm position 

#Voigt Function taking the height, center position, hwhm of the gaussian (alpha) and hwhm of the lorenzian (gamma)
def Voigt(x, height, center, sigma, gamma):
#    sigma = alpha / np.sqrt(2 * np.log(2))
#    print(x.shape, height.shape, sigma.shape, gamma.shape)
    return height*np.real(wofz(((x-center) + 1j*gamma)/sigma/np.sqrt(2))) / sigma / np.sqrt(2*np.pi)

def Gauss(x, height, center, sigma):
#    sigma = alpha / np.sqrt(2 * np.log(2))
    return height*np.exp(-(x-center)**2/(2*sigma**2))

#Voigt functions can change their height and sigma
def VoigtReduced2d(x, center, gamma, offset, *args):
    #Assigning the fitting parameters to heights and sigmas
    heights = args[:int(len(args)/2)]
    sigmas = args[int(len(args)/2):]
    #Make those fitting parameters accessible for a 2d fit
    temp, heights = np.meshgrid([0]*x.shape[1],heights)
    temp, sigmas = np.meshgrid([0]*x.shape[1],sigmas)
    return (Voigt(x, heights, center, sigmas, gamma)+offset).ravel() #ravel is needed for the 2d fit

def VoigtReduced2dNonlinearBinning(x, center, gamma, offset, *args):
    #Assigning the fitting parameters to heights and sigmas
    heights = args[:int(len(args)/2)]
    sigmas = args[int(len(args)/2):]
    #Make those fitting parameters accessible for a 2d fit
    temp, heights = np.meshgrid([0]*x.shape[1],heights)
    temp, sigmas = np.meshgrid([0]*x.shape[1],sigmas)
    temp, offsets = np.meshgrid([0]*x.shape[1],offset*BinningArrayTime)
    return (Voigt(x, heights, center, sigmas, gamma)+offsets).ravel() #ravel is needed for the 2d fit
    
def VoigtReduced2dNormalized(x, center, gamma, offset, *args):
    #Assigning the fitting parameters to heights and sigmas
    heights = args[:int(len(args)/2)]
    sigmas = args[int(len(args)/2):]
    #Make those fitting parameters accessible for a 2d fit
    temp, heights = np.meshgrid([0]*x.shape[1],heights)
    temp, sigmas = np.meshgrid([0]*x.shape[1],sigmas)
    temp, offsets = np.meshgrid([0]*x.shape[1],offset/Maxs)
    return (Voigt(x, heights, center, sigmas, gamma)+offsets).ravel() #ravel is needed for the 2d fit

def hwhmVoigt(gamma, alpha):
    return 0.5346*gamma + (0.2166*gamma**2 + alpha**2)**0.5

def mergingImagesThroughMiddleAlignment(datas, norm = True):
    if len(datas) == 1:#Return array directly if only one is given
        return np.array(datas[0])/np.array(datas[0]).max()
    #Transpose the arrays for easier handling (having space in rows)
    for i, data in enumerate(datas):
        datas[i] = np.array(data).transpose()
    #Normalize the arrays if norm = True. This will lead that all the arrays are weighted equally. If it is false data with more counts is weighted more
    if norm:
        for i, data in enumerate(datas):
            data = np.array(data)
            datas[i] = data/data.max()
    #Find the maxima of the curves through fitting with a voigt function
    MaxList = []
    Middle = (len(datas[0])-1)/2
    MaxMostMiddleIndex = 0
    MaxMostMiddleValue = 1e10
#    plt.figure()
    for i, data in enumerate(datas):
        sum_temp = np.sum(data,axis=1) #Integrate data in time to have the profile
        Xtemp = np.arange(len(sum_temp)) #Make the index array for the xValues
#        plt.plot(Xtemp,sum_temp)
#        plt.plot(Xtemp,Voigt(Xtemp,*[max(sum_temp),Middle,3,3]))
        popt_temp, success_temp = curve_fit(Voigt, Xtemp, sum_temp/max(sum_temp), [max(sum_temp),Middle,3,3]) #Fit Voigt
        MaxList.append(popt_temp[1]) #Save Max
        if abs(MaxList[-1]-Middle) < abs(MaxMostMiddleValue-Middle): #Keep track of which max value is closest to the middle
            MaxMostMiddleIndex = i
            MaxMostMiddleValue = MaxList[-1]
    #Merge the arrays. We start with the array, which has its Masimum closest to the Middle add the others on top        
    Merged = datas[MaxMostMiddleIndex]
#    plt.figure()
    for i, data in enumerate(datas):
        if i != MaxMostMiddleIndex: #We do not need to add that one again
            y = np.linspace(0, data.shape[0]-1, data.shape[0]) #Index of rows
            x = np.linspace(0, data.shape[1]-1, data.shape[1]) #Index of columns
            ynew = y - (MaxList[i] - MaxMostMiddleValue) #Shift the current array to the same maxPosition as the other ones
            # interp2d is not supported 
            #dataInterpolated = interpolate.interp2d(x, ynew, data) #Make a funciton from the current array using the shifted y-vector
            # interpolate using RectBivariateSpline
            r = interpolate.RectBivariateSpline(x, y, data.T)
            dataInterpolated = r(x, ynew).T

            Merged = Merged + dataInterpolated
        Xtemp = np.arange(len(np.sum(data,axis=1))) #Make the index array for the xValues
        sum_temp = np.sum(data,axis=1) #Integrate data in time to have the profile
#        plt.plot(Xtemp-(MaxList[i] - MaxMostMiddleValue),sum_temp/max(sum_temp),label=i)
    #Cutting the data where not all of the curves overlapped
    differences = MaxList - MaxMostMiddleValue
    cutStart = min(differences) #This should be negative if there was a curve that was shifted to the right
    cutEnd = max(differences) #This should be positive if there was a curve that was shifted to the left
    if cutStart < 0: #Check if there was a curve shifted to the right
        shift = int(np.ceil(abs(cutStart)))
        Merged = Merged[shift:]
    if cutEnd > 0: #Check if there was a curve shifted to the left
        shift = int(np.ceil(abs(cutEnd)))
        Merged = Merged[:-shift]
#    plt.plot(np.arange(len(np.sum(Merged,axis=1)))+int(np.ceil(abs(cutStart))), np.sum(Merged,axis=1)/max(np.sum(Merged,axis=1)),label='Merged')
#    plt.legend()
    return np.array(Merged).transpose()/len(datas) #normalize with the number of merged arrays and transpose back to have time in the rows



'''
BINNING FUNCTIONS 
'''

def ConstantBins(x,c):
    return c

def PowerLawForBins(x,a,b,c):
    return a*x**b + c

'''
How the analysis is done:

GET THE RAW DATA. 
N NUMBER OF MEASUREMENTS. LEN() = N

~10000 BINS PER ROW (ACQUISITION FOR CERTAIN TIMES PER POSITION)
EVERY NUMBER OF BINS (NPulsesSeen DETERMINES IT), NEW PULSE, SO EVERY ROW HAS NPulsesSeen PULSES...
steps ROWS (~10000 BINS FOR EACH SPATIAL POSITION...)

Dat_OnePeak_unordered/ordered

1 MEASUREMENT. LEN() = 1

NEXT WE SUM ALL THE PULSES TOGETHER TO GET DATA WITH THE SHAPE:
~3000 BINS PER ROW (EACH newBIN IS ALREADY SUM OF NPulsesSeen BINS)
steps ROWS (FOR EACH SPATIAL POSITION)

Dat_OnePeak_All

SAME THING, BUT WE PUT THEM IN A STRING WITH N MEASUREMENTS...

N MEASUREMENTS. LEN() = N
~3000 BINS PER ROW (EACH newBIN IS ALREADY SUM OF NPulsesSeen BINS)
steps ROWS (FOR EACH SPATIAL POSITION)

time_merged IS CALCULATED AS TimeResolution * BinsBetweenPulses[ARRAY]


ADD PEAKS TOGETHER...

mergePulses

PROBLEM HERE IS WE DO NOT KNOW WHERE THE MAX IS... (unordered)
FIND THE MAX AND PUT IT AT THE VERY BEGINNING OF THE ARRAY

np.argmax() -> np.roll()	

RMS CAN BE ADDED (MICHAEL'S) TO ELIMINATE SOME NOISE ...
BINNING IN SPACE... (WE DON'T DO SPATIAL BINNING RN, BUT I AM ADDING JUST IN CASE). CONSTRUCTION ... 
DataBinnedSpace = []
for array in Dat_OnePeak_All:
    DataBinnedSpace.append(np.array(space_binning(array, binning_distance)))

'''



'''
FOR ONE CSV FILE.
GOES ONE LINE AT A TIME (ONE SPATIAL POSITION), DIVIDES EACH ROW FOR EVERY TIME THE LASER SEES A PULSE AND ADDS EVERY SLICE TOGETHER

EX:
DATA=
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 (20 COLUMNS IN A ROW, EACH WITH COUNTS)

PULSE EVERY 5 COLUMNS
-----------------------------------------------------
A PULSE FOR EACH OF THESE

0-1-2-3-4
5-6-7-8-9
10-11-12-13-14
15-16-17-18-19
------------------------------------------------------
ADDS T0 GET

30 - 34 - 38 - 42 - 46
-------------------------------------------------------
CALCULATE TIME ARRAY AS

ARANGE(1/RepRate) * TimeResolution 

'''

def mergePulses (data, TimeResolution, Divider, NumberOfPulsesSeen, MaxRepetitionRate = 80e6):
    '''
    CALCULATE THE TIME ARRAY FIRST...

    FIND REPETITION RATE, TIME BETWEEN PULSE AND HOW MANY BINS ARE BETWEEN PULSES
    '''
    TimeResolution = TimeResolution*1e-3 #ns

    RepRate = MaxRepetitionRate / Divider
    TimeBetweenPulse = (1/RepRate)*1e9 #ns
    BinsBetweenPulse = int(TimeBetweenPulse/TimeResolution) #ns
    '''
    ARRAY OF BINS UNTIL NEXT PULSE TRANSFORMED INTO TIME (*TimeResolution)
    '''
    time = np.arange(BinsBetweenPulse)*TimeResolution
    
    '''
    MERGE THE COUNTS OF THE PULSES INTO A SINGLE PULSE

    PRELOCATE
    '''
    data_merged = []
    '''
    EACH ROW REPRESENT A TIME SLICE
    '''
    for row in data:
        '''
        FOR EVERY SPATIAL POSITION IT DIVIDES THE DATA INTO NumberOfPulsesSeen SLICES
        MERGE ALL OF THEM TO CREATE THE HISTOGRAM FOR A SINGLE (AVERAGED) PULSE
        '''
        for pulse_nr in range(NumberOfPulsesSeen):
            if pulse_nr == 0:
                '''
                FIRST POSITION
                '''
                counts = row[:BinsBetweenPulse]
            else:
                '''
                SECOND TO LAST POSITIONS
                '''
                counts += row[BinsBetweenPulse*pulse_nr:BinsBetweenPulse*(pulse_nr+1)]
            #
        data_merged.append(counts)
    #
    return data_merged, time

def merge_Pulses (data, TimeResolution, Divider, NumberOfPulsesSeen, MaxRepetitionRate = 80e6):
    '''
    CALCULATE THE TIME ARRAY FIRST...

    FIND REPETITION RATE, TIME BETWEEN PULSE AND HOW MANY BINS ARE BETWEEN PULSES
    '''
    TimeResolution = TimeResolution*1e-3 #ns

    RepRate = MaxRepetitionRate / Divider
    TimeBetweenPulse = (1/RepRate)*1e9 #ns
    BinsBetweenPulse = int(TimeBetweenPulse/TimeResolution) #ns
    '''
    ARRAY OF BINS UNTIL NEXT PULSE TRANSFORMED INTO TIME (*TimeResolution)
    '''
    time = np.arange(BinsBetweenPulse)*TimeResolution
    
    '''
    MERGE THE COUNTS OF THE PULSES INTO A SINGLE PULSE

    PRELOCATE
    '''
    data_merged = []
    '''
    EACH ROW REPRESENT A TIME SLICE
    '''
    for row in data:
        '''
        FOR EVERY SPATIAL POSITION IT DIVIDES THE DATA INTO NumberOfPulsesSeen SLICES
        MERGE ALL OF THEM TO CREATE THE HISTOGRAM FOR A SINGLE (AVERAGED) PULSE
        '''
        for pulse_nr in range(NumberOfPulsesSeen):
            if pulse_nr == 0:
                '''
                FIRST POSITION
                '''
                counts = row[:BinsBetweenPulse]
            else:
                '''
                SECOND TO LAST POSITIONS
                '''
                counts += row[BinsBetweenPulse*pulse_nr:BinsBetweenPulse*(pulse_nr+1)]
            #
        data_merged.append(counts)
    #
    return data_merged, time

'''
Not being used so dont worry about it
'''

def space_binning(x, bins = 2, keep_last_pixels = True):
    bins = int(bins)
    binned_x = []
    if bins == 1:
        return x
    elif bins < 1: 
        print('Number of bins needs to be a positive integer.')
        return None
    elif bins > len(x): 
        print('The binning window is larger than the array.')
        return None
    else:
        for i, element in enumerate(x):
            if bins*(i+1) >= len(x):
                if keep_last_pixels == True: #If this is true the remaining pixel at the end of the array are binned (but the number of pixels is less than bins). If it is false the pixels are thrown away..
                    binned_x.append(sum(x[bins*i:])/len(x[bins*i:]))#if at the end less than #bins are left just add these together.
                return binned_x
            binned_x.append(sum(x[bins*i:bins*(i+1)])/bins)

'''
time binning. 

INPUT:
x = array of data
function = function to be used for binning
functionparams = parameters for the function
keep_last_pixels = if the last pixels are binned or not
returnBinningArray = if the binning array is returned or not

OUTPUT:
binned_x = binned data
binning_array = array of bins used for binning
'''

def binningNonlinear(x, function, functionparams, keep_last_pixels = True, returnBinningArray = False):
    #Make the Binning Array
    binning_array = []
    i = 0
    while np.sum(binning_array) < len(x):
        binning_array.append(int(function(i,*functionparams)))
        i += 1
    binning_array = np.array(binning_array)
    #Check viability of Binning Array and do the binning
    binned_x = []
    if (binning_array < 1).any(): 
        print('Number of bins need to be a positive integer.')
        return None
    elif (binning_array == 1).all():
        if returnBinningArray == True:
            return x, returnBinningArray
        else:
            return x
    elif binning_array[0] > len(x): 
        print('The first binning window is larger than the array.')
        return x
    else: #Do the binning
        for i, element in enumerate(x):
            if sum(binning_array[:i+1]) >= len(x): #Check if the whole data x has already been binned
                if keep_last_pixels == True: #If this is true the remaining pixel at the end of the array are binned (but the number of pixels is less than bins). If it is false the pixels are thrown away..
                    #if at the end less than #bins are left just add these together
                    binned_x.append(sum(x[sum(binning_array[:i]):])/len(x[sum(binning_array[:i]):]))
                if returnBinningArray:
                    return binned_x, binning_array
                else:
                    return binned_x
            binned_x.append(sum(x[sum(binning_array[:i]):sum(binning_array[:i+1])])/binning_array[i])
    

def binningNonlinearErrors(x, function, functionparams, keep_last_pixels = True, returnBinningArray = False):
    #Make the Binning Array
    binning_array = []
    i = 0
    while np.sum(binning_array) < len(x):
        binning_array.append(int(function(i,*functionparams)))
        i += 1
    binning_array = np.array(binning_array)
    #Check viability of Binning Array and do the binning
    binned_x = []
    if (binning_array < 1).any():
        print('Number of bins need to be a positive integer.')
        return None
    elif (binning_array == 1).all():
        if returnBinningArray == True:
            return x, returnBinningArray
        else:
            return x
    elif binning_array[0] > len(x): 
        print('The first binning window is larger than the array.')
        if returnBinningArray == True:
            return x, returnBinningArray
        else:
            return x
    else: #Do the binning
        for i, element in enumerate(x):
            if sum(binning_array[:i+1]) >= len(x): #Check if the whole data x has already been binned
                if keep_last_pixels == True: #If this is true the remaining pixel at the end of the array are binned (but the number of pixels is less than bins). If it is false the pixels are thrown away..
                    #if at the end less than #bins are left just add these together
                    binned_x.append(np.sqrt(sum([value**2 for value in x[sum(binning_array[:i]):]]))/len(x[sum(binning_array[:i]):]))
                if returnBinningArray:
                    return binned_x, binning_array
                else:
                    return binned_x
            binned_x.append(np.sqrt(sum([value**2 for value in x[sum(binning_array[:i]):sum(binning_array[:i+1])] ]))/binning_array[i])

'''
Normalize the data

for each delay, the data is normalized by the maximum value of each row
'''

def Normalize(MergedImage):
    normalizeddata = []
    for dat in MergedImage:
        normalized = (dat)/(np.max(dat))
        normalized = np.array(normalized)
        normalizeddata.append(normalized)
    return normalizeddata

'''
fwhm function

extracts the fwhm of a curve assuming it has voigt/gaussian shape
'''

def fwhm(p1, xp1 = [-1], averaging = 'average', factor = 0.5):
    def interpolate(x, xp, fp): #interpolation function
        return fp[0]+(x-xp[0])*(fp[1]-fp[0])/(xp[1]-xp[0]) 
    if len(xp1) == 1: xp1 = [i for i in range(len(p1))] #If no xp1 is given we just take the indices of p1
    maximum, max_index = max([p1[i],i] for i in range(len(p1)))
    goal = maximum*factor
    lefts = [] #array to store the values on the left of the maximum where the goal is crossed
    rights = [] #array to store the values on the right of the maximum where the goal is crossed 
    for i, value in enumerate(p1[:-1]): #find all the 0.5 crossings
        if p1[i] <= goal and p1[i+1] > goal: #crossing below 0.5 above 0.5
            if i < max_index: #crossing on left side
                lefts.append(interpolate(goal,[p1[i],p1[i+1]],[xp1[i],xp1[i+1]])) 
            else: #crossing on right side
                rights.append(interpolate(goal,[p1[i],p1[i+1]],[xp1[i],xp1[i+1]]))
        elif p1[i] >= maximum*factor and p1[i+1] < maximum*factor: #crossing above 0.5 to below 0.5
            if i < max_index:
                lefts.append(interpolate(goal,[p1[i],p1[i+1]],[xp1[i],xp1[i+1]]))
            else:
                rights.append(interpolate(goal,[p1[i],p1[i+1]],[xp1[i],xp1[i+1]]))
    #Averaging the found values Here we can choose either the arithmetic 'mean' or the 'median'
    if averaging == 'median':
        left = np.median(lefts)
        right = np.median(rights)
    else:
        left = np.mean(lefts)
        right = np.mean(rights)

    return abs(right-left) #If you want to get back the index of the fwhm position 

'''
MERGING DATA FROM MANY MEASUREMENTS, directly from Michaels code
'''


#Voigt Function taking the height, center position, hwhm of the gaussian (alpha) and hwhm of the lorenzian (gamma)
def Voigt(x, height, center, sigma, gamma):
#    sigma = alpha / np.sqrt(2 * np.log(2))
#    print(x.shape, height.shape, sigma.shape, gamma.shape)
    return height*np.real(wofz(((x-center) + 1j*gamma)/sigma/np.sqrt(2))) / sigma / np.sqrt(2*np.pi)


def extract_parameters(path_param):
    '''
    simply gets parameters from the txt file.
    IN:
    path_param = path to the txt file

    OUT:
    many parameters, dont worry about it
    '''

    with open(path_param[0], "r") as f:
        params = f.read()
 
    '''
    DEFINING THE PARAMETERS. I AM EXTRACTING THEM FROM THE TXT FILE...
    '''
    TimeResolution = float(params[params.find('_ns_')-5:params.find('_ns_')])
    AcquisitionTime = float(params[params.find('_ns_')+20:params.find('_ms_')])
    Binning = int(params[params.find('_laserPower_')-1:params.find('_laserPower_')])
    laser_pw = float(params[params.find('laserPower_')+12:params.find('_uW_')])
    laser_div = int(params[params.find('_steps_')-1:params.find('_steps_')])
    range_x = params[params.find('_x_range_')+9:params.find('_y_range_')-3]
    range_y = params[params.find('_y_range_')+9:params.find('_mm_objective_')]
    steps = int(params[params.find('_steps_')+7:params.find('_x_range_')])
    objective = params[params.find('_objective_')+11:params.find('_objective_')+14]
    objective = objective.replace('x','')
    objective = int(objective)

    #MaxRepRate = 80e6 # 80 MHz if INT 1
    MaxRepRate = 1e6 # 80 MHz if INT 2
    


    if laser_div !=1:
        NPulsesSeen = int(8/laser_div)
    else:
        NPulsesSeen = int(7)
    '''
    Not ready 4 16 or more divider...

    if laser_div == 1:
        NPulsesSeen = int(7)
    elif laser_div == 2 or laser_div == 4 or laser_div == 8:
        NPulsesSeen = int(8/laser_div)
    else:
        NPulsesSeen = 1
   
    '''
    

    xmin = float( range_x[1:range_x.find(',')])
    xmax = float( range_x[range_x.find(',')+1:-1])
    ymin = float( range_y[1:range_y.find(',')])
    ymax = float( range_y[range_y.find(',')+1:-1])

    '''
    EXTRACTING THE APD RANGE FROM THE INITIAL AND LAST POSITION
    CALCULATING THE SPATIAL RESOLUTION AS:

    (APD_range/total_magnification)/steps

    '''     
    APD_range = np.linalg.norm(np.array([xmin,ymin]) - np.array([xmax,ymax])) 
#    APD_range = np.abs(xmax-xmin) 
    

    lens_magnification = 3.636363633333333   
    total_magnification = lens_magnification * objective
    sample_range = APD_range / total_magnification
    SpatialResolution = sample_range / steps # mm

    return TimeResolution, AcquisitionTime, Binning, laser_pw, laser_div, NPulsesSeen, xmin, xmax, ymin, ymax, SpatialResolution

def plot_all_measurements(dataBinnedNorm, spatialValues, spatialResolution,timeValuesUncut, 
                          time_end = None, saveFig = False, folder = None):

    '''
    Doing two plots, 2D maps for indivifual measurements and profiles at t = 0

    IN:
    dataBinnedNorm = list of arrays with the data
    spatialValues = array with the spatial values, spatialResolution = resolution for the APD
    timeValuesUncut = array with the time values
    saveFig = if the figures are saved or not
    folder = folder where the figures are saved

    OUT:
    plots
    '''

    for i, array in enumerate(dataBinnedNorm[:]):
        plt.figure()
        plt.title(str(i))
        plt.pcolormesh(spatialValues,timeValuesUncut,array,cmap = "plasma")
        plt.gca().invert_yaxis()
        plt.xlabel('distance (um)')
        plt.ylabel('time (ns)')
        # ct at time_end
        if time_end is not None:
            plt.ylim(0,time_end)
        if saveFig == True:
            plt.savefig(folder+str(i)+'_spectrum.svg', dpi=300)
    #    plt.close()

    plt.figure()
    for i, array in enumerate(dataBinnedNorm):
        plt.title('Profiles at t = 0')
        plt.plot((np.arange(len(array[0]))-(len(array[0])-1)/2)*spatialResolution,array[0], label = i)
        plt.xlabel('distance (um)')
        plt.ylabel('normalized counts (a.u.)')
        plt.legend(loc = 'best')
        if saveFig == True:
            plt.savefig(folder+str(i)+'profile_t0.svg', dpi=300)
    #    plt.close()
        print(fwhm(array[0])*spatialResolution*1e3)

def merge_selected_data(dataBinned, arraySelection, cutData_time, timeValuesUncut, BinningArrayTimeUncut, spatialResolution):
    '''
    Grabs the selected data and merges them (cuts data to cutData), also normalizes the data

    IN:
    dataBinned = list of arrays with the data
    arraySelection = array with the selected arrays
    cutData_time = list with the initial and final time for the cut
    timeValuesUncut = array with the time values
    BinningArrayTimeUncut = array with the binning array for the time values
    spatialResolution = resolution for the APD
    
    OUT:
    Merged = merged data
    MergedNormalized = merged and normalized data
    spatialValues = spatial values, new array
    timeValuesMerged = time values, new array
    BinningArrayTime = binning array for the time values
    '''
    
    #Find index closest to indicated times
    cutData = [(np.abs(timeValuesUncut - t)).argmin() for t in cutData_time]
    #cutData = [2,41*2]

    #Select the selected arrays
    DataBinnedSelection = [dataBinned[i] for i in arraySelection]

    #Flip every second array
    for i, array in enumerate(DataBinnedSelection):
        if i%2 == 0:
            DataBinnedSelection[i] = np.flip(array,axis=1)

    #Merge the arrays
    Merged = mergingImagesThroughMiddleAlignment(DataBinnedSelection)[cutData[0]:cutData[1]]
    #Normalize the merged data
    MergedNormalized = np.array(Normalize(Merged))

    spatialValues = (np.arange(Merged.shape[1]) - np.argmax(sum(Merged)))*spatialResolution
    timeValuesMerged = timeValuesUncut[cutData[0]:cutData[1]]
    timeValuesMerged = timeValuesMerged - min(timeValuesMerged)
    BinningArrayTime = BinningArrayTimeUncut[cutData[0]:cutData[1]]

    return Merged, MergedNormalized, spatialValues, timeValuesMerged, BinningArrayTime

def plot_merge_lifetimes(Merged, timeValuesMerged, timeSlice ,saveFig = False, folder = None):
    '''
    Plots the lifetime trace of the merged data

    IN:
    Merged = merged data
    timeValuesMerged = time values
    timeSlice = slice of time values to be plotted
    saveFig = if the figures are saved or not
    folder = folder where the figures are saved

    OUT:
    plots
    '''

    #Plotting the lifetime trace of the merged measurements
    DataSummed = np.sum(Merged, axis=1) # Intensity variable

    #figureLifetime = plt.figure('Lifetime')
    #plt.figure(figureLifetime.number)

    plt.figure()
    plt.xlabel('time (ns)')
    plt.ylabel('counts')
    plt.semilogy(timeValuesMerged, DataSummed/DataSummed.max(), 'o', label='')
    #plt.legend()
    #plt.close()
    if saveFig == True:
        plt.savefig(folder+'lifetime.svg', dpi=300)

    #Plotting the lifetime trace of the merged measurements
    DataSummed = np.sum(Merged, axis=1)

    #figureLifetime = plt.figure('Lifetime')
    #plt.figure(figureLifetime.number)
    plt.figure()
    plt.xlabel('time (ns)')
    plt.ylabel('counts')
    for i in timeSlice:
        plt.semilogy(timeValuesMerged, Merged[:,i]/Merged[:,i].max(), label=str(np.round(timeValuesMerged[i],2)) + ' ns')
    plt.legend()
    #plt.close()
    if saveFig == True:
        plt.savefig(folder+'lifetime_slices.svg', dpi=300)

    return DataSummed


def plot_merge_lifetimes_fit(Merged, timeValuesMerged, timeSlice, scatter_color='#1f77b4', fit_color='darkred',saveFig = False, folder = None, img = True):
    '''
    Plots the lifetime trace of the merged data

    IN:
    Merged = merged data
    timeValuesMerged = time values
    timeSlice = slice of time values to be plotted
    saveFig = if the figures are saved or not
    folder = folder where the figures are saved

    OUT:
    Intensity variable
    plots
    '''

    #Plotting the lifetime trace of the merged measurements
    DataSummed = np.sum(Merged, axis=1) # Intensity variable

    # three plots lin-lin, log-lin and log-log.
    # lin lin has the single exponential fit

    def double_exp(x, a1, t1, a2, t2):
        return a1*np.exp(-x/t1) + a2*np.exp(-x/t2)
    
    popt, pcov = curve_fit(double_exp, timeValuesMerged, DataSummed/DataSummed.max(), p0 = [1, 0.5, 0.1, 5])
    # Intensity is normalized.

    if img:
        fig, ax = plt.subplots(figsize = (6,6))

        plt.scatter(timeValuesMerged, DataSummed/DataSummed.max(), facecolors='None',edgecolors=scatter_color, s = 50)
        ax.plot(timeValuesMerged, double_exp(timeValuesMerged, *popt), color = fit_color, linestyle = '--', linewidth = 2.5)

        ax.set_xlabel('Time (ns)', fontsize=16)

        ax.set_ylabel('Intensity (a.u.)', fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xlim(-0.1,5)
        plt.text(0.6, 0.7, f'a1={popt[0]:.2f}\ntau1={popt[1]:.2f} ns\na2={popt[2]:.2f}\ntau2={popt[3]:.2f} ns', transform=ax.transAxes, fontsize=14,bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))

        print('Fitted parameters (a1, tau1, a2, tau2): ', popt)

        plt.savefig(folder+'lifetime_c_axis.svg', dpi=600)

    return DataSummed, [popt], [pcov]


'''
After merging the data with the selected arrays, 
plot the merged 2d map + gaussian fits for the slices
also return the fitting parameters

IN:
Merged = merged data
spatialValues = spatial values
timeValuesMerged = time values
BinningArrayTime = binning array for the time values
arraySelection = selected arrays
SpatialResolution = resolution for the APD
saveFig = if the figures are saved or not
folder = folder where the figures are saved

OUT:
plots
popt, success = fitting parameters
'''

def plot_merge_diffusion_map(Merged, spatialValues, timeValuesMerged, BinningArrayTime, arraySelection, SpatialResolution,cut_time,time_slice_ns = None, cmap = None, saveFig = False, folder = None, img = True):
    
    '''
    Auxiliary function for the fitting to voigt function, also from Michaels code
    '''

    def VoigtReduced2dNonlinearBinning(x, center, gamma, offset, *args):
        #Assigning the fitting parameters to heights and sigmas
        heights = args[:int(len(args)/2)]
        sigmas = args[int(len(args)/2):]
        #Make those fitting parameters accessible for a 2d fit
        temp, heights = np.meshgrid([0]*x.shape[1],heights)
        temp, sigmas = np.meshgrid([0]*x.shape[1],sigmas)
        temp, offsets = np.meshgrid([0]*x.shape[1],offset*BinningArrayTime)
        return (Voigt(x, heights, center, sigmas, gamma)+offsets).ravel() #ravel is needed for the 2d fit
    
    xOffset = SpatialResolution/2

    from matplotlib.ticker import MaxNLocator

    Normalized_Merged = (Merged - Merged.min())/(Merged.max() - Merged.min())

    #Plotting the figure of the merged measurements
    if img:
        fig, ax = plt.subplots(figsize=(6,6))
        CS = ax.contourf(Normalized_Merged, levels=20, cmap='magma', 
                interpolation='bilinear',
             extent=[spatialValues[0]*1e3, spatialValues[-1]*1e3, timeValuesMerged[0], timeValuesMerged[-1]],
             location='lower', origin='lower')
        plt.pcolormesh(spatialValues-xOffset,timeValuesMerged,Normalize(Merged-Merged.min()*0),cmap = "magma",
        linewidth=0., edgecolors='black', rasterized=True)
        #plt.gca().invert_yaxis()
        plt.ylim(0,4.5)
        plt.xlim(-1.5,1.5)

        plt.xlabel('Position ($\mu$m)',fontsize=16)
        plt.ylabel('Time (ns)', fontsize = 16)
        #plt.title('Merged arrays: '+str(arraySelection))
        
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))   
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        plt.tick_params(labelsize=12)

        cbar = plt.colorbar(CS, ax=ax, orientation='horizontal', pad=0.02, aspect=50,location ='top', )
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
        cbar.ax.tick_params(labelsize=12)
        # Move the colorbar to the top of the plot
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        #title colorbar
        cbar.set_label('Normalized Intensity(a.u)', fontsize=14)
        #position title on top of colorbar
        cbar.ax.xaxis.set_label_coords(0.5, 5)
    
        if saveFig == True:
            plt.savefig(folder+str(arraySelection[:])+'merged.pdf', dpi=3000)

 
    #Making the fits
        #Get the normalization factor for each row (needed to account for the offset correctly)
        #Afterwards, we will fit the normalized data in order to give later times the same importance as earlier times
    Maxs = []
    for i, row in enumerate(Merged):
        Maxs.append(max(Merged[i]))
    Maxs = np.array(Maxs)
        #Constructing the bound arrays - def VoigtReduced2d(x, center, gamma, offset, *heights, *sigmas):
    lowBounds = [-100000]+[0]+[0]+[0]*Merged.shape[0]*2
    highBounds = [10000000]+[1000000000]+[1000000000]+[10000000]*Merged.shape[0]*2
        #Making the position values for the 2D function
    X, temp = np.meshgrid(spatialValues, [0]*Merged.shape[0])
        #Data to be fitted is weighted with the nonlinear binning Values. As the nonlinear binning returns the average of the binned values. However, here we want weigh all data values based on the signal to noise ratio. Hence we need to multiply each row with the number of bins it is representing. Since we used nonlinear binning, this will change the weight (intensity of each row)
    temp, BinningArrayTime2D = np.meshgrid([0]*Merged.shape[1],BinningArrayTime)
    DataToFit = Merged*BinningArrayTime2D
        #The guess for the initial parameters
        #def VoigtReduced2d(x, center, gamma, offset, *args):
    guess = [0, 0.1, 0, *np.amax(DataToFit,1), *([0.3]*DataToFit.shape[0])]
        #Fit of every row (<-> 2d fit)
    popt, success = curve_fit(VoigtReduced2dNonlinearBinning, X, DataToFit.ravel(), guess, bounds = (lowBounds,highBounds))


    #Plot a subset of the rows and their fits
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    fineSpatialValues = np.linspace(spatialValues[0],spatialValues[-1],100)
    fineX, temp = np.meshgrid(fineSpatialValues, [0]*DataToFit.shape[0])

    magma_no_yellow = cm.get_cmap('magma')
    colors = magma_no_yellow(np.linspace(0.75, 0.2, 3))  # 10 colori diversi

    from matplotlib.colors import LinearSegmentedColormap, to_rgb

    if cmap is None:
        cmap = LinearSegmentedColormap.from_list('#1f77b4',['#1f77b4',"#6baed6"])

    
    #cmap = LinearSegmentedColormap.from_list('#d62728',['#d62728', "#fb6a6a"])
    colors = cmap(np.linspace(0, 1, 3))

    fineSpatialValues = np.linspace(spatialValues[0], spatialValues[-1], 100)
    fineX, temp = np.meshgrid(fineSpatialValues, [0]*DataToFit.shape[0])

    if img:
        plt.figure(figsize = (8,6))

        for i, time_temp in enumerate(time_slice_ns):  
            row = (np.abs(timeValuesMerged - time_temp)).argmin()

            plt.scatter(
                spatialValues - popt[0],
                DataToFit[row] / VoigtReduced2dNonlinearBinning(fineX, *popt)
                    .reshape((DataToFit.shape[0], len(fineSpatialValues)))[row]
                    .max(),
                facecolors='None',
                edgecolors=colors[i],
                linewidths=2,
                label=f"{np.round(timeValuesMerged[row],1)} ns"
            )

            plt.plot(
                fineSpatialValues - popt[0],
                VoigtReduced2dNonlinearBinning(fineX, *popt)
                    .reshape((DataToFit.shape[0], len(fineSpatialValues)))[row] /
                VoigtReduced2dNonlinearBinning(fineX, *popt)
                    .reshape((DataToFit.shape[0], len(fineSpatialValues)))[row]
                    .max(),
                color=colors[i],
                linewidth=2
            )

            plt.tick_params(labelsize=16)
            plt.legend(fontsize=12)
            plt.xlabel('Position ($\mu$m)', fontsize=18)
            plt.ylabel('Normalized Intensity (a.u.)', fontsize=18)
            if saveFig == True:
                plt.savefig(folder+str(arraySelection[:])+'gauss.pdf', dpi=600)
    
    return popt.copy(), success.copy()

def rawdata_to_readydata(rawData, TimeResolution, laser_div, NPulsesSeen, MaxRepRate, binning_time_function, binning_time_params):
    '''
    From raw data we perform some operations to use the data in 2d maps...
    For each operation check their respective functions

    IN:
    rawData = raw data
    some parameters needed...
 
    OUT:
    dat_OnePeak_All = unbinned ordered and ready data
    dataBinned = binned data
    dataBinnedNorm = normalized binned data
    '''
    rawData_2 = []
    for matrix in rawData:
        matrix = matrix[1:]
        rawData_2.append(matrix)
        DataArr = np.array(rawData_2)

    dat_OnePeak_All = []
    for data in DataArr:
        dat_OnePeak_unordered, time_merged_v = merge_Pulses(data, TimeResolution, laser_div, NPulsesSeen, MaxRepRate)
        max_pos = np.argmax(sum(dat_OnePeak_unordered))
        dat_OnePeak_ordered = np.roll(dat_OnePeak_unordered, -max_pos, axis=1)
        dat_OnePeak_All.append(dat_OnePeak_ordered)

    #dataBinned = [] # BINNING IN TIME 
    dataBinned = [] # BINNING IN SPACE
    for data in dat_OnePeak_All:
    #    dataBinned.append(Constant(data.transpose(), binning_time_function, binning_time_params, keep_last_pixels=True))
        dataBinned.append(binningNonlinear(data.transpose(), binning_time_function, binning_time_params, keep_last_pixels=True))

    dataBinnedNorm = [] # Normalize the data
    for data in dataBinned:
        dataBinnedNorm.append(np.array(Normalize(data)))
    
    return dat_OnePeak_All, dataBinned, dataBinnedNorm

def extract_MSD(MergedNormalized, timeValuesMerged, popt, success, cutOff, time_fit, binningSigmasFunction, binningSigmas, t1_index = 0):
    '''
    Given a set of data and the fitting parameters, 
    extract the MSD for each time slice with errors

    IN:
    MergedNormalized = merged and normalized data
    timeValuesMerged = time values
    popt = fitting parameters
    success = success of the fitting
    cutOff = cut off for the fitting parameters
    time_fit = time for the fit
    binningSigmasFunction = function for the binning of the sigmas
    binningSigmas = parameters for the binning of the sigmas
    t1_index = index for the first time slice

    OUT:
    timeValues = time values
    tend_index = index for the last time slice
    deltaSigmaSquare = MSD
    errorsquaresum = errors for the MSD
    '''

    sigmas, sigmasBinningArray = binningNonlinear(popt[-MergedNormalized.shape[0]+cutOff:],binningSigmasFunction,binningSigmas,returnBinningArray = True)
    sigmas = np.array(sigmas)
    timeValues = np.array(binningNonlinear(timeValuesMerged[cutOff:],binningSigmasFunction,binningSigmas))
    timeValues = timeValues - min(timeValues)
    tend_index =(np.abs(timeValues -time_fit)).argmin()


    deltaSigmaSquare = sigmas**2-sigmas[0]**2
    #Calculating errors of sigma (covariance is negligible. I checked and it did not make a difference)
    error = np.sqrt(np.diag(success)[-MergedNormalized.shape[0]+cutOff:])
    error = np.array(binningNonlinearErrors(error, binningSigmasFunction, binningSigmas))
    errorsquare = abs(2*sigmas*error) 
    errorsquare0 = errorsquare[t1_index]
    errorsquaresum = np.array(np.sqrt(errorsquare**2 + errorsquare0**2))
    errorsquaresum[t1_index] = errorsquare0 #Value gets substracted by itself. Hence, the error stays the same

    return timeValues, tend_index, deltaSigmaSquare, errorsquare, errorsquaresum, sigmas


def extract_diffusivity(timeValues, deltaSigmaSquare, errorsquare, t1_index, tend_index):

    '''
    Get D from the MSD data.

    IN:
    timeValues = time values
    deltaSigmaSquare = MSD
    errorsquare = errors for the MSD
    t1_index = index for the first time slice
    tend_index = index for the last time slice

    OUT:
    pDiffusion = fitting parameters
    successDiffusion = success of the fitting
    errorDiffusion = errors for the fitting parameters
    '''

    #Definition diffusion 
    def diffusion(t, D, alpha):
        t = np.array(t)
        return 2*D*(t-timeValues[t1_index])**1 + deltaSigmaSquare[t1_index]

    #Do the fit given the data
    guessDiffusion = [0.001,1]
    lowBounds = [-1]+[0]
    highBounds = [10000000]+[2+1e-10]
    pDiffusion, successDiffusion = curve_fit(diffusion,timeValues[t1_index:tend_index],deltaSigmaSquare[t1_index:tend_index], guessDiffusion, errorsquare[t1_index:tend_index], bounds = (lowBounds,highBounds))
    errorDiffusion = np.sqrt(np.diag(successDiffusion)) 

    return pDiffusion, successDiffusion, errorDiffusion


def plot_MSD_and_Diff(timeValues, deltaSigmaSquare, errorsquaresum,
                      pDiffusion, errorDiffusion, arraySelection,
                      cutOff, t1_index, tend_index, time_fit,
                      saveFig=False, folder=None, figsize=(8,6),
                      xlim=None, ylim=None):


    fig, ax = plt.subplots(1, 1, figsize=figsize)

    blue_fit  = '#1f77b4'         
    blue_out  =  '#1f77b4'    
    red_fit   = '#d62728'              
    red_out   = '#d62728'        

    if isinstance(arraySelection, int) or len(arraySelection) == 1:
        arraySelection = list(arraySelection)
        timeValues = [timeValues]
        deltaSigmaSquare = [deltaSigmaSquare]
        errorsquaresum = [errorsquaresum]
        pDiffusion = [pDiffusion]
        errorDiffusion = [errorDiffusion]
        cutOff = [cutOff]
        t1_index = [t1_index]
        tend_index = [tend_index]
        time_fit = [time_fit]

    for i, arr in enumerate(arraySelection):

        t  = timeValues[i]
        msd = deltaSigmaSquare[i]
        err = errorsquaresum[i]
        p   = pDiffusion[i]
        ep  = errorDiffusion[i]
        t1  = t1_index[i]
        te  = tend_index[i]

        def diffusion(tval, D, alpha):
            tval = np.array(tval)
            return 2 * D * (tval - t[t1]) + msd[t1]

        if i == 0: 
            color_fit = blue_fit
            color_out = blue_out
            legend_label = "$c$-axis"
            capsize_val = 3
        else:       
            color_fit = red_fit
            color_out = red_out
            legend_label = "$a$-axis"
            capsize_val = 3

        ax.errorbar(
            t, msd, yerr=err,
            fmt='o', linestyle='None',
            markerfacecolor='None',
            markeredgecolor=color_out,
            ecolor=color_out,
            alpha=0.6,capsize=capsize_val,
        )

        ax.errorbar(
            t[t1:te], msd[t1:te], yerr=err[t1:te],
            fmt='o',
            markerfacecolor=color_fit,
            markeredgecolor=color_fit,
            ecolor=color_fit,
            linestyle='None',
            alpha=1.0,
            capsize=capsize_val,
            label=legend_label
        )

        t_fit_array = t[t1:te + te//5]
        ax.plot(t_fit_array, diffusion(t_fit_array, *p),
                linestyle='dashed',
                color=color_fit,
                linewidth=3)

        ax.text(
            0.05, 0.85 - 0.12*i,
            f"D = {p[0]*1e1:.3f} ± {ep[0]*1e1:.3f}  (cm²/s)",
            transform=ax.transAxes,
            fontsize=16,
            color=color_fit,
            verticalalignment='top'
        )

    ax.set_xlabel('Time (ns)', fontsize=18)
    ax.set_ylabel('MSD(t) = $\\sigma(t)^2 - \\sigma(0)^2$  (µm²)', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    ax.tick_params(axis='both', which='major', direction='in') 
    ax.tick_params(axis='both', which='minor', direction='in')

    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)

    plt.tight_layout()

    if saveFig == True:
        plt.savefig(folder+'diff_time_fit_'+str(time_fit)+'.pdf', dpi=600)
    

def analysis_parameters(folder = None):

    picoharp330 = False

    # Constant bins
    binning_time_function =  ConstantBins
    binning_time_params = [25] # 25 = 10 bins per ns (4ps * n = x bin/ns), assuming 4 ps resolution

    # No need to change anything below this line

    path = glob.glob(folder + '2*.csv')
    path.sort()
    metadata_path = glob.glob(folder + '2*meas_params.txt')

    cutData_time = [0, 5] #ns
    import os
    # Print filenames being analyzed
    for i, fname in enumerate(path):
        print(f"{i}: {os.path.basename(fname)}")

    TimeResolution, AcquisitionTime, Binning, laser_pw, laser_div, NPulsesSeen, xmin, xmax, ymin, ymax, SpatialResolution, MaxRepRate = extract_parameters_from_metadata(metadata_path, folder)

    if picoharp330:
        NPulsesSeen = int(1)

    print('Parameters extracted: analysing measurements...')
    rawData = LoadData(path)
    dat_OnePeak_All, dataBinned, dataBinnedNorm = rawdata_to_readydata(rawData, TimeResolution, laser_div, NPulsesSeen, MaxRepRate, binning_time_function, binning_time_params)
    spatialValues, timeValuesUncut, BinningArrayTimeUncut = get_spatial_time_values(dat_OnePeak_All, TimeResolution, SpatialResolution, binning_time_function, binning_time_params)

    return path, dataBinned, spatialValues, timeValuesUncut, BinningArrayTimeUncut, cutData_time, SpatialResolution

def max_LF_index(Merged, timeValuesMerged, binningSigmasFunction, binningSigmas):

    '''
    Find the index for the maximum of the lifetime trace
    This script support binning...
    '''

    # get binned time values
    timeValues = np.array(binningNonlinear(timeValuesMerged,binningSigmasFunction,binningSigmas))
    timeValues = timeValues - min(timeValues)

    # get the lifetime trace
    DataSummed = np.sum(Merged, axis=1)

    # find the index for the maximum
    max_index = np.argmax(DataSummed)
    t_max = timeValues[max_index]

    # find the index in the binned time values that corresponds to the maximum
    max_index_binned = (np.abs(timeValues - t_max)).argmin()

    return max_index_binned

def get_weights(sigmas, errorsquare):

    # get the weights for the fitting
    weights = []
    for i in range(len(sigmas)):
        sigma2 = sigmas[i]**2
        stderr = errorsquare[i]
        weights.append([np.power(stderr / sigma2, -2.) if (sigma2 !=0 and stderr != 0) else 0])
    weights = np.array(weights)
    # normalize so the sum of the weights is unity
    weights = [weight/np.sum(weights) for weight in weights]

    return weights

def extract_diffusivity_weights(timeValues, deltaSigmaSquare, errorsquare, t1_index, tend_index, weights):

    '''
    Get D from the MSD data.

    IN:
    timeValues = time values
    deltaSigmaSquare = MSD
    errorsquare = errors for the MSD
    t1_index = index for the first time slice
    tend_index = index for the last time slice

    OUT:
    pDiffusion = fitting parameters
    successDiffusion = success of the fitting
    errorDiffusion = errors for the fitting parameters
    '''

    #Definition diffusion 
    def diffusion(t, D, alpha):
        t = np.array(t)
        return 2*D*(t-timeValues[t1_index])**1 + deltaSigmaSquare[t1_index]

    #Do the fit given the data
    guessDiffusion = [0.001,1]
    lowBounds = [-1]+[0]
    highBounds = [10000000]+[2+1e-10]

    mod_wls = sm.WLS(deltaSigmaSquare[t1_index:tend_index], timeValues[t1_index:tend_index], weights = weights[t1_index:tend_index])
    pDiffusion = mod_wls.fit().params
    errorDiffusion = mod_wls.fit(guessDiffusion, bounds = (lowBounds,highBounds)).bse

    return pDiffusion, errorDiffusion


def get_params_df(file):

    '''
    Open the .csv file in the style of a dataframe and load the parameters from it
    '''

    df = pd.read_csv(file)
    params = df.iloc[0]
    TimeResolution = float(params['TimeResolution_ns'])
    AcquisitionTime = float(params['AcquisitionTime_ms'])
    Binning = int(params['Binning'])
    laser_pw = float(params['LaserPower_uW'])
    laser_div = int(params['LaserDivider'])
    xmin = float(params['X_range_mm'].split(',')[0][1:])
    xmax = float(params['X_range_mm'].split(',')[1][:-1])
    ymin = float(params['Y_range_mm'].split(',')[0][1:])
    ymax = float(params['Y_range_mm'].split(',')[1][:-1])
    steps = int(params['Steps'])
    objective = float(params['Objective'])

    if laser_div ==1:
        NPulsesSeen = int(7)
    elif laser_div == 2:
        NPulsesSeen = int(4)
    elif laser_div == 4:
        NPulsesSeen = int(2)
    else:
        NPulsesSeen = int(1)

    '''
    EXTRACTING THE APD RANGE FROM THE INITIAL AND LAST POSITION
    CALCULATING THE SPATIAL RESOLUTION AS:

    (APD_range/total_magnification)/steps

    '''     
    APD_range = np.linalg.norm(np.array([xmin,ymin]) - np.array([xmax,ymax])) 
#    APD_range = np.abs(xmax-xmin) 
    lens_magnification = 3.636363633333333   
    total_magnification = lens_magnification * objective
    sample_range = APD_range / total_magnification
    SpatialResolution = sample_range / (steps-1) # mm

    return TimeResolution, AcquisitionTime, Binning, laser_pw, laser_div, NPulsesSeen, xmin, xmax, ymin, ymax, SpatialResolution

def param_from_metadata(metadata_json):

    # given a metadata json file, extract the parameters

    TimeResolution = metadata_json['TimeResolution']
    AcquisitionTime = metadata_json['AcquisitionTime']
    Binning = metadata_json['Binning']
    laser_pw = metadata_json['laserPower']
    laser_div = metadata_json['laserDivider']
    objective = metadata_json['objective']
    steps = metadata_json['steps']
    MaxRepRate = metadata_json['MaxRepRate']*1e6

    # find the number of pulses seen
    # first check if the picoharp is 300 or 330. It could be an old file without this key

    try :
        picoharp_300 = metadata_json['picoharp_300']

    except KeyError:
        picoharp_300 = True # if it is not there, then assume it is the 300

    if picoharp_300:
        if MaxRepRate == 80e6:
            if laser_div < 8:
                NPulsesSeen = int(8/ laser_div)
            else:
                NPulsesSeen = 1
        else:
            NPulsesSeen = 1

        lens_magnification = 3.636363633333333

    else:
        NPulsesSeen = 1
        lens_magnification = 3.68686869 #20251124 


    xmin = metadata_json['x_range'][0]
    xmax = metadata_json['x_range'][1]

    ymin = metadata_json['y_range'][0]
    ymax = metadata_json['y_range'][1]

    APD_range = np.linalg.norm(np.array([xmin,ymin]) - np.array([xmax,ymax])) 
    #APD_range = np.abs(xmax-xmin) 
    #lens_magnification = 3.636363633333333   
    total_magnification = lens_magnification * objective
    sample_range = APD_range / total_magnification
    SpatialResolution = sample_range / steps # mm

    return TimeResolution, AcquisitionTime, Binning, laser_pw, laser_div, NPulsesSeen, xmin, xmax, ymin, ymax, SpatialResolution, MaxRepRate


def extract_parameters_from_metadata(metadata_path, folder):
    if len(metadata_path) == 0:
        # then it is json, new measurements
        metadata_path = glob.glob(folder + '2*metadata.json')

        # get the dictionary

        with open(metadata_path[0], 'r') as f:

            metadata = json.load(f)

            TimeResolution, AcquisitionTime, Binning, laser_pw, laser_div, NPulsesSeen, xmin, xmax, ymin, ymax, SpatialResolution, MaxRepRate = param_from_metadata(metadata)

    else:
        # then it is the old one
        path_param = metadata_path
        TimeResolution, AcquisitionTime, Binning, laser_pw, laser_div, NPulsesSeen, xmin, xmax, ymin, ymax, SpatialResolution = extract_parameters(path_param)
        MaxRepRate = 80e6
        print('Assuming MaxRepRate = 80 MHz')

    return TimeResolution, AcquisitionTime, Binning, laser_pw, laser_div, NPulsesSeen, xmin, xmax, ymin, ymax, SpatialResolution, MaxRepRate

def get_spatial_time_values(dat_OnePeak_All, TimeResolution, spatialResolution, binning_time_function, binning_time_params):
    TimeResolution = TimeResolution *1e-3 # ns
    spatialValues = np.arange(dat_OnePeak_All[0].shape[0])*spatialResolution # mm
    spatialValues = spatialValues - (spatialValues[-1] - spatialValues[0])/2
    timeValuesUncut, BinningArrayTimeUncut = binningNonlinear(np.arange(dat_OnePeak_All[0].shape[1])*TimeResolution, binning_time_function, binning_time_params, keep_last_pixels = True, returnBinningArray = True)
    timeValuesUncut = timeValuesUncut - min(timeValuesUncut)

    return spatialValues, timeValuesUncut, BinningArrayTimeUncut

#def merge_new_arrays(data, selection)

def merge_selected_arrays(dataBinned, arraySelection,  cutData_time, timeValuesUncut, BinningArrayTimeUncut, spatialResolution):

     
    #Find index closest to indicated times
    cutData = [(np.abs(timeValuesUncut - t)).argmin() for t in cutData_time]

    min_shape = 200 # first iteration
    dat_to_merge = []
    for i in arraySelection:

        uncentered_dat = np.array(dataBinned[i][cutData[0]:cutData[1]])
        # center the data, find the max value at time 0 and use it as the center (make the array symmetryc )

        averaged_first = np.mean(uncentered_dat[0:5,:], axis=0)
        max_index = np.argmax(averaged_first)

        # roll the axis 1 so the max index is at the center of the array

        remove_value = max_index*2 - uncentered_dat.shape[1] # centered data should end up with shape (max_index*2, uncentered_dat.shape[0])

        if remove_value > 0:
            centered_dat = np.delete(uncentered_dat, range(np.abs(remove_value)), axis=1)

        elif remove_value < 0:
            # flip it and do the same
            #uncentered_dat = np.flip(uncentered_dat, axis=1)
            centered_dat = np.delete(uncentered_dat, range(np.abs(remove_value)) + max_index*2, axis=1)
        else:
            centered_dat = uncentered_dat

        min_shape = np.min([centered_dat.shape[1], min_shape])

        dat_to_merge.append(centered_dat)

    # ok, each array inside dat_to_merge has a different shape, we need to make them the same size. Use min_shape as the target size

    dat_ready_merge = []
    for i in range(len(dat_to_merge)):

        # check the shape of the array
        current_shape = dat_to_merge[i].shape[1]
        # centered data is set to shape min_index*2 so it will always be even

        if current_shape > min_shape:
            cut_col = (current_shape - min_shape)//2
            dat_ready_merge.append( dat_to_merge[i][:, cut_col:-cut_col] )

        dat_ready_merge.append( dat_to_merge[i][:, :min_shape] )

    Merged = np.array(dat_ready_merge).mean(axis=0)

    MergedNormalized = []
    for row in Merged:
        # normalize each row in Merged to the max value of the row
        MergedNormalized.append(row/np.max(row))
    MergedNormalized = np.array(MergedNormalized)

    spatialValues = (np.arange(Merged.shape[1]) - np.argmax(sum(Merged)))*spatialResolution
    timeValuesMerged = timeValuesUncut[cutData[0]:cutData[1]]
    timeValuesMerged = timeValuesMerged - min(timeValuesMerged)
    BinningArrayTime = BinningArrayTimeUncut[cutData[0]:cutData[1]]


    return Merged, MergedNormalized, spatialValues, timeValuesMerged, BinningArrayTime



def plot_diff_map_profiles_MSD(dataBinned, spatialValues, timeValuesUncut, BinningArrayTimeUncut, cutData_time, SpatialResolution, path, folder, saveFig):

    datasets = [] 
    binningSigmas = [0,1,1] # no extra binning for the sigma. [0,,1,1] or [0,1.25,1] typical values for the binning of the sigma
    binningSigmasFunction = PowerLawForBins
    cutOff = 0 # if the first data points shouldnt be plotted and fitted
    t1_index = 0 # Begin of fitting, index #t1_index = max_LF_index(Merged, timeValuesMerged, binningSigmasFunction, binningSigmas) # Get the MSD for each time slice...
    time_fit = 1.5 # end of fi, ns
                        
    # Dont change anything below this line

    Merged, MergedNormalized, spatialValues, timeValuesMerged, BinningArrayTime = merge_selected_arrays(dataBinned, [0] , cutData_time, timeValuesUncut, BinningArrayTimeUncut, SpatialResolution)

    Merged_0, MergedNormalized_0, spatialValues_0, timeMerged_0, BinningTime_0 = merge_selected_arrays(dataBinned, [0], cutData_time, timeValuesUncut, BinningArrayTimeUncut, SpatialResolution)

    Merged_1, MergedNormalized_1, spatialValues_1, timeMerged_1, BinningTime_1 = merge_selected_arrays(dataBinned, [1], cutData_time, timeValuesUncut, BinningArrayTimeUncut, SpatialResolution)
    popt0, success0 = plot_merge_diffusion_map(Merged_0, spatialValues_0*1e3, timeValuesMerged, BinningArrayTime, [0], SpatialResolution, cut_time= cutData_time[1]-1,time_slice_ns=[0,1.0,2.4],cmap = LinearSegmentedColormap.from_list('#1f77b4',['#1f77b4',"#6baed6"]),saveFig = saveFig, folder = folder)
    t0, tend0, msd0, err0_sq, err0, sigmas0 = extract_MSD(
        MergedNormalized_0, timeMerged_0, popt0, success0, cutOff, time_fit,
        binningSigmasFunction, binningSigmas, t1_index
    )
    popt1, success1 = plot_merge_diffusion_map(Merged_1, spatialValues_1*1e3, timeValuesMerged, BinningArrayTime, [0], SpatialResolution, cut_time= cutData_time[1]-1,time_slice_ns=[0,0.6,1.2],cmap = LinearSegmentedColormap.from_list('#d62728',['#d62728',"#fb6a6a"]),saveFig = saveFig, folder = folder)

    t1, tend1, msd1, err1_sq, err1, sigmas1 = extract_MSD(
        MergedNormalized_1, timeMerged_1, popt1, success1, cutOff, time_fit,
        binningSigmasFunction, binningSigmas, t1_index
    )
    p0, succD0, ep0 = extract_diffusivity(t0, msd0, err0_sq, t1_index, tend0)
    p1, succD1, ep1 = extract_diffusivity(t1, msd1, err1_sq, t1_index, tend1)

    dataset0 = {
        "array": 0,
        "time": t0,
        "msd": msd0,
        "error": err0,
        "p": p0,
        "err_p": ep0,
        "t1_index": t1_index,
        "tend_index": tend0
    }

    dataset1 = {
        "array": 1,
        "time": t1,
        "msd": msd1,
        "error": err1,
        "p": p1,
        "err_p": ep1,
        "t1_index": t1_index,
        "tend_index": tend1
    }

    datasets.append(dataset0)
    datasets.append(dataset1)

    timeValues_list = [dataset0["time"], dataset1["time"]]
    deltaSigmaSquare_list = [dataset0["msd"], dataset1["msd"]]
    errorsquaresum_list = [dataset0["error"], dataset1["error"]]
    pDiffusion_list = [dataset0["p"], dataset1["p"]]
    errorDiffusion_list = [dataset0["err_p"], dataset1["err_p"]]
    t1_index_list = [dataset0["t1_index"], dataset1["t1_index"]]
    tend_index_list = [dataset0["tend_index"], dataset1["tend_index"]]
    cutoff_list = [cutOff, cutOff]
    time_fit_list = [time_fit, time_fit]
    saveFig = True
    arraySelection = [0, 1]

    plot_MSD_and_Diff(
        timeValues_list,
        deltaSigmaSquare_list,
        errorsquaresum_list,
        pDiffusion_list,
        errorDiffusion_list,
        arraySelection,
        cutoff_list,
        t1_index_list,
        tend_index_list,
        time_fit_list,
        saveFig=saveFig,
        folder = folder,
        figsize=(8,6),
        xlim=(-0.0001,2),
        ylim=(-0.001,0.05)
    )

    return Merged_0, Merged_1, timeValuesMerged


def plot_lifetime_trace(timeValuesMerged,Merged_0,Merged_1,path,folder,saveFig):

    colors = ['#d62728', '#1f77b4']

    tau_eff_all = []
    sigma_tau_eff_all = []

    for i, file in enumerate(path):

        timeValuesMerged = timeValuesMerged
        if i == 0:
            Merged = Merged_0   
            DataSummed, lifetime_fit, lifetime_pcov = plot_merge_lifetimes_fit(
            Merged_0,
            timeValuesMerged,timeSlice=[1, 0],
            scatter_color=colors[i],
            fit_color='darkred',
            saveFig=saveFig,
            folder=folder
        )
        else:
            Merged = Merged_1
            DataSummed, lifetime_fit, lifetime_pcov = plot_merge_lifetimes_fit(
            Merged_1,
            timeValuesMerged,
            timeSlice=[1, 0],scatter_color=colors[i],
            fit_color='darkred',
            saveFig=saveFig,
            folder=folder
        )
                
        a1 = lifetime_fit[0][0]
        tau1 = lifetime_fit[0][1]
        a2 = lifetime_fit[0][2]
        tau2 = lifetime_fit[0][3]

        den = (a1 + a2)**2
        grad = np.array([
            (a2*(tau1 - tau2))/den,
            a2/(a1 + a2),
            (a1*(tau2 - tau1))/den,
            a1/(a1 + a2)
        ])

        var_tau_eff = grad @ lifetime_pcov[0] @ grad.T
        tau_eff = ((a1*tau1) + (a2*tau2)) / (a1 + a2)
        sigma_tau_eff = np.sqrt(var_tau_eff)

        tau_eff_all.append(tau_eff)
        sigma_tau_eff_all.append(sigma_tau_eff)

        print(f'{file}')
        print(f'tau_eff = {tau_eff:.2f} ± {sigma_tau_eff:.2f} ns\n')


    plt.xlabel('Time (ns)', fontsize=16)
    plt.ylabel('Intensity (a.u.)', fontsize=16)
    plt.tight_layout()
