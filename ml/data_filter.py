"""

data_filter.py
By Sophie Saunders
Created 12/2019
Tufts University EE-97
Team Maximum Blue Green

This script analyzes one data file and produces a count of how many 
reps of an exercise are performed, by looking at peaks in acceleration. 

Whether the exercise is a deadlift or squat is not relevant to this file. 

Run this file from the command line using:
python data_filter.py 

"""

#import statements
import pandas as pd
import csv
import matplotlib.pyplot as plt
import statistics
from scipy.signal import find_peaks, peak_prominences 
from scipy.signal import detrend
import numpy as np
import argparse

# establish command line argument that gives name of file to examine
parser = argparse.ArgumentParser(description='Calculate how many reps of an exercise from a CSV file.')
parser.add_argument("file_name")
args = parser.parse_args()

# load in the CSV file you want to look at
squat = pd.read_csv("data/"+args.file_name) 

# convert acceleration data to float data type
x = squat['x-axis (g)'].astype(float) 
y = squat['y-axis (g)'].astype(float)
z = squat['z-axis (g)'].astype(float)

# merge all 3 data sets into 1 using distance formula
s = x.pow(2) + y.pow(2) + z.pow(2)
s = np.sqrt(s)

# plot original data
s.plot()
plt.show()

# delete datapoints where there is no change (ie, horizontal lines of no motion)
s = s[s.diff() > 0.004]  # Try 0.003 to 0.01

# perform rolling average to get neater graph
s[:-2] = s[:-2].rolling(window=3).mean()  # Try windows 1, 2, 3
s = s.dropna()

# shift data to center the mean at zero 
s = s.add(-s.mean())

# remove trend lines -- this function is built in for dataframes
s = detrend(s)

# scale data so max acceleration is at 1 
s = s*(1/s.max())   # Scale data

# find peaks in dataframe that are at least 7 data points apart
peaks = find_peaks(s, distance=7)[0]   # Try distances: 5, 10, 15

# plot the results so user can see what (if anything) went wrong 
prominences = peak_prominences(s, peaks)[0]
heights = s[peaks] - prominences
plt.plot(s)
plt.plot(peaks, s[peaks], "s")
plt.vlines(x=peaks, ymin=heights, ymax=s[peaks])
plt.show()

# count all peaks that have height >0.5* max_height
# print this count for the user
prominences = prominences[prominences>0.5*max(prominences)] # Try 0.4 to 0.5
print(len(prominences))

