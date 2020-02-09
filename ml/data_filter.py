import pandas as pd
import csv
import matplotlib.pyplot as plt
import statistics
from scipy.signal import find_peaks, peak_prominences 
from scipy.signal import detrend
import numpy as np

squat = pd.read_csv("data/Sabrina_DL_205_5reps_bad.csv") # load in some csv data

s = squat['x-axis (g)']
s.plot() # plot original data
plt.show()

s.drop(s.tail(30).index, inplace=True) # Try 30, 40

s = s[s.diff() > 0.005]  # Ignore points where change is less than some amount from previous

s = s.rolling(window=2).mean()  # Try 1, 2, 3
s = s.dropna()

s = s.add(-s.mean())    # Shift data to center the mean at 0

s = detrend(s) # remove trend lines 
s = s*(1/s.max())   # Scale data

peak1 = find_peaks(s, height=0.5*s.max(), distance=12)[0]   # METHOD 1
peaks = find_peaks(s, distance=11)[0]   # Try distances: 10, 20
print(len(peak1))

prominences = peak_prominences(s, peaks)[0]
heights = s[peaks] - prominences
plt.plot(s)
plt.plot(peaks, s[peaks], "s")
plt.vlines(x=peaks, ymin=heights, ymax=s[peaks])
plt.show()

prominences = prominences[prominences>0.3*max(prominences)] # Try 0.25, 0.3, 0.35
print(len(prominences)) # METHOD 2

