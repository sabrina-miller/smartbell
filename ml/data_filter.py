import pandas as pd
import csv
import matplotlib.pyplot as plt
import statistics
from scipy.signal import find_peaks, peak_prominences 
from scipy.signal import detrend
import numpy as np

squat = pd.read_csv("data/Emily_DL_135_5reps.csv") # load in some csv data

x = squat['x-axis (g)'].astype(float)
y = squat['y-axis (g)'].astype(float)
z = squat['z-axis (g)'].astype(float)

s = x.pow(2) + y.pow(2) + z.pow(2)
s = np.sqrt(s)

s.plot() # plot original data
plt.show()

s.drop(s.tail(30).index, inplace=True) # Try 30, 40

s = s[s.diff() > 0.005]  # Ignore points where change is less than some amount from previous

s = s.rolling(window=2).mean()  # Try 1, 2, 3
s = s.dropna()

s = s.add(-s.mean())    # Shift data to center the mean at 0

s = detrend(s) # remove trend lines 
s = s*(1/s.max())   # Scale data

peaks = find_peaks(s, distance=11)[0]   # Try distances: 10, 20
print(len(peak1))

prominences = peak_prominences(s, peaks)[0]
heights = s[peaks] - prominences
plt.plot(s)
plt.plot(peaks, s[peaks], "s")
plt.vlines(x=peaks, ymin=heights, ymax=s[peaks])
plt.show()

prominences = prominences[prominences>0.35*max(prominences)] # Try 0.25, 0.3, 0.35
print(len(prominences)) # METHOD 2

