import pandas as pd
import csv
import matplotlib.pyplot as plt
import statistics
from scipy.signal import find_peaks, peak_prominences 
from scipy.signal import detrend
import numpy as np

squat = pd.read_csv("data/sabrina_sq_155_2_4_5reps.csv") # load in some csv data

x = squat['x-axis (g)'].astype(float)
y = squat['y-axis (g)'].astype(float)
z = squat['z-axis (g)'].astype(float)

s = x.pow(2) + y.pow(2) + z.pow(2)
s = np.sqrt(s)

s.plot() # plot original data
plt.show()

s = s[s.diff() > 0.004]  # Try 0.003 to 0.01

s[:-2] = s[:-2].rolling(window=3).mean()  # Try windows 1, 2, 3
s = s.dropna()

s = s.add(-s.mean())    # Shift data to center the mean at 0

s = detrend(s) # Remove trend lines 
s = s*(1/s.max())   # Scale data

peaks = find_peaks(s, distance=10)[0]   # Try distances: 10, 15

prominences = peak_prominences(s, peaks)[0]
heights = s[peaks] - prominences
plt.plot(s)
plt.plot(peaks, s[peaks], "s")
plt.vlines(x=peaks, ymin=heights, ymax=s[peaks])
plt.show()

prominences = prominences[prominences>0.5*max(prominences)] # Try 0.4 to 0.5
print(len(prominences))

