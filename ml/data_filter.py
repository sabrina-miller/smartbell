import pandas as pd
import csv
import matplotlib.pyplot as plt
import statistics
from scipy.signal import find_peaks
from scipy.signal import detrend
import numpy as np

squat = pd.read_csv("data/Sabrina_DL_90.csv") # load in some csv data

s = squat['x-axis (g)'] # look only at x data
s.plot() # plot original data
plt.show()

s.drop(s.tail(30).index, inplace=True) 

s = s[s.diff() > 0.005]  # Ignore points where change is less than some amount from previous

#s = s.rolling(window=3).mean()  # Do rolling average to clean up data ???
s = s.add(-s.mean())    # Shift data to center the mean at 0

s = detrend(s) # remove trend lines 

plt.plot(s)
plt.show()

count = 0 # Count up number of peaks (> 0.5*max)
prev = -999
for i,v in enumerate(s):
    if v > 0.5*(s.max()):
        if i - prev > 20: # ignore if right next to another peak
            prev = i
            count+=1
    
print(count)

peaks = find_peaks(s, height=0.5*s.max(), distance=10)[0]   # The better way to count peaks??
print(len(peaks))