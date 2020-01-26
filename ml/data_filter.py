import pandas as pd
import csv
import matplotlib.pyplot as plt
import statistics
from scipy.signal import find_peaks

squat = pd.read_csv("data/Emily_SQ_160.csv") # load in some csv data

s = squat['x-axis (g)'] # look only at x data
s.plot() # plot original data
plt.show()

print(statistics.stdev(s)) # print std deviation    NOTE: Large standard deviation (0.36 instead of 0.1 or 0.05) indicates skewed data, downward trend

s.drop(s.tail(50).index, inplace=True) 
s = s[s.diff() > 0.02]  # Ignore points where change is less than some amount from previous
s = s.add(-s.mean())    # Shift data to center the mean at 0

s = s.rolling(window=2).mean()  # Do rolling average to clean up data

s = s*(1/s.max())   # Scale data 

s.plot()
plt.show()

count = 0 # Count up number of peaks (> 0.5*max)
prev = -999
for i,v in s.iteritems():
    if v > 0.5*(s.max()):
        if i - prev > 20:   # ignore if right next to another peak
            prev = i
            count+=1
    
print(count)

peaks = find_peaks(s, distance=10)[0]
print(len(peaks))