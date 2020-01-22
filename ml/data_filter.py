import pandas as pd
import csv
import matplotlib.pyplot as plt

squat = pd.read_csv("Emily_squat_11_4.csv")

s = squat['x-axis (g)']
s.plot()
plt.show()

s = s[s.diff() > 0.04]
s = s.add(-s.mean())

s = s.rolling(window=2).mean()

s.plot()
plt.show()

count = 0
prev = -999
for i,v in s.iteritems():
    if v > 0.5*(s.max()):
        if i - prev > 20:
            prev = i
            count+=1
    
print(count)