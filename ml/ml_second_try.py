from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import coremltools
from scipy import stats
from IPython.display import display, HTML

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils


# Set some standard parameters upfront
pd.options.display.float_format = '{:.1f}'.format
sns.set() # Default seaborn look and feel
plt.style.use('ggplot')
# Same labels will be reused throughout the program
LABELS = ['Deadlift',
          'Squat']
# The number of steps within one time segment
TIME_PERIODS = 40
# The steps to take from one segment to the next; if this value is equal to
# TIME_PERIODS, then there is no overlap between the segments
STEP_DISTANCE = 20


def read_data(file_path):
    df = pd.read_csv(file_path)
    # ... and then this column must be transformed to float explicitly
    df['x-axis (g)'] = df['x-axis (g)'].astype(float)
    df['y-axis (g)'] = df['y-axis (g)'].astype(float)
    df['z-axis (g)'] = df['z-axis (g)'].astype(float)
    df.drop('epoc (ms)', axis=1, inplace=True)
    df.drop('timestamp (-04:00)', axis=1, inplace=True)
    df.drop('elapsed (s)', axis=1, inplace=True)
    
    show_basic_dataframe_info(df)

    return df

 
def show_basic_dataframe_info(dataframe):

    # Shape and how many rows and columns
    print('Number of columns in the dataframe: %i' % (dataframe.shape[1]))
    print('Number of rows in the dataframe: %i\n' % (dataframe.shape[0]))

# Load data set containing all the data from csv
df = read_data('data/Sabrina_DL_200.csv')