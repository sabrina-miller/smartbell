from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import coremltools
import os, sys
import csv
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
    for index,row in enumerate(csv.reader(open('data/'+file_path, 'r'))):
        file_path = file_path.lower()
        if index != 0:
            if "sabrina" in file_path and "sq" in file_path:
                writer.writerow(row+[file_num]+["Sabrina"]+["squat"])
            elif "sabrina" in file_path and "dl" in file_path:
                writer.writerow(row+[file_num]+["Sabrina"]+["deadlift"])
            elif "emily" in file_path and "sq" in file_path:
                writer.writerow(row+[file_num]+["Emily"]+["squat"])
            elif "emily" in file_path and "dl" in file_path:
                writer.writerow(row+[file_num]+["Emily"]+["deadlift"])
            elif "sophie" in file_path and "sq" in file_path:
                writer.writerow(row+[file_num]+["Sophie"]+["squat"])
            elif "sophie" in file_path and "dl" in file_path:
                writer.writerow(row+[file_num]+["Sophie"]+["deadlift"])
        elif file_num == 1:
            writer.writerow(row+["file_num"]+["person"]+["activity"])

    
def machine_learn():

    df = pd.read_csv('all_data.csv')
    df['x-axis (g)'] = df['x-axis (g)'].astype(float)
    df['y-axis (g)'] = df['y-axis (g)'].astype(float)
    df['z-axis (g)'] = df['z-axis (g)'].astype(float)
    #df.drop('y-axis (g)', axis=1, inplace=True)
    #df.drop('z-axis (g)', axis=1, inplace=True)
    df.drop('epoc (ms)', axis=1, inplace=True)
    # df.drop('timestamp (-04:00)', axis=1, inplace=True)
    df.drop('elapsed (s)', axis=1, inplace=True)
    
    show_basic_dataframe_info(df)
    
    df['activity'].value_counts().plot(kind='bar', title='Training Examples by Activity Type')
    #plt.show()
    # Better understand how the recordings are spread across the different
    # users who participated in the study
    df['person'].value_counts().plot(kind='bar', title='Training Examples by User')
    #plt.show()

    LABEL = 'ActivityEncoded'
    # Transform the labels from String to Integer via LabelEncoder
    le = preprocessing.LabelEncoder()
    # Add a new column to the existing DataFrame with the encoded values
    df[LABEL] = le.fit_transform(df['activity'].values.ravel())

    # Differentiate between test set and training set
    df_test = df[df['file_num'] > 7]
    df_train = df[df['file_num'] <= 7]
    
    # Normalize features for training data set (values between 0 and 1)
    # Surpress warning for next 3 operation
    pd.options.mode.chained_assignment = None  # default='warn'
    df_train['x-axis (g)'] = df_train['x-axis (g)'] / df_train['x-axis (g)'].max()
    df_train['y-axis (g)'] = df_train['y-axis (g)'] / df_train['y-axis (g)'].max()
    df_train['z-axis (g)'] = df_train['z-axis (g)'] / df_train['z-axis (g)'].max()
    # Round numbers
    df_train = df_train.round({'x-axis (g)': 4, 'y-axis (g)': 4, 'z-axis (g)': 4})
    
    # x, y, z acceleration as features
    N_FEATURES = 3
    # Number of steps to advance in each iteration (for me, it should always
    # be equal to the TIME_PERIODS in order to have no overlap between segments)
    # STEP_DISTANCE = TIME_PERIODS
    segments = []
    labels = []
    for i in range(0, len(df) - TIME_PERIODS, STEP_DISTANCE):
        xs = df['x-axis (g)'].values[i: i + TIME_PERIODS]
        ys = df['y-axis (g)'].values[i: i + TIME_PERIODS]
        zs = df['z-axis (g)'].values[i: i + TIME_PERIODS]
        # Retrieve the most often used label in this segment
        label = stats.mode(df['ActivityEncoded'][i: i + TIME_PERIODS])[0][0]
        segments.append([xs, ys, zs])
        labels.append(label)

    # Bring the segments into a better shape
    x_train = np.asarray(segments, dtype= np.float32).reshape(-1, TIME_PERIODS, N_FEATURES)
    y_train = np.asarray(labels)
    
    print('x_train shape: ', x_train.shape)
    print(x_train.shape[0], 'training samples')
    print('y_train shape: ', y_train.shape)
    
    # Set input & output dimensions
    num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
    num_classes = le.classes_.size
    print(list(le.classes_))
    
    input_shape = (num_time_periods*num_sensors)
    x_train = x_train.reshape(x_train.shape[0], input_shape)
    print('x_train shape:', x_train.shape)
    print('input_shape:', input_shape)
    
    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')
    
    y_train_hot = np_utils.to_categorical(y_train, num_classes)
    print('New y_train shape: ', y_train_hot.shape)
    
    model_m = Sequential()
    # Remark: since coreml cannot accept vector shapes of complex shape like
    # [80,3] this workaround is used in order to reshape the vector internally
    # prior feeding it into the network
    model_m.add(Reshape((TIME_PERIODS, 3), input_shape=(input_shape,)))
    model_m.add(Dense(100, activation='relu'))
    model_m.add(Dense(100, activation='relu'))
    model_m.add(Dense(100, activation='relu'))
    model_m.add(Flatten())
    model_m.add(Dense(num_classes, activation='softmax'))
    print(model_m.summary())
    
    callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='acc', patience=1)]

    model_m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Hyper-parameters
    BATCH_SIZE = 200
    EPOCHS = 50

    # Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
    history = model_m.fit(x_train,
                      y_train_hot,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.2,
                      verbose=1)
                      
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['accuracy'], 'r', label='Accuracy of training data')
    plt.plot(history.history['val_accuracy'], 'b', label='Accuracy of validation data')
    plt.plot(history.history['loss'], 'r--', label='Loss of training data')
    plt.plot(history.history['val_loss'], 'b--', label='Loss of validation data')
    plt.title('Model Accuracy and Loss')
    plt.ylabel('Accuracy and Loss')
    plt.xlabel('Training Epoch')
    plt.ylim(0)
    plt.legend()
    plt.show()

    # Print confusion matrix for training data
    y_pred_train = model_m.predict(x_train)
    # Take the class with the highest probability from the train predictions
    max_y_pred_train = np.argmax(y_pred_train, axis=1)
    print(classification_report(y_train, max_y_pred_train))
    
    
    
def show_basic_dataframe_info(dataframe):

    # Shape and how many rows and columns
    print('Number of columns in the dataframe: %i' % (dataframe.shape[1]))
    print('Number of rows in the dataframe: %i\n' % (dataframe.shape[0]))


file_num = 1
writer = csv.writer(open('all_data.csv', 'w'))
for file in os.listdir("data/"):
    read_data(file)
    file_num +=1

machine_learn()




