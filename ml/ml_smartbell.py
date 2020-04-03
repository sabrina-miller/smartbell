"""

ml_smartbell.py
By Sophie Saunders
For ECE Senior Design EE-98 at Tufts University
Team Maximum Blue Green
Adapted from tutorial at:
https://towardsdatascience.com/human-activity-recognition-har-tutorial-with-keras-and-core-ml-part-1-8c05e365dfa0

This program loops through all data files (containing x, y, z acceleration)
and reads them into a Python dataframe, then performs neural network classification.

The input to the ML algorithm is 45 acceleration values (doubles) set up as 
[(15 x acceleration values) + (15 y accleration values) + (15 z acceleration values)]
The output is an exercise label of either 'squat' or 'deadlift'

Run this file from command line using:
python ml_smartbell.py

"""

# import statements
from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import csv
from scipy import stats

from sklearn.metrics import classification_report
from sklearn import preprocessing

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape
from keras.utils import np_utils

sns.set() # graph aesthetics

# Note: if STEPS_DISTANCE = TIME_PERIODS then there is no overlap between segments
TIME_PERIODS = 15 # number of steps in one time segment
STEP_DISTANCE = 8 # steps taken from one segment to another

# read_data() function
# takes in CSV data file path and reads it into one compiled file called all_data.csv
# this file keeps track of the file number, who performed the exercise, the x/y/z acceleration data,
# and what activity they performed
def read_data(file_path):
    for index,row in enumerate(csv.reader(open('data/'+file_path, 'r'))):
        file_path = file_path.lower()
        try:
            if (abs(prev - float(row[3])) > 0.005):
                if index != 0:
                    if "sabrina" in file_path and "sq" in file_path:
                        writer.writerow(row+[file_num]+["Sabrina"]+["squat"])
                    elif "sabrina" in file_path and "dl" in file_path:
                        writer.writerow(row+[file_num]+["Sabrina"]+["deadlift"])
                    elif "emilyc" in file_path and "sq" in file_path:
                        writer.writerow(row+[file_num]+["Emily C"]+["squat"])
                    elif "emilyc" in file_path and "dl" in file_path:
                        writer.writerow(row+[file_num]+["Emily C"]+["deadlift"])
                    elif "emily" in file_path and "sq" in file_path:
                        writer.writerow(row+[file_num]+["Emily"]+["squat"])
                    elif "emily" in file_path and "dl" in file_path:
                        writer.writerow(row+[file_num]+["Emily"]+["deadlift"])
                    elif "sophie" in file_path and "sq" in file_path:
                        writer.writerow(row+[file_num]+["Sophie"]+["squat"])
                    elif "sophie" in file_path and "dl" in file_path:
                        writer.writerow(row+[file_num]+["Sophie"]+["deadlift"])
                    elif "katie" in file_path and "sq" in file_path:
                        writer.writerow(row+[file_num]+["Katie"]+["squat"])
                    elif "katie" in file_path and "dl" in file_path:
                        writer.writerow(row+[file_num]+["Katie"]+["deadlift"])
                    elif "david" in file_path and "sq" in file_path:
                        writer.writerow(row+[file_num]+["David"]+["squat"])
                    elif "david" in file_path and "dl" in file_path:
                        writer.writerow(row+[file_num]+["David"]+["deadlift"])
            prev = float(row[3])
        except:
            if index ==0 and file_num == 1:
                writer.writerow(row+["file_num"]+["person"]+["activity"])
            prev = 0

    
# machine_learn() function
# creates .h5 machine learning model from all_data.csv 
def machine_learn():

    df = pd.read_csv('all_data.csv') # create dataframe with all info from all_data.csv 
    df['x-axis (g)'] = df['x-axis (g)'].astype(float) # convert acceleration data to float data type
    df['y-axis (g)'] = df['y-axis (g)'].astype(float)
    df['z-axis (g)'] = df['z-axis (g)'].astype(float)
    df.drop('epoc (ms)', axis=1, inplace=True) # remove timing information from dataframe
    df.drop('elapsed (s)', axis=1, inplace=True)
    
    # Data analysis (can delete later) -- just for informational use
    #df['activity'].value_counts().plot(kind='bar', title='Training Examples by Activity Type')
    #plt.show()
    #df['person'].value_counts().plot(kind='bar', title='Training Examples by User')
    #plt.show()

    le = preprocessing.LabelEncoder() # convert labels from string to integer
    df['ActivityEncoded'] = le.fit_transform(df['activity'].values.ravel()) # add encoded values to dataframe

    # Differentiate between files used for test set and training set
    df_test = df[df['file_num'] <= 12]
    df_train = df[df['file_num'] > 12]
    
    
    N_FEATURES = 3 # set up x, y, z acceleration as features
    
    segments = [] # the input to the ML algorithm
    labels = [] # the output to the ML algorithm
    for i in range(0, len(df_train) - TIME_PERIODS, STEP_DISTANCE):
        xs = df_train['x-axis (g)'].values[i: i + TIME_PERIODS]
        ys = df_train['y-axis (g)'].values[i: i + TIME_PERIODS]
        zs = df_train['z-axis (g)'].values[i: i + TIME_PERIODS]
        # Retrieve the most often used label in this segment
        label = stats.mode(df_train['ActivityEncoded'][i: i + TIME_PERIODS])[0][0]
        segments.append([xs, ys, zs]) # set up input as array of x, y, z accelerations
        labels.append(label) # set up output as label of 'squat' or 'deadlift'

    # Reshape the inputs into arrays of the correct dimension
    x_train = np.asarray(segments, dtype= np.float32).reshape(-1, TIME_PERIODS, N_FEATURES)
    y_train = np.asarray(labels)
    
    # print information about input, output shapes
    print('x_train shape: ', x_train.shape)
    print(x_train.shape[0], 'training samples')
    print('y_train shape: ', y_train.shape)
    
    # Set input and output dimensions
    num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
    num_classes = le.classes_.size # num_classes = 2
    input_shape = (num_time_periods*num_sensors)
    x_train = x_train.reshape(x_train.shape[0], input_shape)
    print('x_train shape:', x_train.shape)
    print('input_shape:', input_shape)
    
    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')
    
    # use hot encoding to transform output labels into integers
    y_train_hot = np_utils.to_categorical(y_train, num_classes)
    print('New y_train shape: ', y_train_hot.shape)
    
    # perform neural network classification
    model_m = Sequential()
    model_m.add(Reshape((TIME_PERIODS, 3), input_shape=(input_shape,)))
    model_m.add(Dense(100, activation='relu'))
    model_m.add(Dense(100, activation='relu'))
    model_m.add(Dense(100, activation='relu'))
    model_m.add(Flatten())
    model_m.add(Dense(num_classes, activation='softmax'))
    print(model_m.summary())
    
    # save the best algorithms as checkpoints (best means high accuracy, low val_loss)
    callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='acc', patience=1)]

    model_m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # hyper-parameters
    BATCH_SIZE = 200 # 200
    EPOCHS = 15

    # perform Keras ML algorithm development
    history = model_m.fit(x_train,
                      y_train_hot,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.2,
                      verbose=1)
                      
    # plot visualization of accuracy/loss over all epochs so user can see trends                  
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

    # print confusion matrix for training data
    y_pred_train = model_m.predict(x_train)
    # Take the class with the highest probability from the train predictions
    max_y_pred_train = np.argmax(y_pred_train, axis=1)
    print(classification_report(y_train, max_y_pred_train))
    
    # save final model 
    model_m.save('my_model.h5')
    
    # try using model to predict specific 
    # 0 = deadlift, 1 = squat
    # print(x_train[800]) # pick any number to try 
    #print(y_train[800])
    #print('\nPrediction from Keras:')
    #test_record = x_train[800].reshape(1,input_shape)
    #keras_prediction = np.argmax(model_m.predict(test_record), axis=1)
    #print(le.inverse_transform(keras_prediction)[0])
    
    #print('\nPrediction from Coreml:')
    #coreml_prediction = coreml_model.predict({'accelData': test_record.reshape(input_shape)})
    #print(coreml_prediction["classLabel"])
    
#######################################################################################################    

file_num = 1 # initialize file number to start at 1

writer = csv.writer(open('all_data.csv', 'w')) # create file to hold all data
for file in os.listdir("data/"): # read in all files in the smartbell/ml/data folder
    prev = 0
    read_data(file)
    file_num +=1

machine_learn() # create ML model 


