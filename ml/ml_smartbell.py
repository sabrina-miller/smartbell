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

sns.set() # Graph aesthetics
# The number of steps within one time segment
TIME_PERIODS = 15
# The steps to take from one segment to the next; if this value is equal to
# TIME_PERIODS, then there is no overlap between the segments
STEP_DISTANCE = 8

# 10, 5: 94% but skewed weirdly
# 10, 8: 93%
# 10, 10: 92%
# 20, 10: 93% 
# 20, 20: 92%
# 40, 40: 90% 
# 40, 20: 93% Weirdly skewed also

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

    
def machine_learn():

    df = pd.read_csv('all_data.csv')
    df['x-axis (g)'] = df['x-axis (g)'].astype(float)
    df['y-axis (g)'] = df['y-axis (g)'].astype(float)
    df['z-axis (g)'] = df['z-axis (g)'].astype(float)
    df.drop('epoc (ms)', axis=1, inplace=True)
    df.drop('elapsed (s)', axis=1, inplace=True)
    
    # Data analysis (can delete later)
    df['activity'].value_counts().plot(kind='bar', title='Training Examples by Activity Type')
    #plt.show()
    df['person'].value_counts().plot(kind='bar', title='Training Examples by User')
    #plt.show()

    le = preprocessing.LabelEncoder() # Convert labels from string to integer
    df['ActivityEncoded'] = le.fit_transform(df['activity'].values.ravel()) # add encoded values to dataframe

    # Differentiate between test set and training set
    df_test = df[df['file_num'] <= 12]
    df_train = df[df['file_num'] > 12]
    
    # x, y, z acceleration as features
    N_FEATURES = 3
    # Number of steps to advance in each iteration
    segments = []
    labels = []
    for i in range(0, len(df_train) - TIME_PERIODS, STEP_DISTANCE):
        xs = df_train['x-axis (g)'].values[i: i + TIME_PERIODS]
        ys = df_train['y-axis (g)'].values[i: i + TIME_PERIODS]
        zs = df_train['z-axis (g)'].values[i: i + TIME_PERIODS]
        # Retrieve the most often used label in this segment
        label = stats.mode(df_train['ActivityEncoded'][i: i + TIME_PERIODS])[0][0]
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
    num_classes = le.classes_.size # num_classes = 2
    input_shape = (num_time_periods*num_sensors)
    x_train = x_train.reshape(x_train.shape[0], input_shape)
    print('x_train shape:', x_train.shape)
    print('input_shape:', input_shape)
    
    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')
    
    y_train_hot = np_utils.to_categorical(y_train, num_classes)
    print('New y_train shape: ', y_train_hot.shape)
    
    model_m = Sequential()
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
    BATCH_SIZE = 200 # 200
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
    
    model_m.save('my_model.h5')
    
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
    
    
    
file_num = 1
writer = csv.writer(open('all_data.csv', 'w'))
for file in os.listdir("data/"):
    prev = 0
    read_data(file)
    file_num +=1

machine_learn()


