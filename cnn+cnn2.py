# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 10:09:21 2020

@author: caiom
"""

# %%
# Importing modules
import numpy as np
import pandas as pd
from keras.utils import np_utils
import scipy.io
import scipy.signal
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score)
from tensorflow.keras.callbacks import EarlyStopping
import scipy.io
from skimage import util
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, AveragePooling2D, Embedding, LSTM
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam
from statistics import mean
import pickle

# import serial
import struct

from time import sleep

# %% Defining some constants to be used throughout
# number os data points per set
data_points = 480000
sample_size = 2000

# images in each class
samples_per_class = (data_points)//sample_size

Sl = np.zeros((0, sample_size))

# # number of data points used to build image
# N = img_length*2
# # overlap when making samples
# div = 1
# step = N//div
# # # images in each class
# samples_per_class = ((data_points)//(N-(N-step)))-(div-1)

# bitmap style
style = "viridis_r"

# path to files
path = 'C:/Users/caiom/Documents/propecaut2019/disciplinas/topicos-wavelet/CWRU-DE/48ksps/'


# %% normal baseline at 48ksps
# 0hp (5 seconds only)
# import matlab file using scipy
normal_48_0hp = scipy.io.loadmat(path+'0hp/97.mat')
# get only the acc data points
normal_48_0hp = normal_48_0hp['X097_DE_time']
# undersample by 4 to get 12ksps (resampled)
normal_48_0hp_rs = normal_48_0hp[::4]
# resample to 20ksps
normal_48_0hp_rs = scipy.signal.resample(normal_48_0hp, data_points)

# 1hp
# import matlab file using scipy
normal_48_1hp = scipy.io.loadmat(path+'1hp/98.mat')
# get only the acc data points
normal_48_1hp = normal_48_1hp['X098_DE_time']
# undersample by 4 to get 12ksps (resampled)
normal_48_1hp_rs = normal_48_1hp[::4]
# resample to 20ksps
normal_48_1hp_rs = scipy.signal.resample(normal_48_1hp, data_points)

# 2hp
# import matlab file using scipy
normal_48_2hp = scipy.io.loadmat(path+'2hp/99.mat')
# get only the acc data points
normal_48_2hp = normal_48_2hp['X099_DE_time']
# undersample by 4 to get 12ksps (resampled)
normal_48_2hp_rs = normal_48_2hp[::4]
# resample to 20ksps
normal_48_2hp_rs = scipy.signal.resample(normal_48_2hp, data_points)

# 3hp
# import matlab file using scipy
normal_48_3hp = scipy.io.loadmat(path+'3hp/100.mat')
# get only the acc data points
normal_48_3hp = normal_48_3hp['X100_DE_time']
# undersample by 4 to get 12ksps (resampled)
normal_48_3hp_rs = normal_48_3hp[::4]
# resample to 20ksps
normal_48_3hp_rs = scipy.signal.resample(normal_48_3hp, data_points)

# %% rolling element (ball) at 48ksps
# 0hp (5 seconds)
# import matlab file using scipy
ball_48_0hp = scipy.io.loadmat(path+'0hp/226.mat')
# get only the acc data points
ball_48_0hp = ball_48_0hp['X226_DE_time']
# undersample to 12ksps
ball_48_0hp_rs = ball_48_0hp[::4]
# resample to 20ksps
ball_48_0hp_rs = scipy.signal.resample(ball_48_0hp, data_points)

# 1hp
# import matlab file using scipy
ball_48_1hp = scipy.io.loadmat(path+'1hp/227.mat')
# get only the acc data points
ball_48_1hp = ball_48_1hp['X227_DE_time']
# undersample to 12ksps
ball_48_1hp_rs = ball_48_1hp[::4]
# resample to 20ksps
ball_48_1hp_rs = scipy.signal.resample(ball_48_1hp, data_points)

# 2hp
# import matlab file using scipy
ball_48_2hp = scipy.io.loadmat(path+'2hp/228.mat')
# get only the acc data points
ball_48_2hp = ball_48_2hp['X228_DE_time']
# undersample to 12ksps
ball_48_2hp_rs = ball_48_2hp[::4]
# resample to 20ksps
ball_48_2hp_rs = scipy.signal.resample(ball_48_2hp, data_points)

# 3hp
# import matlab file using scipy
ball_48_3hp = scipy.io.loadmat(path+'3hp/229.mat')
# get only the acc data points
ball_48_3hp = ball_48_3hp['X229_DE_time']
# undersample to 12ksps
ball_48_3hp_rs = ball_48_1hp[::4]
# resample to 20ksps
ball_48_3hp_rs = scipy.signal.resample(ball_48_3hp, data_points)

# %% inner race at 48ksps
# 0hp (5 seconds)
# import matlab file using scipy
inner_race_48_0hp = scipy.io.loadmat(path+'0hp/213.mat')
# get only the acc data points
inner_race_48_0hp = inner_race_48_0hp['X213_DE_time']
# undersample to 12ksps
inner_race_48_0hp_rs = inner_race_48_0hp[::4]
# resample to 20ksps
inner_race_48_0hp_rs = scipy.signal.resample(inner_race_48_0hp, data_points)

# 1hp
# import matlab file using scipy
inner_race_48_1hp = scipy.io.loadmat(path+'1hp/214.mat')
# get only the acc data points
inner_race_48_1hp = inner_race_48_1hp['X214_DE_time']
# undersample to 12ksps
inner_race_48_1hp_rs = inner_race_48_1hp[::4]
# resample to 20ksps
inner_race_48_1hp_rs = scipy.signal.resample(inner_race_48_1hp, data_points)

# 2hp
# import matlab file using scipy
inner_race_48_2hp = scipy.io.loadmat(path+'2hp/215.mat')
# get only the acc data points
inner_race_48_2hp = inner_race_48_2hp['X215_DE_time']
# undersample to 12ksps
inner_race_48_2hp_rs = inner_race_48_2hp[::4]
# resample to 20ksps
inner_race_48_2hp_rs = scipy.signal.resample(inner_race_48_2hp, data_points)

# 3hp
# import matlab file using scipy
inner_race_48_3hp = scipy.io.loadmat(path+'3hp/217.mat')
# get only the acc data points
inner_race_48_3hp = inner_race_48_3hp['X217_DE_time']
# undersample to 12ksps
inner_race_48_3hp_rs = inner_race_48_3hp[::4]
# resample to 20ksps
inner_race_48_3hp_rs = scipy.signal.resample(inner_race_48_3hp, data_points)

# %% outer race at 3oclock at 48ksps
# 0hp
# import matlab file using scipy
outer_race_at3_48_0hp = scipy.io.loadmat(path+'0hp/250.mat')
# get only the acc data points
outer_race_at3_48_0hp = outer_race_at3_48_0hp['X250_DE_time']
# undersample to 12ksps
outer_race_at3_48_0hp_rs = outer_race_at3_48_0hp[::4]
# resample to 20ksps
outer_race_at3_48_0hp_rs = scipy.signal.resample(outer_race_at3_48_0hp,
                                                 data_points)

# 1hp
# import matlab file using scipy
outer_race_at3_48_1hp = scipy.io.loadmat(path+'1hp/251.mat')
# get only the acc data points
outer_race_at3_48_1hp = outer_race_at3_48_1hp['X251_DE_time']
# undersample to 12ksps
outer_race_at3_48_1hp_rs = outer_race_at3_48_1hp[::4]
# resample to 20ksps
outer_race_at3_48_1hp_rs = scipy.signal.resample(outer_race_at3_48_1hp,
                                                 data_points)

# 2hp
# import matlab file using scipy
outer_race_at3_48_2hp = scipy.io.loadmat(path+'2hp/252.mat')
# get only the acc data points
outer_race_at3_48_2hp = outer_race_at3_48_2hp['X252_DE_time']
# undersample to 12ksps
outer_race_at3_48_2hp_rs = outer_race_at3_48_2hp[::4]
# resample to 20ksps
outer_race_at3_48_2hp_rs = scipy.signal.resample(outer_race_at3_48_2hp,
                                                 data_points)

# 3hp
# import matlab file using scipy
outer_race_at3_48_3hp = scipy.io.loadmat(path+'3hp/253.mat')
# get only the acc data points
outer_race_at3_48_3hp = outer_race_at3_48_3hp['X253_DE_time']
# undersample to 12ksps
outer_race_at3_48_3hp_rs = outer_race_at3_48_3hp[::4]
# resample to 20ksps
outer_race_at3_48_3hp_rs = scipy.signal.resample(outer_race_at3_48_3hp,
                                                 data_points)

# %% outer race at 6oclock at 48ksps
# 0hp
# import matlab file using scipy
outer_race_at6_48_0hp = scipy.io.loadmat(path+'0hp/238.mat')
# get only the acc data points
outer_race_at6_48_0hp = outer_race_at6_48_0hp['X238_DE_time']
# undersample to 12ksps
outer_race_at6_48_0hp_rs = outer_race_at6_48_0hp[::4]
# resample to 20ksps
outer_race_at6_48_0hp_rs = scipy.signal.resample(outer_race_at6_48_0hp,
                                                 data_points)

# 1hp
# import matlab file using scipy
outer_race_at6_48_1hp = scipy.io.loadmat(path+'1hp/239.mat')
# get only the acc data points
outer_race_at6_48_1hp = outer_race_at6_48_1hp['X239_DE_time']
# undersample to 12ksps
outer_race_at6_48_1hp_rs = outer_race_at6_48_1hp[::4]
# resample to 20ksps
outer_race_at6_48_1hp_rs = scipy.signal.resample(outer_race_at6_48_1hp,
                                                 data_points)

# 2hp
# import matlab file using scipy
outer_race_at6_48_2hp = scipy.io.loadmat(path+'2hp/240.mat')
# get only the acc data points
outer_race_at6_48_2hp = outer_race_at6_48_2hp['X240_DE_time']
# undersample to 12ksps
outer_race_at6_48_2hp_rs = outer_race_at6_48_2hp[::4]
# resample to 20ksps
outer_race_at6_48_2hp_rs = scipy.signal.resample(outer_race_at6_48_2hp,
                                                 data_points)

# 3hp
# import matlab file using scipy
outer_race_at6_48_3hp = scipy.io.loadmat(path+'3hp/241.mat')
# get only the acc data points
outer_race_at6_48_3hp = outer_race_at6_48_3hp['X241_DE_time']
# undersample to 12ksps
outer_race_at6_48_3hp_rs = outer_race_at6_48_3hp[::4]
# resample to 20ksps
outer_race_at6_48_3hp_rs = scipy.signal.resample(outer_race_at6_48_3hp,
                                                 data_points)

# %% outer race at 12oclock at 48ksps
# 0hp
# import matlab file using scipy
outer_race_at12_48_0hp = scipy.io.loadmat(path+'0hp/262.mat')
# get only the acc data points
outer_race_at12_48_0hp = outer_race_at12_48_0hp['X262_DE_time']
# undersample to 12ksps
outer_race_at12_48_0hp_rs = outer_race_at12_48_0hp[::4]
# resample to 20ksps
outer_race_at12_48_0hp_rs = scipy.signal.resample(outer_race_at12_48_0hp,
                                                  data_points)

# 1hp
# import matlab file using scipy
outer_race_at12_48_1hp = scipy.io.loadmat(path+'1hp/263.mat')
# get only the acc data points
outer_race_at12_48_1hp = outer_race_at12_48_1hp['X263_DE_time']
# undersample to 12ksps
outer_race_at12_48_1hp_rs = outer_race_at12_48_1hp[::4]
# resample to 20ksps
outer_race_at12_48_1hp_rs = scipy.signal.resample(outer_race_at12_48_1hp,
                                                  data_points)

# 2hp
# import matlab file using scipy
outer_race_at12_48_2hp = scipy.io.loadmat(path+'2hp/264.mat')
# get only the acc data points
outer_race_at12_48_2hp = outer_race_at12_48_2hp['X264_DE_time']
# undersample to 12ksps
outer_race_at12_48_2hp_rs = outer_race_at12_48_2hp[::4]
# resample to 20ksps
outer_race_at12_48_2hp_rs = scipy.signal.resample(outer_race_at12_48_2hp,
                                                  data_points)

# 3hp
# import matlab file using scipy
outer_race_at12_48_3hp = scipy.io.loadmat(path+'3hp/265.mat')
# get only the acc data points
outer_race_at12_48_3hp = outer_race_at12_48_3hp['X265_DE_time']
# undersample to 12ksps
outer_race_at12_48_3hp_rs = outer_race_at12_48_3hp[::4]
# resample to 20ksps
outer_race_at12_48_3hp_rs = scipy.signal.resample(outer_race_at12_48_3hp,
                                                  data_points)

# %% some statistics
df_48_rs = pd.DataFrame(inner_race_48_2hp_rs)
df_48 = pd.DataFrame(inner_race_48_2hp)
avg_48_rs = sum(inner_race_48_2hp_rs)/len(inner_race_48_2hp_rs)
avg_12 = sum(inner_race_48_2hp)/len(inner_race_48_2hp)
rms_48_rs = np.sqrt(np.mean(inner_race_48_2hp_rs**2))
rms_48 = np.sqrt(np.mean(inner_race_48_2hp**2))

# %% slice everything in lengths of 2048


def slice_signal(df):
    df = df.reshape(df.size,)
    sl = util.view_as_windows(df, window_shape=(sample_size, ), step=(sample_size))
    print(f'Signal shape: {df.shape}, Sliced signal shape: {sl.shape}')
    return sl

# %% build images for normal baseline at 48ksps, 0hp, 1hp, 2hp and 3hp
# 0hp
slaices = slice_signal(normal_48_0hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

# 1hp
slaices = slice_signal(normal_48_1hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

# 2hp
slaices = slice_signal(normal_48_2hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

# 3hp
slaices = slice_signal(normal_48_3hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

# %% build images for rolling element (ball) failure at 48ksps
# 0hp
slaices = slice_signal(ball_48_0hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

# 1hp
slaices = slice_signal(ball_48_1hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

# 2hp
slaices = slice_signal(ball_48_2hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

# 3hp
slaices = slice_signal(ball_48_3hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

# %% build images for inner race failure at 48ksps
# 0hp
slaices = slice_signal(inner_race_48_0hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

# 1hp
slaices = slice_signal(inner_race_48_1hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

# 2hp
slaices = slice_signal(inner_race_48_2hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

# 3hp
slaices = slice_signal(inner_race_48_3hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

# %% build images for outer race at 3oclock failure at 48ksps
# 0hp
slaices = slice_signal(outer_race_at3_48_0hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

# 1hp
slaices = slice_signal(outer_race_at3_48_1hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

# 2hp
slaices = slice_signal(outer_race_at3_48_2hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

# 3hp
slaices = slice_signal(outer_race_at3_48_3hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

# %% build images for outer race at 6oclock failure at 48ksps
# 0hp
slaices = slice_signal(outer_race_at6_48_0hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

# 1hp
slaices = slice_signal(outer_race_at6_48_1hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

# 2hp
slaices = slice_signal(outer_race_at6_48_2hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

# 3hp
slaices = slice_signal(outer_race_at6_48_3hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

# %% build images for outer race at 12oclock failure at 48ksps
# 0hp
slaices = slice_signal(outer_race_at12_48_0hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

# 1hp
slaices = slice_signal(outer_race_at12_48_1hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

# 2hp
slaices = slice_signal(outer_race_at12_48_2hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

# 3hp
slaices = slice_signal(outer_race_at12_48_3hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)
# %%
Sl = Sl.reshape(Sl.shape[0], sample_size, 1)

# %% Appy labels to samples
# Label1 identifies only normal baseline and fault, two classes
# label1 = np.zeros(samples_per_class*18)
# label1[0:(samples_per_class*3)] = 1


# label2 identifies normal baseline and each specific fault for each load,
# 18 classes
label2 = np.zeros(samples_per_class*24)

label2[0:samples_per_class] = 23
label2[samples_per_class:samples_per_class*2] = 22
label2[samples_per_class*2:samples_per_class*3] = 21
label2[samples_per_class*3:samples_per_class*4] = 20
label2[samples_per_class*4:samples_per_class*5] = 19
label2[samples_per_class*5:samples_per_class*6] = 18
label2[samples_per_class*6:samples_per_class*7] = 17
label2[samples_per_class*7:samples_per_class*8] = 16
label2[samples_per_class*8:samples_per_class*9] = 15
label2[samples_per_class*9:samples_per_class*10] = 14
label2[samples_per_class*10:samples_per_class*11] = 13
label2[samples_per_class*11:samples_per_class*12] = 12
label2[samples_per_class*12:samples_per_class*13] = 11
label2[samples_per_class*13:samples_per_class*14] = 10
label2[samples_per_class*14:samples_per_class*15] = 9
label2[samples_per_class*15:samples_per_class*16] = 8
label2[samples_per_class*16:samples_per_class*17] = 7
label2[samples_per_class*17:samples_per_class*18] = 6
label2[samples_per_class*18:samples_per_class*19] = 5
label2[samples_per_class*19:samples_per_class*20] = 4
label2[samples_per_class*20:samples_per_class*21] = 3
label2[samples_per_class*21:samples_per_class*22] = 2
label2[samples_per_class*22:samples_per_class*23] = 1

label2 = np_utils.to_categorical(label2, 24)

# label3 identifies normal baseline and each specific fault accross different
# loads, 6 classes
label3 = np.zeros(samples_per_class*6*4)
label3[0:samples_per_class*4] = 5
label3[samples_per_class*4:samples_per_class*2*4] = 4
label3[samples_per_class*2*4:samples_per_class*3*4] = 3
label3[samples_per_class*3*4:samples_per_class*4*4] = 2
label3[samples_per_class*4*4:samples_per_class*5*4] = 1

label3 = np_utils.to_categorical(label3, 6)
# %%
confusion_matrices_1 = []
accuracies_1 = []
precisions_1 = []
recalls_1 = []
f1s_1 = []
confusion_matrices_2 = []
accuracies_2 = []
precisions_2 = []
recalls_2 = []
f1s_2 = []

# %%
# patient early stopping
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
es = EarlyStopping(monitor='val_accuracy', mode='max')

# %% Separate classes, labels, train and test
Sl_train, Sl_test, y_train2, y_test2 = train_test_split(Sl, label3,
                                                        test_size=0.3,
                                                        random_state=21)
# X_train = X_train.astype('uint8')
# X_test = X_test.astype('uint8')
# %%
from tensorflow.keras.layers import LSTM, Input, Conv1D, Conv2D, AveragePooling1D, Dense, Flatten, Reshape, MaxPooling2D, MaxPooling1D, Multiply, Permute
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model


# %%
in1 = Input(shape=(sample_size,1))

# %% parallel 1
sub_sample = AveragePooling1D(pool_size=8, strides=8)(in1)

# %% parallel 1
conv_1 = Conv1D(filters=50, kernel_size=20, strides=2, activation='tanh')(sub_sample)
print(conv_1.shape)
conv_2 = Conv1D(filters=30, kernel_size=10, strides=2, activation='tanh')(conv_1)
print(conv_2.shape)
pooling_1 = MaxPooling1D(pool_size=2, strides=2)(conv_2)
print(pooling_1.shape)
pooling_1_reshape = Reshape((27, 30))(pooling_1)
# %% parallel 2
conv_3 = Conv1D(filters=50, kernel_size=6, strides=1, activation='tanh')(sub_sample)
print(conv_3.shape)
conv_4 = Conv1D(filters=40, kernel_size=6, strides=1, activation='tanh')(conv_3)
print(conv_4.shape)
pooling_2 = MaxPooling1D(pool_size=2, strides=2)(conv_4)
print(pooling_2.shape)
conv_5 = Conv1D(filters=30, kernel_size=6, strides=1, activation='tanh')(pooling_2)
print(conv_5.shape)
conv_6 = Conv1D(filters=30, kernel_size=6, strides=2, activation='tanh')(conv_5)
print(conv_6.shape)
pooling_3 = MaxPooling1D(pool_size=2, strides=2)(conv_6)
print(pooling_3.shape)
pooling_3_reshape = Reshape((27, 30))(pooling_3)

# %% fusion
fused = Multiply()([pooling_1_reshape, pooling_3_reshape])
print(fused.shape)
fused = Permute((2,1))(fused)
print(fused.shape)
fused = Reshape((30,27,1))(fused)


# %% classifier
conv_7 = Conv2D(2, (2, 2), activation='relu')(fused)
pooling_4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv_7)
flat_1 = Flatten()(pooling_4)
dense_1 = Dense(8, activation='relu')(flat_1)
output = Dense(6, activation='softmax')(dense_1)

# %% build model
model = Model(inputs=in1, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# summarize layers
print(model.summary())
# plot graph
plot_model(model, to_file='shared_input_layer.png', show_shapes=True, show_layer_names=True)

# %% fit
history = model.fit(Sl_train, y_train2, validation_data=(Sl_test, y_test2), epochs=20, batch_size=10)
# Final evaluation of the model
scores = model.evaluate(Sl_test, y_test2, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# %%
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
plt.title('model 2 accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='lower right')
plt.show()


# Final evaluation of the model
scores = model.evaluate(Sl_test, y_test2, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# %% Results
# Make inference
# Predict and normalize predictions into 0s and 1s
predictions = model.predict(Sl_test)
predictions = (predictions > 0.5)
# Find accuracy of inference
accuracy = accuracy_score(y_test2, predictions)
precision = precision_score(y_test2, predictions, average='weighted')
recall = recall_score(y_test2, predictions, average='weighted')
f1 = f1_score(y_test2, predictions, average='weighted')
print(accuracy)
accuracies_2.append(accuracy)
precisions_2.append(precision)
recalls_2.append(recall)
f1s_2.append(f1)
# calculate confusion matrix and print it
matrix = confusion_matrix(y_test2.argmax(axis=1),
                          predictions.argmax(axis=1))
print(matrix)
confusion_matrices = (matrix)

# Use evaluate to test, just another way to do the same thing
result = model.evaluate(Sl_test, y_test2)
print(result)
confusion_matrix(y_test2.argmax(axis=1), predictions.argmax(axis=1))
# %%
confusion_1_final = sum(confusion_matrices_1)
print("matriz confusão")
print(matrix)

print("acurácia ", accuracy)
print("precisão", precision)
print("recall", recall)
print("f1", f1)

# %% save model
model.save("./model_cnn_cnn.h5")

# %% convert to tf lite
# import tensorflow as tf
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
# open("./sine_model.tflite", "wb").write(tflite_model)
# import os 
# basic_model_size = os.path.getsize("sine_model.tflite")
# print("Basic model is %d bytes" % basic_model_size)
