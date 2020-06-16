# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 18:43:25 2020
CWRU functions
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
from sklearn.metrics import confusion_matrix, accuracy_score
import scipy.io
from skimage import util
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras import models
from keras.optimizers import Adam


# %% Defining some constants to be used throughout
# number os data points per set
data_points = 66600
# image width
img_w = 22
# image length
img_h = 22
# matrix used to hold final images
A = np.zeros((0, img_w, img_h))
# image length when unidimensional
img_length = img_w*img_h
# number of data points used to build image
N = img_length*2
# overlap when making samples
div = 1
step = N//div
# # images in each class
samples_per_class = ((data_points)//(N-(N-step)))-(div-1)
# bitmap style
style = "viridis_r"


# %% normal baseline at 48ksps
# 0hp (5 seconds only)
# import matlab file using scipy
normal_48_0hp = scipy.io.loadmat('./48ksps/0hp/97.mat')
# get only the acc data points
normal_48_0hp = normal_48_0hp['X097_DE_time']
# undersample by 4 to get 12ksps (resampled)
normal_48_0hp_rs = normal_48_0hp[::4]
# resample to 20ksps
normal_48_0hp_rs = scipy.signal.resample(normal_48_0hp, data_points)

# 1hp
# import matlab file using scipy
normal_48_1hp = scipy.io.loadmat('./48ksps/1hp/98.mat')
# get only the acc data points
normal_48_1hp = normal_48_1hp['X098_DE_time']
# undersample by 4 to get 12ksps (resampled)
normal_48_1hp_rs = normal_48_1hp[::4]
# resample to 20ksps
normal_48_1hp_rs = scipy.signal.resample(normal_48_1hp, data_points)

# 2hp
# import matlab file using scipy
normal_48_2hp = scipy.io.loadmat('./48ksps/2hp/99.mat')
# get only the acc data points
normal_48_2hp = normal_48_2hp['X099_DE_time']
# undersample by 4 to get 12ksps (resampled)
normal_48_2hp_rs = normal_48_2hp[::4]
# resample to 20ksps
normal_48_2hp_rs = scipy.signal.resample(normal_48_2hp, data_points)

# 3hp
# import matlab file using scipy
normal_48_3hp = scipy.io.loadmat('./48ksps/3hp/100.mat')
# get only the acc data points
normal_48_3hp = normal_48_3hp['X100_DE_time']
# undersample by 4 to get 12ksps (resampled)
normal_48_3hp_rs = normal_48_3hp[::4]
# resample to 20ksps
normal_48_3hp_rs = scipy.signal.resample(normal_48_3hp, data_points)

# %% rolling element (ball) at 48ksps
# 0hp (5 seconds)
# import matlab file using scipy
ball_48_0hp = scipy.io.loadmat('./48ksps/0hp/226.mat')
# get only the acc data points
ball_48_0hp = ball_48_0hp['X226_DE_time']
# undersample to 12ksps
ball_48_0hp_rs = ball_48_0hp[::4]
# resample to 20ksps
ball_48_0hp_rs = scipy.signal.resample(ball_48_0hp, data_points)

# 1hp
# import matlab file using scipy
ball_48_1hp = scipy.io.loadmat('./48ksps/1hp/227.mat')
# get only the acc data points
ball_48_1hp = ball_48_1hp['X227_DE_time']
# undersample to 12ksps
ball_48_1hp_rs = ball_48_1hp[::4]
# resample to 20ksps
ball_48_1hp_rs = scipy.signal.resample(ball_48_1hp, data_points)

# 2hp
# import matlab file using scipy
ball_48_2hp = scipy.io.loadmat('./48ksps/2hp/228.mat')
# get only the acc data points
ball_48_2hp = ball_48_2hp['X228_DE_time']
# undersample to 12ksps
ball_48_2hp_rs = ball_48_2hp[::4]
# resample to 20ksps
ball_48_2hp_rs = scipy.signal.resample(ball_48_2hp, data_points)

# 3hp
# import matlab file using scipy
ball_48_3hp = scipy.io.loadmat('./48ksps/3hp/229.mat')
# get only the acc data points
ball_48_3hp = ball_48_3hp['X229_DE_time']
# undersample to 12ksps
ball_48_3hp_rs = ball_48_1hp[::4]
# resample to 20ksps
ball_48_3hp_rs = scipy.signal.resample(ball_48_3hp, data_points)

# %% inner race at 48ksps
# 0hp (5 seconds)
# import matlab file using scipy
inner_race_48_0hp = scipy.io.loadmat('./48ksps/0hp/213.mat')
# get only the acc data points
inner_race_48_0hp = inner_race_48_0hp['X213_DE_time']
# undersample to 12ksps
inner_race_48_0hp_rs = inner_race_48_0hp[::4]
# resample to 20ksps
inner_race_48_0hp_rs = scipy.signal.resample(inner_race_48_0hp, data_points)

# 1hp
# import matlab file using scipy
inner_race_48_1hp = scipy.io.loadmat('./48ksps/1hp/214.mat')
# get only the acc data points
inner_race_48_1hp = inner_race_48_1hp['X214_DE_time']
# undersample to 12ksps
inner_race_48_1hp_rs = inner_race_48_1hp[::4]
# resample to 20ksps
inner_race_48_1hp_rs = scipy.signal.resample(inner_race_48_1hp, data_points)

# 2hp
# import matlab file using scipy
inner_race_48_2hp = scipy.io.loadmat('./48ksps/2hp/215.mat')
# get only the acc data points
inner_race_48_2hp = inner_race_48_2hp['X215_DE_time']
# undersample to 12ksps
inner_race_48_2hp_rs = inner_race_48_2hp[::4]
# resample to 20ksps
inner_race_48_2hp_rs = scipy.signal.resample(inner_race_48_2hp, data_points)

# 3hp
# import matlab file using scipy
inner_race_48_3hp = scipy.io.loadmat('./48ksps/3hp/217.mat')
# get only the acc data points
inner_race_48_3hp = inner_race_48_3hp['X217_DE_time']
# undersample to 12ksps
inner_race_48_3hp_rs = inner_race_48_3hp[::4]
# resample to 20ksps
inner_race_48_3hp_rs = scipy.signal.resample(inner_race_48_3hp, data_points)

# %% outer race at 3oclock at 48ksps
# 0hp
# import matlab file using scipy
outer_race_at3_48_0hp = scipy.io.loadmat('./48ksps/0hp/250.mat')
# get only the acc data points
outer_race_at3_48_0hp = outer_race_at3_48_0hp['X250_DE_time']
# undersample to 12ksps
outer_race_at3_48_0hp_rs = outer_race_at3_48_0hp[::4]
# resample to 20ksps
outer_race_at3_48_0hp_rs = scipy.signal.resample(outer_race_at3_48_0hp,
                                                 data_points)

# 1hp
# import matlab file using scipy
outer_race_at3_48_1hp = scipy.io.loadmat('./48ksps/1hp/251.mat')
# get only the acc data points
outer_race_at3_48_1hp = outer_race_at3_48_1hp['X251_DE_time']
# undersample to 12ksps
outer_race_at3_48_1hp_rs = outer_race_at3_48_1hp[::4]
# resample to 20ksps
outer_race_at3_48_1hp_rs = scipy.signal.resample(outer_race_at3_48_1hp,
                                                 data_points)

# 2hp
# import matlab file using scipy
outer_race_at3_48_2hp = scipy.io.loadmat('./48ksps/2hp/252.mat')
# get only the acc data points
outer_race_at3_48_2hp = outer_race_at3_48_2hp['X252_DE_time']
# undersample to 12ksps
outer_race_at3_48_2hp_rs = outer_race_at3_48_2hp[::4]
# resample to 20ksps
outer_race_at3_48_2hp_rs = scipy.signal.resample(outer_race_at3_48_2hp,
                                                 data_points)

# 3hp
# import matlab file using scipy
outer_race_at3_48_3hp = scipy.io.loadmat('./48ksps/3hp/253.mat')
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
outer_race_at6_48_0hp = scipy.io.loadmat('./48ksps/0hp/238.mat')
# get only the acc data points
outer_race_at6_48_0hp = outer_race_at6_48_0hp['X238_DE_time']
# undersample to 12ksps
outer_race_at6_48_0hp_rs = outer_race_at6_48_0hp[::4]
# resample to 20ksps
outer_race_at6_48_0hp_rs = scipy.signal.resample(outer_race_at6_48_0hp,
                                                 data_points)

# 1hp
# import matlab file using scipy
outer_race_at6_48_1hp = scipy.io.loadmat('./48ksps/1hp/239.mat')
# get only the acc data points
outer_race_at6_48_1hp = outer_race_at6_48_1hp['X239_DE_time']
# undersample to 12ksps
outer_race_at6_48_1hp_rs = outer_race_at6_48_1hp[::4]
# resample to 20ksps
outer_race_at6_48_1hp_rs = scipy.signal.resample(outer_race_at6_48_1hp,
                                                 data_points)

# 2hp
# import matlab file using scipy
outer_race_at6_48_2hp = scipy.io.loadmat('./48ksps/2hp/240.mat')
# get only the acc data points
outer_race_at6_48_2hp = outer_race_at6_48_2hp['X240_DE_time']
# undersample to 12ksps
outer_race_at6_48_2hp_rs = outer_race_at6_48_2hp[::4]
# resample to 20ksps
outer_race_at6_48_2hp_rs = scipy.signal.resample(outer_race_at6_48_2hp,
                                                 data_points)

# 3hp
# import matlab file using scipy
outer_race_at6_48_3hp = scipy.io.loadmat('./48ksps/3hp/241.mat')
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
outer_race_at12_48_0hp = scipy.io.loadmat('./48ksps/0hp/262.mat')
# get only the acc data points
outer_race_at12_48_0hp = outer_race_at12_48_0hp['X262_DE_time']
# undersample to 12ksps
outer_race_at12_48_0hp_rs = outer_race_at12_48_0hp[::4]
# resample to 20ksps
outer_race_at12_48_0hp_rs = scipy.signal.resample(outer_race_at12_48_0hp,
                                                  data_points)

# 1hp
# import matlab file using scipy
outer_race_at12_48_1hp = scipy.io.loadmat('./48ksps/1hp/263.mat')
# get only the acc data points
outer_race_at12_48_1hp = outer_race_at12_48_1hp['X263_DE_time']
# undersample to 12ksps
outer_race_at12_48_1hp_rs = outer_race_at12_48_1hp[::4]
# resample to 20ksps
outer_race_at12_48_1hp_rs = scipy.signal.resample(outer_race_at12_48_1hp,
                                                  data_points)

# 2hp
# import matlab file using scipy
outer_race_at12_48_2hp = scipy.io.loadmat('./48ksps/2hp/264.mat')
# get only the acc data points
outer_race_at12_48_2hp = outer_race_at12_48_2hp['X264_DE_time']
# undersample to 12ksps
outer_race_at12_48_2hp_rs = outer_race_at12_48_2hp[::4]
# resample to 20ksps
outer_race_at12_48_2hp_rs = scipy.signal.resample(outer_race_at12_48_2hp,
                                                  data_points)

# 3hp
# import matlab file using scipy
outer_race_at12_48_3hp = scipy.io.loadmat('./48ksps/3hp/265.mat')
# get only the acc data points
outer_race_at12_48_3hp = outer_race_at12_48_3hp['X265_DE_time']
# undersample to 12ksps
outer_race_at12_48_3hp_rs = outer_race_at12_48_3hp[::4]
# resample to 20ksps
outer_race_at12_48_3hp_rs = scipy.signal.resample(outer_race_at12_48_3hp,
                                                  data_points)

# %% build final data set for each class using both 48ksps and 12ksps
# (resampled accordingly)
# normal = np.append(normal_48, normal_48_rs)
# inner_race = np.append(inner_race_48, inner_race_12)
# ball = np.append(ball_48, ball_12)
# outer_race_at3 = np.append(outer_race_at3_48, outer_race_at3_12)
# outer_race_at6 = np.append(outer_race_at6_48, outer_race_at6_12)
# outer_race_at12 = np.append(outer_race_at12_48, outer_race_at12_12)

# %% some statistics
df_48_rs = pd.DataFrame(inner_race_48_2hp_rs)
df_48 = pd.DataFrame(inner_race_48_2hp)
avg_48_rs = sum(inner_race_48_2hp_rs)/len(inner_race_48_2hp_rs)
avg_48 = sum(inner_race_48_2hp)/len(inner_race_48_2hp)
rms_48_rs = np.sqrt(np.mean(inner_race_48_2hp_rs**2))
rms_48 = np.sqrt(np.mean(inner_race_48_2hp**2))

# %%


def generate_images(df, step):

    df = df.reshape(df.size,)
    slices = util.view_as_windows(df, window_shape=(N, ), step=(step))
    print(f'Signal shape: {df.shape}, Sliced signal shape: {slices.shape}')

    B = np.zeros((0, img_w, img_h))

    for slice in slices:
        fftsl = np.abs(fft(slice)[:N // 2])
        T = []
        for x in range(0, len(fftsl)):
            T.append(abs(np.log2(fftsl[x]/img_length)))
        T = np.asarray(T)

        n = np.max(T)
        H = T/n
        # H = (255*H).astype(np.uint8)

        H.shape = (1, H.size//img_h, img_h)
        B = np.insert(B, 0, H, axis=0)

    Ht = H
    Ht.shape = (Ht.size//img_h, img_h)
    plt.imshow(Ht, cmap=style)
    plt.show()

    return B


# %% build images for normal baseline at 48ksps, 0hp, 1hp, 2hp and 3hp
# 0hp

# 1hp
Imgs = generate_images(normal_48_1hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)
# 2hp
Imgs = generate_images(normal_48_2hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)
# 3hp
Imgs = generate_images(normal_48_3hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)

# %% build images for rolling element (ball) failure at 48ksps
# 0hp

# 1hp
Imgs = generate_images(ball_48_1hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)
# 2hp
Imgs = generate_images(ball_48_2hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)
# 3hp
Imgs = generate_images(ball_48_3hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)

# %% build images for inner race failure at 48ksps
# 0hp

# 1hp
Imgs = generate_images(inner_race_48_1hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)
# 2hp
Imgs = generate_images(inner_race_48_2hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)
# 3hp
Imgs = generate_images(inner_race_48_3hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)

# %% build images for outer race at 3oclock failure at 48ksps
# 0hp

# 1hp
Imgs = generate_images(outer_race_at3_48_1hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)
# 2hp
Imgs = generate_images(outer_race_at3_48_2hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)
# 3hp
Imgs = generate_images(outer_race_at3_48_3hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)

# %% build images for outer race at 6oclock failure at 48ksps
# 0hp

# 1hp
Imgs = generate_images(outer_race_at6_48_1hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)
# 2hp
Imgs = generate_images(outer_race_at6_48_2hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)
# 3hp
Imgs = generate_images(outer_race_at6_48_3hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)

# %% build images for outer race at 12oclock failure at 48ksps
# 0hp

# 1hp
Imgs = generate_images(outer_race_at12_48_1hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)
# 2hp
Imgs = generate_images(outer_race_at12_48_2hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)
# 3hp
Imgs = generate_images(outer_race_at12_48_3hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)

# %% Reshape A
A = A.reshape(A.shape[0], img_w, img_h, 1)

# %% Appy labels to samples
# Label1 identifies only normal baseline and fault, two classes
# label1 = np.zeros(samples_per_class*6)
# label1[0:(samples_per_class*4)] = 1

# label2 identifies normal baseline and each specific fault at 2hp, six classes
label2 = np.zeros(samples_per_class*6)
label2[0:samples_per_class] = 5
label2[samples_per_class:samples_per_class*2] = 4
label2[samples_per_class*2:samples_per_class*3] = 3
label2[samples_per_class*3:samples_per_class*4] = 2
label2[samples_per_class*4:samples_per_class*5] = 1
label2 = np_utils.to_categorical(label2, 6)

# label3 identifies normal baseline and each specific fault for each load,
# 18 classes
label3 = np.zeros(samples_per_class*18)
label3[0:samples_per_class] = 17
label3[samples_per_class:samples_per_class*2] = 16
label3[samples_per_class*2:samples_per_class*3] = 15

label3[samples_per_class*3:samples_per_class*4] = 14
label3[samples_per_class*4:samples_per_class*5] = 13
label3[samples_per_class*5:samples_per_class*6] = 12

label3[samples_per_class*6:samples_per_class*7] = 11
label3[samples_per_class*7:samples_per_class*8] = 10
label3[samples_per_class*8:samples_per_class*9] = 9

label3[samples_per_class*9:samples_per_class*10] = 8
label3[samples_per_class*10:samples_per_class*11] = 7
label3[samples_per_class*11:samples_per_class*12] = 6

label3[samples_per_class*12:samples_per_class*13] = 5
label3[samples_per_class*13:samples_per_class*14] = 4
label3[samples_per_class*14:samples_per_class*15] = 3

label3[samples_per_class*15:samples_per_class*16] = 2
label3[samples_per_class*16:samples_per_class*17] = 1
# label3[samples_per_class*17:samples_per_class*18] = 1
label3 = np_utils.to_categorical(label3, 18)

# label4 identifies normal baseline and each specific fault accross different
# loads, 6 classes
label4 = np.zeros(samples_per_class*6*3)
label4[0:samples_per_class*3] = 5
label4[samples_per_class*3:samples_per_class*2*3] = 4
label4[samples_per_class*2*3:samples_per_class*3*3] = 3
label4[samples_per_class*3*3:samples_per_class*4*3] = 2
label4[samples_per_class*4*3:samples_per_class*5*3] = 1
label4 = np_utils.to_categorical(label4, 6)
# %% Separate classes, labels, train and test
# X_train, X_test, y_train, y_test =train_test_split(A, label1, test_size=0.25)
X_train, X_test, y_train, y_test = train_test_split(A, label4, test_size=0.25)
# X_train = X_train.astype('uint8')
# X_test = X_test.astype('uint8')

# %% Build first CNN

# define as sequential
model1 = models.Sequential()
# add first convolutional layer
model1.add(Conv2D(16, (3, 3), activation='relu',
                  input_shape=(img_w, img_h, 1)))
# add first max pooling layer
model1.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
# add second convolutional layer
model1.add(Conv2D(32, (3, 3), activation='relu'))
# add second max pooling layer
model1.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
# flatten before mlp
model1.add(Flatten())
# add fully connected wih 128 neurons and relu activation
model1.add(Dense(128, activation='relu'))
# output six classes with softmax activtion
model1.add(Dense(6, activation='softmax'))

# print CNN info
model1.summary()
# compile CNN and define its functions
model1.compile(loss='categorical_crossentropy', optimizer=Adam(),
               metrics=['accuracy'])

# %% Train CNN model1
model1.fit(X_train, y_train, batch_size=10, nb_epoch=100,
           validation_data=(X_test, y_test))

# %% Export model to .h5 file
model1.save("./model1_28x28_float_6k.h5")
print("Saved model 1 to disk")

# %% Results
# Make inference
# Predict and normalize predictions into 0s and 1s
predictions = model1.predict(X_test)
predictions2 = (predictions > 0.5)
# Find accuracy of inference
accuracy = accuracy_score(y_test, predictions2)
print(accuracy)
# calculate confusion matrix and print it
matrix = confusion_matrix(y_test.argmax(axis=1), predictions2.argmax(axis=1))
print(matrix)

# Use evaluate to test, just another way to do the same thing
result = model1.evaluate(X_test, y_test)
print(result)
confusion_matrix(y_test.argmax(axis=1), predictions2.argmax(axis=1))
###############################################################################

# # %% Same as above for another, simpler CNN model
# # define as sequential
# model2 = models.Sequential()
# # add first convolutional layer
# model2.add(Conv2D(2, (2, 2), activation='relu',
#                   input_shape=(img_w, img_h, 1)))
# # add first max pooling layer
# model2.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
# # flatten befor MLP
# model2.add(Flatten())
# # add fully connected wih 8 neurons and relu activation
# model2.add(Dense(8, activation='relu'))
# # output six classes with softmax activtion
# model2.add(Dense(6, activation='softmax'))

# # print CNN info
# model2.summary()
# # compile CNN and define its functions
# model2.compile(loss='categorical_crossentropy', optimizer=Adam(),
#                metrics=['accuracy'])

# # %% Train CNN model2
# model2.fit(X_train, y_train, batch_size=10, nb_epoch=100,
#            validation_data=(X_test, y_test))

# # %% Export model to .h5 file
# model2.save("./model2.h5")
# print("Saved model to disk")

# # %% Results
# # Make inference
# # Predict and normalize predictions into 0s and 1s
# predictions = model2.predict(X_test)
# predictions = (predictions > 0.5)
# # Find accuracy of inference
# accuracy = accuracy_score(y_test, predictions)
# print(accuracy)
# # calculate confusion matrix and print it
# matrix = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
# print(matrix)

# # Use evaluate to test, just another way to do the same thing
# result = model2.evaluate(X_test, y_test)
# print(result)
# confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
