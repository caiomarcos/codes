# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 11:06:52 2020

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
from keras.callbacks import EarlyStopping
import scipy.io
from skimage import util
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, AveragePooling2D
from keras import models
from keras.optimizers import Adam
from statistics import mean
import pickle

import serial
import struct

from time import sleep

# %% Defining some constants to be used throughout
# number os data points per set
data_points = 480000
# image width
img_w = 28
# image length
img_h = 28
# matrix used to hold final images
A = np.zeros((0, img_w, img_h))

# image length when unidimensional
img_length = img_w*img_h

# FFT size (power of 2 in order to work with ARM DSP library)
FFT_size = 2048
# images in each class
samples_per_class = (data_points)//FFT_size

Sl = np.zeros((0, 2048))

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
# %%
# fig, axs = plt.subplots(6)

# # fig.suptitle('Amplitude de vibração')
# # fig.suptitle('Amplitude de vibração')
# axs[0].plot(outer_race_at12_48_2hp_rs[0:6000])
# axs[0].set_title('RE Oposto')
# axs[1].plot(outer_race_at6_48_2hp[0:6000])
# axs[1].set_title('RE Centralizado')
# axs[2].plot(outer_race_at3_48_2hp_rs[0:6000])
# axs[2].set_title('RE Ortogonal')
# axs[3].plot(inner_race_48_2hp_rs[0:6000])
# axs[3].set_title('RI')
# axs[4].plot(ball_48_2hp_rs[0:6000])
# axs[4].set_title('Esfera')
# axs[5].plot(normal_48_2hp_rs[0:6000])
# axs[5].set_title('Normal')
# # axs[0].set(ylabel='amplitude')
# for ax in axs.flat:
#     ax.set(xlabel='amostra')
# for ax in axs.flat:
#     ax.label_outer()

# fig.tight_layout(pad=0)
# fig.set_figheight(6)
# fig.set_figwidth(7)
# fig.tight_layout(pad=0)
# fig.text(0.0, 0.5, 'amplitude', ha='center', va='center',
#          rotation='vertical')

# plt.figure(figsize=(6, 2))
# plt.plot(outer_race_at12_48_2hp_rs[0:120000])
# plt.title('Sinal original - RE Oposto, 2 hp')
# plt.ylabel('amplitude')
# plt.xlabel('amostras')
# plt.show()

# plt.figure(figsize=(6, 2))
# plt.plot(outer_race_at12_48_2hp_rs[0:1568])
# plt.title('Sinal amostrado - RE Oposto, 2 hp')
# plt.ylabel('amplitude')
# plt.xlabel('amostras')
# plt.show()

# plt.plot(normal_48_2hp_rs[0:6000])
# plt.title('Normal')
# plt.ylabel('amplitude')
# plt.xlabel('amostras')
# plt.show()

# plt.plot(inner_race_48_2hp_rs[0:6000])
# plt.title('RI')
# plt.ylabel('amplitude')
# plt.xlabel('amostras')
# plt.show()

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
avg_12 = sum(inner_race_48_2hp)/len(inner_race_48_2hp)
rms_48_rs = np.sqrt(np.mean(inner_race_48_2hp_rs**2))
rms_48 = np.sqrt(np.mean(inner_race_48_2hp**2))

# %% slice everything in lengths of 2048


def slice_signal(df):
    df = df.reshape(df.size,)
    sl = util.view_as_windows(df, window_shape=(FFT_size, ), step=(FFT_size))
    print(f'Signal shape: {df.shape}, Sliced signal shape: {sl.shape}')
    return sl

# %%


def generate_images(slcs):

    # df = df.reshape(df.size,)
    # slices = util.view_as_windows(df, window_shape=(N, ), step=(step))
    # print(f'Signal shape: {df.shape}, Sliced signal shape: {slices.shape}')

    B = np.zeros((0, img_w, img_h))

    for slice in slcs:
        fftsl = np.abs(fft(slice)[:FFT_size // 2])
        T = []
        for x in range(0, img_length):
            T.append(abs(np.log2(fftsl[x]/img_length)))
        T = np.asarray(T)
        n = np.max(T)
        H = T/n
        # plt.figure(figsize=(6, 2))
        # plt.plot(H)
        # plt.show()
        # H = (255*H).astype(np.uint8)

        H.shape = (1, H.size//img_h, img_h)
        B = np.insert(B, 0, H, axis=0)

    Ht = H
    Ht.shape = (Ht.size//img_h, img_h)
    # plt.imshow(Ht, cmap=style)
    # plt.show()

    return (B, Ht)


# %% build images for normal baseline at 48ksps, 0hp, 1hp, 2hp and 3hp
# 0hp
slaices = slice_signal(normal_48_0hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

(Imgs, img) = generate_images(slaices)
A = np.insert(A, 0, Imgs, axis=0)
# 1hp
slaices = slice_signal(normal_48_1hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

(Imgs, img) = generate_images(slaices)
A = np.insert(A, 0, Imgs, axis=0)
# 2hp
slaices = slice_signal(normal_48_2hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

(Imgs, img1) = generate_images(slaices)
A = np.insert(A, 0, Imgs, axis=0)
plt.axis('off')
plt.title('Normal, 2 hp')
plt.imshow(img, cmap=style)
plt.show()
# 3hp
slaices = slice_signal(normal_48_3hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

(Imgs, img) = generate_images(slaices)
A = np.insert(A, 0, Imgs, axis=0)

# %% build images for rolling element (ball) failure at 48ksps
# 0hp
slaices = slice_signal(ball_48_0hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

(Imgs, img) = generate_images(slaices)
A = np.insert(A, 0, Imgs, axis=0)
# 1hp
slaices = slice_signal(ball_48_1hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

(Imgs, img) = generate_images(slaices)
A = np.insert(A, 0, Imgs, axis=0)
# 2hp
slaices = slice_signal(ball_48_2hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

(Imgs, img2) = generate_images(slaices)
A = np.insert(A, 0, Imgs, axis=0)
plt.axis('off')
plt.title('Esfera, 2 hp')
plt.imshow(img, cmap=style)
plt.show()
# 3hp
slaices = slice_signal(ball_48_3hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

(Imgs, img) = generate_images(slaices)
A = np.insert(A, 0, Imgs, axis=0)

# %% build images for inner race failure at 48ksps
# 0hp
slaices = slice_signal(inner_race_48_0hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

(Imgs, img) = generate_images(slaices)
A = np.insert(A, 0, Imgs, axis=0)
# 1hp
slaices = slice_signal(inner_race_48_1hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

(Imgs, img) = generate_images(slaices)
A = np.insert(A, 0, Imgs, axis=0)
# 2hp
slaices = slice_signal(inner_race_48_2hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

(Imgs, img3) = generate_images(slaices)
A = np.insert(A, 0, Imgs, axis=0)
plt.axis('off')
plt.title('RI, 2 hp')
plt.imshow(img, cmap=style)
plt.show()
# 3hp
slaices = slice_signal(inner_race_48_3hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

(Imgs, img) = generate_images(slaices)
A = np.insert(A, 0, Imgs, axis=0)

# %% build images for outer race at 3oclock failure at 48ksps
# 0hp
slaices = slice_signal(outer_race_at3_48_0hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

(Imgs, img) = generate_images(slaices)
A = np.insert(A, 0, Imgs, axis=0)
# 1hp
slaices = slice_signal(outer_race_at3_48_1hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

(Imgs, img) = generate_images(slaices)
A = np.insert(A, 0, Imgs, axis=0)
# 2hp
slaices = slice_signal(outer_race_at3_48_2hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

(Imgs, img4) = generate_images(slaices)
A = np.insert(A, 0, Imgs, axis=0)
plt.axis('off')
plt.title('RE Ortogonal, 2 hp')
plt.imshow(img, cmap=style)
plt.show()
# 3hp
slaices = slice_signal(outer_race_at3_48_3hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

(Imgs, img) = generate_images(slaices)
A = np.insert(A, 0, Imgs, axis=0)

# %% build images for outer race at 6oclock failure at 48ksps
# 0hp
slaices = slice_signal(outer_race_at6_48_0hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

(Imgs, img) = generate_images(slaices)
A = np.insert(A, 0, Imgs, axis=0)
# 1hp
slaices = slice_signal(outer_race_at6_48_1hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

(Imgs, img) = generate_images(slaices)
A = np.insert(A, 0, Imgs, axis=0)
# 2hp
slaices = slice_signal(outer_race_at6_48_2hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

(Imgs, img5) = generate_images(slaices)
A = np.insert(A, 0, Imgs, axis=0)
plt.axis('off')
plt.title('RE Centralizado, 2 hp')
plt.imshow(img, cmap=style)
plt.show()
# 3hp
slaices = slice_signal(outer_race_at6_48_3hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

(Imgs, img) = generate_images(slaices)
A = np.insert(A, 0, Imgs, axis=0)

# %% build images for outer race at 12oclock failure at 48ksps
# 0hp
slaices = slice_signal(outer_race_at12_48_0hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

(Imgs, img) = generate_images(slaices)
A = np.insert(A, 0, Imgs, axis=0)
# 1hp
slaices = slice_signal(outer_race_at12_48_1hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

(Imgs, img) = generate_images(slaices)
A = np.insert(A, 0, Imgs, axis=0)
# 2hp
slaices = slice_signal(outer_race_at12_48_2hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

(Imgs, img6) = generate_images(slaices)
A = np.insert(A, 0, Imgs, axis=0)
plt.axis('off')
plt.title('RE Oposto, 2 hp')
plt.imshow(img, cmap=style)
plt.show()
# 3hp
slaices = slice_signal(outer_race_at12_48_3hp_rs)
Sl = np.insert(Sl, 0, slaices, axis=0)

(Imgs, img) = generate_images(slaices)
A = np.insert(A, 0, Imgs, axis=0)

# %%
# fig, axs = plt.subplots(2, 3)

# fig.suptitle('Amplitude de vibração')
# fig.suptitle('Amplitude de vibração')
# axs[0, 0].imshow(img6, cmap=style)
# axs[0, 0].set_title('C1 - RE Oposto')
# axs[0, 0].axis('off')
# axs[0, 1].imshow(img5, cmap=style)
# axs[0, 1].set_title('C2 - RE Centralizado')
# axs[0, 1].axis('off')
# axs[0, 2].imshow(img4, cmap=style)
# axs[0, 2].set_title('C3 - RE Ortogonal')
# axs[0, 2].axis('off')
# axs[1, 0].imshow(img3, cmap=style)
# axs[1, 0].set_title('C4 - RI')
# axs[1, 0].axis('off')
# axs[1, 1].imshow(img2, cmap=style)
# axs[1, 1].set_title('C5 - Esfera')
# axs[1, 1].axis('off')
# axs[1, 2].imshow(img1, cmap=style)
# axs[1, 2].set_title('C6 - Normal')
# axs[1, 2].axis('off')
# axs[0].set(ylabel='amplitude')
# for ax in axs.flat:
#     ax.set(xlabel='amostra')
# for ax in axs.flat:
#     ax.label_outer()

# fig.tight_layout(pad=0)
# fig.set_figheight(6)
# fig.set_figwidth(7)
# fig.tight_layout(pad=1.5)
# fig.text(0.0, 0.5, 'amplitude', ha='center', va='center',
#          rotation='vertical')

# %% Reshape A
A = A.reshape(A.shape[0], img_w, img_h, 1)

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
X_train, X_test, y_train, y_test = train_test_split(A, label3,
                                                    test_size=0.3,
                                                    random_state=21)
Sl_train, Sl_test, y_train2, y_test2 = train_test_split(Sl, label3,
                                                        test_size=0.3,
                                                        random_state=21)
# X_train = X_train.astype('uint8')
# X_test = X_test.astype('uint8')

# %%
# patient early stopping
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
es = EarlyStopping(monitor='val_accuracy', mode='max')
# %%
for i in range(1, 11):
    # %% Separate classes, labels, train and test
    X_train, X_test, y_train, y_test = train_test_split(A, label3,
                                                        test_size=0.3)
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
    history = model1.fit(X_train, y_train, batch_size=10, nb_epoch=100)
                         # validation_data=(X_test, y_test))
    # %%
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    plt.title('model 1 accuracy, run #'+str(i))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='lower right')
    plt.show()

    # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model 1 loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

    # %% Results
    # Make inference
    # Predict and normalize predictions into 0s and 1s
    predictions = model1.predict(X_test)
    predictions2 = (predictions > 0.5)
    # Find accuracy of inference
    accuracy = accuracy_score(y_test, predictions2)
    precision = precision_score(y_test, predictions2, average='weighted')
    recall = recall_score(y_test, predictions2, average='weighted')
    f1 = f1_score(y_test, predictions2, average='weighted')
    print(accuracy)
    print(precision)
    print(recall)
    print(f1)
    accuracies_1.append(accuracy)
    precisions_1.append(precision)
    recalls_1.append(recall)
    f1s_1.append(f1)
    # calculate confusion matrix and print it
    matrix = confusion_matrix(y_test.argmax(axis=1),
                              predictions2.argmax(axis=1))
    print(matrix)
    confusion_matrices_1.append(matrix)

    # Use evaluate to test, just another way to do the same thing
    result = model1.evaluate(X_test, y_test)
    print(result)
    confusion_matrix(y_test.argmax(axis=1), predictions2.argmax(axis=1))
    ##########################################################################
    # %% Same as above for another, simpler CNN model
    # define as sequential
    model2 = models.Sequential()
    # add first convolutional layer
    model2.add(Conv2D(2, (2, 2), activation='relu',
                      input_shape=(img_w, img_h, 1)))
    # add first max pooling layer
    model2.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    # flatten befor MLP
    model2.add(Flatten())
    # add fully connected wih 8 neurons and relu activation
    model2.add(Dense(8, activation='relu'))
    # output six classes with softmax activtion
    model2.add(Dense(6, activation='softmax'))

    # print CNN info
    model2.summary()
    # compile CNN and define its functions
    model2.compile(loss='categorical_crossentropy', optimizer=Adam(),
                   metrics=['accuracy'])

    # %% Train CNN model2
    history = model2.fit(X_train, y_train, batch_size=10, nb_epoch=100)
                         # validation_data=(X_test, y_test))

    # %%
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    plt.title('model 2 accuracy, run #'+str(i))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='lower right')
    plt.show()

    # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model 2 loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

    # %% Export model to .h5 file
    # model2.save("./model2.h5")
    # print("Saved model to disk")

    # %% Results
    # Make inference
    # Predict and normalize predictions into 0s and 1s
    predictions = model2.predict(X_test)
    predictions = (predictions > 0.5)
    # Find accuracy of inference
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    print(accuracy)
    accuracies_2.append(accuracy)
    precisions_2.append(precision)
    recalls_2.append(recall)
    f1s_2.append(f1)
    # calculate confusion matrix and print it
    matrix = confusion_matrix(y_test.argmax(axis=1),
                              predictions.argmax(axis=1))
    print(matrix)
    confusion_matrices_2.append(matrix)

    # Use evaluate to test, just another way to do the same thing
    result = model2.evaluate(X_test, y_test)
    print(result)
    confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))

# %% Export model to .h5 file
model2.save("./b2ls_cnn_large.h5")
model1.save("./b2ls_cnn_small.h5")
print("Saved model to disk")

# %%

# %%
confusion_1_final = sum(confusion_matrices_1)
print("matriz confusão total 1")
print(confusion_1_final)

max_accuracy_1 = max(accuracies_1)
print("acurácia máxima 1", max_accuracy_1)
min_accuracy_1 = min(accuracies_1)
print("acurácia mínima 1", min_accuracy_1)
mean_accuracy_1 = mean(accuracies_1)
print("média acurácias 1", mean_accuracy_1)

max_precision_1 = max(precisions_1)
print("precisão máxima 1", max_precision_1)
min_precision_1 = min(precisions_1)
print("precisão mínima 1", min_precision_1)
mean_precision_1 = mean(precisions_1)
print("média precisão 1", mean_precision_1)

max_recall_1 = max(recalls_1)
print("recall máxima 1", max_recall_1)
min_recall_1 = min(recalls_1)
print("recall mínima 1", min_recall_1)
mean_recall_1 = mean(recalls_1)
print("média recall 1", mean_recall_1)

max_f1_1 = max(f1s_1)
print("f1 máxima 1", max_f1_1)
min_f1_1 = min(f1s_1)
print("f1 mínima 1", min_f1_1)
mean_f1_1 = mean(f1s_1)
print("média f1 1", mean_f1_1)
# %%
confusion_2_final = sum(confusion_matrices_2)
print("matriz confusão total 2")
print(confusion_2_final)
# j2 = [i for i in j if i >= 5]

accuracies_2_clean = [i for i in accuracies_2 if i >= .9]
max_2 = max(accuracies_2)
print("acurácia máxima 2", max_2)
min_2 = min(accuracies_2)
print("acurácia mínima 2", min_2)
mean_2 = mean(accuracies_2)
print("média acurácias 2", mean_2)

max_precision_2 = max(precisions_2)
print("precisão máxima 2", max_precision_2)
min_precision_2 = min(precisions_2)
print("precisão mínima 2", min_precision_2)
mean_precision_2 = mean(precisions_2)
print("média precisão 2", mean_precision_2)

max_recall_2 = max(recalls_2)
print("recall máxima 2", max_recall_2)
min_recall_2 = min(recalls_2)
print("recall mínima 2", min_recall_2)
mean_recall_2 = mean(recalls_2)
print("média recall 2", mean_recall_2)

max_f1_2 = max(f1s_2)
print("f1 máxima 2", max_f1_2)
min_f1_2 = min(f1s_2)
print("f1 mínima 2", min_f1_2)
mean_f1_2 = mean(f1s_2)
print("média f1 2", mean_f1_2)

