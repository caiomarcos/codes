# -*- coding: utf-8 -*-
"""
Created on Thu May 28 19:35:19 2020
wavelet
@author: caiom
"""

# %%
# Importing modules
import numpy as np
import pandas as pd
from keras.utils import np_utils
import scipy.io
import scipy.signal
import math
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

import pywt
import pywt.data
from pywt import wavedec
from pywt import dwt_max_level

# %% Defining some constants to be used throughout
# number os data points per set
data_points = 200704
# image width
img_w = 28
# image length
img_h = 28
# matrix used to hold final images
A = np.zeros((0, img_w, img_h))
# image length when unidimensional
img_length = img_w*img_h
# number of data points used to build image
N = img_length*2
# images in each class
samples_per_class = (data_points)//N
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

# %%
w = pywt.Wavelet('db3')
print(w)
print(w.dec_hi)
print(w.rec_len)

ball_48_2hp_rs = ball_48_2hp_rs.reshape(data_points,)

maxLvl = pywt.dwt_max_level(200704, w)
print('max lvl')
print(maxLvl)
coeffs = wavedec(ball_48_2hp_rs, 'db3', level=3)

# %%
w = pywt.Wavelet('sym4')
(phi, psi, x) = w.wavefun(level=5)
print(w.dec_hi)
print(w.rec_len)

# import matlab file using scipy
wheat = scipy.io.loadmat('./wheat.mat')
# get only the acc data points
ball_48_1hp = ball_48_1hp['X227_DE_time']