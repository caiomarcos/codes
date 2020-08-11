# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 00:47:30 2020

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
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras import models
from keras.optimizers import Adam
from statistics import mean
import pickle


# %% Defining some constants to be used throughout
# number os data points per set
data_points = 120000
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
# overlap when making samples
div = 1
step = N//div
# # images in each class
samples_per_class = ((data_points)//(N-(N-step)))-(div-1)
# bitmap style
style = "viridis_r"
# path to files
path = 'C:/Users/caiom/Documents/propecaut2019/disciplinas/topicos-wavelet/CWRU-DE/'


# %% normal baseline at 48ksps
# 0hp (5 seconds only)
# import matlab file using scipy
normal_48_0hp = scipy.io.loadmat(path+'48ksps/0hp/97.mat')
# get only the acc data points
normal_48_0hp = normal_48_0hp['X097_DE_time']
# undersample by 4 to get 12ksps (resampled)
normal_48_0hp_rs = normal_48_0hp[::4]
# resample to 20ksps
normal_48_0hp_rs = scipy.signal.resample(normal_48_0hp, data_points)

# 1hp
# import matlab file using scipy
normal_48_1hp = scipy.io.loadmat(path+'48ksps/1hp/98.mat')
# get only the acc data points
normal_48_1hp = normal_48_1hp['X098_DE_time']
# undersample by 4 to get 12ksps (resampled)
normal_48_1hp_rs = normal_48_1hp[::4]
# resample to 20ksps
normal_48_1hp_rs = scipy.signal.resample(normal_48_1hp, data_points)

# 2hp
# import matlab file using scipy
normal_48_2hp = scipy.io.loadmat(path+'48ksps/2hp/99.mat')
# get only the acc data points
normal_48_2hp = normal_48_2hp['X099_DE_time']
# undersample by 4 to get 12ksps (resampled)
normal_48_2hp_rs = normal_48_2hp[::4]
# resample to 20ksps
normal_48_2hp_rs = scipy.signal.resample(normal_48_2hp, data_points)

# 3hp
# import matlab file using scipy
normal_48_3hp = scipy.io.loadmat(path+'48ksps/3hp/100.mat')
# get only the acc data points
normal_48_3hp = normal_48_3hp['X100_DE_time']
# undersample by 4 to get 12ksps (resampled)
normal_48_3hp_rs = normal_48_3hp[::4]
# resample to 20ksps
normal_48_3hp_rs = scipy.signal.resample(normal_48_3hp, data_points)

# %% rolling element (ball) at 48ksps
# 0hp (5 seconds)
# import matlab file using scipy
ball_12_0hp = scipy.io.loadmat(path+'12ksps/0hp/222.mat')
# get only the acc data points
ball_12_0hp = ball_12_0hp['X222_DE_time']
# undersample to 12ksps
ball_12_0hp_rs = ball_12_0hp[::4]
# resample to 20ksps
ball_12_0hp_rs = scipy.signal.resample(ball_12_0hp, data_points)

# 1hp
# import matlab file using scipy
ball_12_1hp = scipy.io.loadmat(path+'12ksps/1hp/223.mat')
# get only the acc data points
ball_12_1hp = ball_12_1hp['X223_DE_time']
# undersample to 12ksps
ball_12_1hp_rs = ball_12_1hp[::4]
# resample to 20ksps
ball_12_1hp_rs = scipy.signal.resample(ball_12_1hp, data_points)

# 2hp
# import matlab file using scipy
ball_12_2hp = scipy.io.loadmat(path+'12ksps/2hp/224.mat')
# get only the acc data points
ball_12_2hp = ball_12_2hp['X224_DE_time']
# undersample to 12ksps
ball_12_2hp_rs = ball_12_2hp[::4]
# resample to 20ksps
ball_12_2hp_rs = scipy.signal.resample(ball_12_2hp, data_points)

# 3hp
# import matlab file using scipy
ball_12_3hp = scipy.io.loadmat(path+'12ksps/3hp/225.mat')
# get only the acc data points
ball_12_3hp = ball_12_3hp['X225_DE_time']
# undersample to 12ksps
ball_12_3hp_rs = ball_12_1hp[::4]
# resample to 20ksps
ball_12_3hp_rs = scipy.signal.resample(ball_12_3hp, data_points)

# %% inner race at 48ksps
# 0hp (5 seconds)
# import matlab file using scipy
inner_race_12_0hp = scipy.io.loadmat(path+'12ksps/0hp/209.mat')
# get only the acc data points
inner_race_12_0hp = inner_race_12_0hp['X209_DE_time']
# undersample to 12ksps
inner_race_12_0hp_rs = inner_race_12_0hp[::4]
# resample to 20ksps
inner_race_12_0hp_rs = scipy.signal.resample(inner_race_12_0hp, data_points)

# 1hp
# import matlab file using scipy
inner_race_12_1hp = scipy.io.loadmat(path+'12ksps/1hp/210.mat')
# get only the acc data points
inner_race_12_1hp = inner_race_12_1hp['X210_DE_time']
# undersample to 12ksps
inner_race_12_1hp_rs = inner_race_12_1hp[::4]
# resample to 20ksps
inner_race_12_1hp_rs = scipy.signal.resample(inner_race_12_1hp, data_points)

# 2hp
# import matlab file using scipy
inner_race_12_2hp = scipy.io.loadmat(path+'12ksps/2hp/211.mat')
# get only the acc data points
inner_race_12_2hp = inner_race_12_2hp['X211_DE_time']
# undersample to 12ksps
inner_race_12_2hp_rs = inner_race_12_2hp[::4]
# resample to 20ksps
inner_race_12_2hp_rs = scipy.signal.resample(inner_race_12_2hp, data_points)

# 3hp
# import matlab file using scipy
inner_race_12_3hp = scipy.io.loadmat(path+'12ksps/3hp/212.mat')
# get only the acc data points
inner_race_12_3hp = inner_race_12_3hp['X212_DE_time']
# undersample to 12ksps
inner_race_12_3hp_rs = inner_race_12_3hp[::4]
# resample to 20ksps
inner_race_12_3hp_rs = scipy.signal.resample(inner_race_12_3hp, data_points)

# %% outer race at 3oclock at 48ksps
# 0hp
# import matlab file using scipy
outer_race_at3_12_0hp = scipy.io.loadmat(path+'12ksps/0hp/246.mat')
# get only the acc data points
outer_race_at3_12_0hp = outer_race_at3_12_0hp['X246_DE_time']
# undersample to 12ksps
outer_race_at3_12_0hp_rs = outer_race_at3_12_0hp[::4]
# resample to 20ksps
outer_race_at3_12_0hp_rs = scipy.signal.resample(outer_race_at3_12_0hp,
                                                 data_points)

# 1hp
# import matlab file using scipy
outer_race_at3_12_1hp = scipy.io.loadmat(path+'12ksps/1hp/247.mat')
# get only the acc data points
outer_race_at3_12_1hp = outer_race_at3_12_1hp['X247_DE_time']
# undersample to 12ksps
outer_race_at3_12_1hp_rs = outer_race_at3_12_1hp[::4]
# resample to 20ksps
outer_race_at3_12_1hp_rs = scipy.signal.resample(outer_race_at3_12_1hp,
                                                 data_points)

# 2hp
# import matlab file using scipy
outer_race_at3_12_2hp = scipy.io.loadmat(path+'12ksps/2hp/248.mat')
# get only the acc data points
outer_race_at3_12_2hp = outer_race_at3_12_2hp['X248_DE_time']
# undersample to 12ksps
outer_race_at3_12_2hp_rs = outer_race_at3_12_2hp[::4]
# resample to 20ksps
outer_race_at3_12_2hp_rs = scipy.signal.resample(outer_race_at3_12_2hp,
                                                 data_points)

# 3hp
# import matlab file using scipy
outer_race_at3_12_3hp = scipy.io.loadmat(path+'12ksps/3hp/249.mat')
# get only the acc data points
outer_race_at3_12_3hp = outer_race_at3_12_3hp['X249_DE_time']
# undersample to 12ksps
outer_race_at3_12_3hp_rs = outer_race_at3_12_3hp[::4]
# resample to 20ksps
outer_race_at3_12_3hp_rs = scipy.signal.resample(outer_race_at3_12_3hp,
                                                 data_points)

# %% outer race at 6oclock at 48ksps
# 0hp
# import matlab file using scipy
outer_race_at6_12_0hp = scipy.io.loadmat(path+'12ksps/0hp/234.mat')
# get only the acc data points
outer_race_at6_12_0hp = outer_race_at6_12_0hp['X234_DE_time']
# undersample to 12ksps
outer_race_at6_12_0hp_rs = outer_race_at6_12_0hp[::4]
# resample to 20ksps
outer_race_at6_12_0hp_rs = scipy.signal.resample(outer_race_at6_12_0hp,
                                                 data_points)

# 1hp
# import matlab file using scipy
outer_race_at6_12_1hp = scipy.io.loadmat(path+'12ksps/1hp/235.mat')
# get only the acc data points
outer_race_at6_12_1hp = outer_race_at6_12_1hp['X235_DE_time']
# undersample to 12ksps
outer_race_at6_12_1hp_rs = outer_race_at6_12_1hp[::4]
# resample to 20ksps
outer_race_at6_12_1hp_rs = scipy.signal.resample(outer_race_at6_12_1hp,
                                                 data_points)

# 2hp
# import matlab file using scipy
outer_race_at6_12_2hp = scipy.io.loadmat(path+'12ksps/2hp/236.mat')
# get only the acc data points
outer_race_at6_12_2hp = outer_race_at6_12_2hp['X236_DE_time']
# undersample to 12ksps
outer_race_at6_12_2hp_rs = outer_race_at6_12_2hp[::4]
# resample to 20ksps
outer_race_at6_12_2hp_rs = scipy.signal.resample(outer_race_at6_12_2hp,
                                                 data_points)

# 3hp
# import matlab file using scipy
outer_race_at6_12_3hp = scipy.io.loadmat(path+'12ksps/3hp/237.mat')
# get only the acc data points
outer_race_at6_12_3hp = outer_race_at6_12_3hp['X237_DE_time']
# undersample to 12ksps
outer_race_at6_12_3hp_rs = outer_race_at6_12_3hp[::4]
# resample to 20ksps
outer_race_at6_12_3hp_rs = scipy.signal.resample(outer_race_at6_12_3hp,
                                                 data_points)

# %% outer race at 12oclock at 48ksps
# 0hp
# import matlab file using scipy
outer_race_at12_12_0hp = scipy.io.loadmat(path+'12ksps/0hp/258.mat')
# get only the acc data points
outer_race_at12_12_0hp = outer_race_at12_12_0hp['X258_DE_time']
# undersample to 12ksps
outer_race_at12_12_0hp_rs = outer_race_at12_12_0hp[::4]
# resample to 20ksps
outer_race_at12_12_0hp_rs = scipy.signal.resample(outer_race_at12_12_0hp,
                                                  data_points)

# 1hp
# import matlab file using scipy
outer_race_at12_12_1hp = scipy.io.loadmat(path+'12ksps/1hp/259.mat')
# get only the acc data points
outer_race_at12_12_1hp = outer_race_at12_12_1hp['X259_DE_time']
# undersample to 12ksps
outer_race_at12_12_1hp_rs = outer_race_at12_12_1hp[::4]
# resample to 20ksps
outer_race_at12_12_1hp_rs = scipy.signal.resample(outer_race_at12_12_1hp,
                                                  data_points)

# 2hp
# import matlab file using scipy
outer_race_at12_12_2hp = scipy.io.loadmat(path+'12ksps/2hp/260.mat')
# get only the acc data points
outer_race_at12_12_2hp = outer_race_at12_12_2hp['X260_DE_time']
# undersample to 12ksps
outer_race_at12_12_2hp_rs = outer_race_at12_12_2hp[::4]
# resample to 20ksps
outer_race_at12_12_2hp_rs = scipy.signal.resample(outer_race_at12_12_2hp,
                                                  data_points)

# 3hp
# import matlab file using scipy
outer_race_at12_12_3hp = scipy.io.loadmat(path+'12ksps/3hp/261.mat')
# get only the acc data points
outer_race_at12_12_3hp = outer_race_at12_12_3hp['X261_DE_time']
# undersample to 12ksps
outer_race_at12_12_3hp_rs = outer_race_at12_12_3hp[::4]
# resample to 20ksps
outer_race_at12_12_3hp_rs = scipy.signal.resample(outer_race_at12_12_3hp,
                                                  data_points)
# %%
# fig, axs = plt.subplots(6)

# # fig.suptitle('Amplitude de vibração')
# # fig.suptitle('Amplitude de vibração')
# axs[0].plot(outer_race_at12_12_2hp_rs[0:6000])
# axs[0].set_title('RE Oposto')
# axs[1].plot(outer_race_at6_12_2hp[0:6000])
# axs[1].set_title('RE Centralizado')
# axs[2].plot(outer_race_at3_12_2hp_rs[0:6000])
# axs[2].set_title('RE Ortogonal')
# axs[3].plot(inner_race_12_2hp_rs[0:6000])
# axs[3].set_title('RI')
# axs[4].plot(ball_12_2hp_rs[0:6000])
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
# plt.plot(outer_race_at12_12_2hp_rs[0:120000])
# plt.title('Sinal original - RE Oposto, 2 hp')
# plt.ylabel('amplitude')
# plt.xlabel('amostras')
# plt.show()

# plt.figure(figsize=(6, 2))
# plt.plot(outer_race_at12_12_2hp_rs[0:1568])
# plt.title('Sinal amostrado - RE Oposto, 2 hp')
# plt.ylabel('amplitude')
# plt.xlabel('amostras')
# plt.show()

# plt.plot(normal_48_2hp_rs[0:6000])
# plt.title('Normal')
# plt.ylabel('amplitude')
# plt.xlabel('amostras')
# plt.show()

# plt.plot(inner_race_12_2hp_rs[0:6000])
# plt.title('RI')
# plt.ylabel('amplitude')
# plt.xlabel('amostras')
# plt.show()

# %% build final data set for each class using both 48ksps and 12ksps
# (resampled accordingly)
# normal = np.append(normal_48, normal_12_rs)
# inner_race = np.append(inner_race_48, inner_race_12)
# ball = np.append(ball_48, ball_12)
# outer_race_at3 = np.append(outer_race_at3_48, outer_race_at3_12)
# outer_race_at6 = np.append(outer_race_at6_48, outer_race_at6_12)
# outer_race_at12 = np.append(outer_race_at12_48, outer_race_at12_12)

# %% some statistics
df_12_rs = pd.DataFrame(inner_race_12_2hp_rs)
df_12 = pd.DataFrame(inner_race_12_2hp)
avg_12_rs = sum(inner_race_12_2hp_rs)/len(inner_race_12_2hp_rs)
avg_12 = sum(inner_race_12_2hp)/len(inner_race_12_2hp)
rms_12_rs = np.sqrt(np.mean(inner_race_12_2hp_rs**2))
rms_48 = np.sqrt(np.mean(inner_race_12_2hp**2))

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
# 1hp
(Imgs, img) = generate_images(normal_48_0hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)
# 1hp
(Imgs, img) = generate_images(normal_48_1hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)
# 2hp
(Imgs, img1) = generate_images(normal_48_2hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)
plt.axis('off')
plt.title('Normal, 2 hp')
plt.imshow(img, cmap=style)
plt.show()
# 3hp
(Imgs, img) = generate_images(normal_48_3hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)

# %% build images for rolling element (ball) failure at 48ksps
# 0hp
(Imgs, img) = generate_images(ball_12_0hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)
# 1hp
(Imgs, img) = generate_images(ball_12_1hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)
# 2hp
(Imgs, img2) = generate_images(ball_12_2hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)
plt.axis('off')
plt.title('Esfera, 2 hp')
plt.imshow(img, cmap=style)
plt.show()
# 3hp
(Imgs, img) = generate_images(ball_12_3hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)

# %% build images for inner race failure at 48ksps
# 0hp
(Imgs, img) = generate_images(inner_race_12_0hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)
# 1hp
(Imgs, img) = generate_images(inner_race_12_1hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)
# 2hp
(Imgs, img3) = generate_images(inner_race_12_2hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)
plt.axis('off')
plt.title('RI, 2 hp')
plt.imshow(img, cmap=style)
plt.show()
# 3hp
(Imgs, img) = generate_images(inner_race_12_3hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)

# %% build images for outer race at 3oclock failure at 48ksps
# 0hp
(Imgs, img) = generate_images(outer_race_at3_12_0hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)
# 1hp
(Imgs, img) = generate_images(outer_race_at3_12_1hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)
# 2hp
(Imgs, img4) = generate_images(outer_race_at3_12_2hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)
plt.axis('off')
plt.title('RE Ortogonal, 2 hp')
plt.imshow(img, cmap=style)
plt.show()
# 3hp
(Imgs, img) = generate_images(outer_race_at3_12_3hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)

# %% build images for outer race at 6oclock failure at 48ksps
# 0hp
(Imgs, img) = generate_images(outer_race_at6_12_0hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)
# 1hp
(Imgs, img) = generate_images(outer_race_at6_12_1hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)
# 2hp
(Imgs, img5) = generate_images(outer_race_at6_12_2hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)
plt.axis('off')
plt.title('RE Centralizado, 2 hp')
plt.imshow(img, cmap=style)
plt.show()
# 3hp
(Imgs, img) = generate_images(outer_race_at6_12_3hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)

# %% build images for outer race at 12oclock failure at 48ksps
# 0hp
(Imgs, img) = generate_images(outer_race_at12_12_0hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)
# 1hp
(Imgs, img) = generate_images(outer_race_at12_12_1hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)
# 2hp
(Imgs, img6) = generate_images(outer_race_at12_12_2hp_rs, step)
A = np.insert(A, 0, Imgs, axis=0)
plt.axis('off')
# plt.title('RE Oposto, 2 hp')
plt.imshow(img, cmap=style)
plt.show()
# 3hp
(Imgs, img) = generate_images(outer_race_at12_12_3hp_rs, step)
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
model2.save("./model2_sizetest.h5")
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
# %%
with open('objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([confusion_matrices_1, confusion_matrices_2, accuracies_1,
                 precisions_1, recalls_1, f1s_1, accuracies_2, precisions_2,
                 recalls_2, f1s_2], f)