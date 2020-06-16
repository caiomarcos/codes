# -*- coding: utf-8 -*-
"""
Created on Thu May 28 21:38:34 2020

wavelet examples using WHEAT dataset

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

import pywt
import pywt.data
from pywt import wavedec
from pywt import dwt_max_level

# %%
# import matlab file using scipy
wheat = scipy.io.loadmat('./wheat.mat')
Xwave = np.zeros((0, 0))

X = wheat['WHEAT_SPECTRA']
(nVar, nSamp) = (len(X[0]), len(X))

wavelet = pywt.Wavelet('sym2')
Lmax = pywt.dwt_max_level(nVar, wavelet)

wavelets = ['db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10',
            'coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'sym4', 'sym5',
            'sym6', 'sym7', 'sym8', 'sym9', 'sym10']
Lmaxs = []
for wave in wavelets:
    Lmaxs.append(pywt.dwt_max_level(nVar, wave))

Xwave = np.zeros((nSamp, 711))

# for each line in the dataset
for i in range(0, nSamp):
    # coeffs is list of arrays with coefficients cA5 and cD5 to cD1
    coeffs = wavedec(X[i, :], wavelet, level=4)
    # flatten coefficients into one line array
    C = pywt.ravel_coeffs(coeffs)
    C = C[0]
    # add coefficients to matrix of wavelet coefficients
    Xwave[i, 0:len(C)] = C
