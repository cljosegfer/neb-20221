#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 15:41:23 2022

@author: jose
"""

import scipy.io as sio
import numpy as np
import pandas as pd
from tqdm import tqdm

k = 10
path = 'data'
ds = 'comvoi-en'

accents = ['us', 'england', 'indian', 'canada', 'australia']

accent = 'australia'
idx = 0

# read
data = sio.loadmat('{}/{}/{}_fold_{}.mat'.format(path, ds, ds, idx))

# train/test
X_train = data['X_train']
y_train = data['y_train']

X_test = data['X_test']
y_test = data['y_test']

# accent
y_train = 1*(y_train == accent).reshape(-1)
y_test = 1*(y_test == accent).reshape(-1)

# over sampling
delta = y_train.size - 2 * np.count_nonzero(y_train)
cluster = np.where(y_train == 1)[0]
sampling = np.random.choice(a = cluster, size = delta)
add = X_train[sampling, :]

X_train = np.concatenate([X_train, X_train[sampling, :]])
y_train = np.concatenate([y_train, y_train[sampling]])

eta = np.count_nonzero(y_train) / y_train.size
