#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 16:08:33 2022

@author: jose
"""

import scipy.io as sio
import numpy as np
import pandas as pd
from tqdm import tqdm

k = 10
ds_path = 'data'
ds = 'comvoi-en'
gg_path = 'gg'

accents = ['us', 'england', 'indian', 'canada', 'australia']

accent = 'australia'
idx = 0

# read
data = sio.loadmat('{}/{}/{}_fold_{}.mat'.format(ds_path, ds, ds, idx))

# train/test
X_train = data['X_train']
y_train = data['y_train']

X_test = data['X_test']
y_test = data['y_test']

# accent
y_train = 1*(y_train == accent).reshape(-1)
y_test = 1*(y_test == accent).reshape(-1)

# gg
path = '{}/{}/{}_fold_{}.csv'.format(ds_path, gg_path, ds, idx)
gg = pd.read_csv(path).to_numpy()

# qualidade
scores = []
for i, row in enumerate(gg):
    vizinhos = np.where(row == 1)[0]
    
    degree = len(vizinhos)
    opposite = 0
    for vizinho in vizinhos:
        opposite += np.abs(y_train[0] - y_train[vizinho])
    q = 1 - opposite / degree
    scores.append(q)

# np.percentile(scores, 50)
border = np.where(np.array(scores) < 0.954)[0]
print(border.size)

# under sampling
X_train = X_train[border, :]
y_train = y_train[border]

eta = np.count_nonzero(y_train) / y_train.size
print(eta)
