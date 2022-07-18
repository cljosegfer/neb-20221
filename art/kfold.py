#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 16:51:39 2022

@author: jose
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from scipy.io import savemat

# read
data_path = 'data'
ds = 'comvoi-en.csv'
data = pd.read_csv('{}/{}'.format(data_path, ds))

X = data.drop(columns = 'accent').values
y = data.accent.values

# kfold
kf = KFold(n_splits = 10, shuffle = True)
for idx, [train_index, test_index] in enumerate(kf.split(X)):
    # train / test
    X_train = X[train_index]
    y_train = y[train_index]
    
    X_test = X[test_index]
    y_test = y[test_index]
    
    # dict
    mdic = {'X_train': X_train, 'y_train': y_train, 
            'X_test': X_test, 'y_test': y_test}
    
    # export
    path = '{}/{}/{}_fold_{}.mat'.format(data_path, ds[:-4], ds[:-4], idx)
    savemat(path, mdic)
