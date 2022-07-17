#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 16:33:31 2022

@author: jose
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
from anfis import anfis
from sklearn.metrics import roc_auc_score

# read
data_path = 'data'
ds = 'comvoi-en.csv'
data = pd.read_csv('{}/{}'.format(data_path, ds))

X = data.drop(columns = 'accent').values
Y = data.accent.values

# kfold
kf = KFold(n_splits = 10)

# one vs all
accents = np.unique(Y)
accent = accents[-1]
y = (Y == accent)*1

# model
log = []
for train_index, test_index in tqdm(kf.split(X)):
    # train / test
    X_train = X[train_index]
    y_train = y[train_index]
    
    X_test = X[test_index]
    y_test = y[test_index]
    
    # anfis
    n = 15
    model = anfis(n = n, m = X_train.shape[1])
    model.fit(X_train, y_train, max_epochs = 10, alpha = 0.01)
    
    # eval
    acc = roc_auc_score(y_test, model.predict(X_test))
    log.append(acc)

# report
print(np.mean(log), np.std(log))
