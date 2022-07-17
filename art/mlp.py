#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 15:44:44 2022

@author: jose
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
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
    
    # mlp
    model = MLPClassifier(hidden_layer_sizes = (64, 32, ), max_iter = 1000)
    model.fit(X_train, y_train)
    
    # eval
    acc = roc_auc_score(y_test, model.predict(X_test))
    log.append(acc)

# report
print(np.mean(log), np.std(log))
