#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 15:05:39 2022

@author: jose
"""

import numpy as np
from anfis import anfis
import matplotlib.pyplot as plt

def g(x):
    num = x[0] * x[1] * x[2] * x[4] * (x[2] - 1) + x[3]
    den = 1 + x[2]**2 + x[3]**2
    return num / den

# input
N = 1000
k = np.arange(N)
u = np.sin(2*np.pi * k / 250)
u[k>500] = 0.8 * u[k>500] + 0.2 * np.sin(2*np.pi * k[k>500] / 25)
# plt.figure()
# plt.plot(u)

y = np.zeros(N)
X = np.zeros(shape = (N - 6, 5))
for k in range(2, N - 1):
    x = np.array([y[k], y[k-1], y[k-2], u[k], u[k-1]])
    y[k + 1] = g(x)
    if k > 4:
        X[k - 5] = x
y = np.delete(y, obj = [0, 1, 2, 3, 4, -1], axis = 0)
# plt.figure()
# plt.plot(y)

# N_train = 5000
# u_train = np.random.uniform(low = -1, high = 1, size = N_train)
# y_train = np.zeros(N_train)
# X_train = np.zeros(shape = (N_train - 6, 5))
# for k in range(2, N_train - 1):
#     x = np.array([y_train[k], y_train[k-1], y_train[k-2], u_train[k], u_train[k-1]])
#     y_train[k + 1] = g(x)
#     if k > 4:
#         X_train[k - 5] = x
# plt.figure()
# plt.plot(y_train)

# kfold
k = 10
size = len(X)
index = list(range(size))
np.random.shuffle(index)
step = round(size / k)
kfolds = [index[i:i+step] for i in range(0, size, step)]

k = 0
kfold = kfolds[k]
fold = np.ones(size, bool)
fold[kfold] = False

X_test = X[np.invert(fold), :]
X_train = X[fold, :]
y_test = y[np.invert(fold)]
y_train = y[fold]

# anfis
n = 15
model = anfis(n = n, m = X_train.shape[1])
model.fit(X_train, y_train, max_epochs = 10, alpha = 0.01)

# report
yhat = model.predict(X).reshape(-1, 1)
mse = model.mse(X_test, y_test)
epm = (np.abs(y - yhat) / yhat).mean()
print('mse: {}, epm: {}'.format(mse, epm))

# plot
plt.figure()
plt.plot(y)
plt.plot(yhat)

# log
plt.figure()
plt.plot(model.log)
