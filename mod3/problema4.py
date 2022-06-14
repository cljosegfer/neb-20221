#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 22:44:57 2022

@author: jose
"""

import numpy as np
from anfis import anfis
import matplotlib.pyplot as plt

def mackey_glass(N = 1000):
    b = 0.1
    c = 0.2
    tau = 17

    y = [0.9697, 0.9699, 0.9794, 1.0003, 1.0319, 1.0703, 1.1076, 1.1352, 1.1485,
     1.1482, 1.1383, 1.1234, 1.1072, 1.0928, 1.0820, 1.0756, 1.0739, 1.0759]

    for n in range(17,N+99):
        y.append(y[n] - b*y[n] + c*y[n-tau]/(1+y[n-tau]**10))
    y = y[100:]
    return np.array(y)

# input
N = 1000
data = mackey_glass(N)
y = np.zeros(N - 18 - 6)
X = np.zeros(shape = (N - 18 - 6, 4))
i = 0
for t in range(18, N - 6):
    x = np.array([data[t - 18], data[t - 12], data[t - 6], data[t]])
    X[i] = x
    y[i] = data[t + 6]
    i += 1

# train test
X_train, X_test = np.split(X, 2)
y_train, y_test = np.split(y, 2)

# anfis
n = 16
model = anfis(n = n, m = X_train.shape[1])
model.fit(X_train, y_train, max_epochs = 100, alpha = 0.01)

# report
yhat = model.predict(X_test).reshape(-1, 1)
mse = model.mse(X_test, y_test)
epm = (np.abs(y_test - yhat) / yhat).mean()
print('mse: {}, epm: {}'.format(mse, epm))

# plot
plt.plot(y_test)
plt.plot(yhat)

# log
plt.figure()
plt.plot(model.log)
