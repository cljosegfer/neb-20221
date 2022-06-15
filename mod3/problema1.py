#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 14:34:52 2022

@author: jose
"""

import numpy as np
from anfis import anfis
from nfn import nfn
import matplotlib.pyplot as plt

# input
N = 100
X_test = np.linspace(-1.95, 1.95, N).reshape(-1, 1)
y_test = X_test ** 2

# shuffle
X_train = np.random.uniform(low = -2, high = 2, size = 9*N).reshape(-1, 1)
y_train = X_train ** 2

# anfis
n = 2
model = anfis(n = n, m = X_train.shape[1])
model.fit(X_train, y_train, alpha = 0.1, max_epochs = 10)

# eval
yhat = model.predict(X_test).reshape(-1, 1)
mse = model.mse(X_test, y_test)
epm = (np.abs(y_test - yhat) / yhat).mean()
print('mse: {}, epm: {}'.format(mse, epm))

# plot
plt.figure()
xx, yy = zip(*sorted(zip(X_test, yhat)))
plt.plot(xx, yy)
xx, yy = zip(*sorted(zip(X_test, y_test)))
plt.plot(xx, yy)

# log
plt.figure()
plt.plot(model.log)

# nfn
model = nfn(N = 100)
model.fit(X_train, y_train, alpha = 0.1, max_epochs = 100)

# eval
yhat = model.predict(X_test).reshape(-1, 1)
mse = model.mse(X_test, y_test)
epm = (np.abs(y_test - yhat) / yhat).mean()
print('mse: {}, epm: {}'.format(mse, epm))

# plot
plt.figure()
xx, yy = zip(*sorted(zip(X_test, yhat)))
plt.plot(xx, yy)
xx, yy = zip(*sorted(zip(X_test, y_test)))
plt.plot(xx, yy)

# log
plt.figure()
plt.plot(model.log)
