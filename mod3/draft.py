#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 12:16:25 2022

@author: jose
"""

import numpy as np
from anfis import anfis
import matplotlib.pyplot as plt

# entrada
N = 50
X = np.linspace(-2, 2, N).reshape(-1, 1)
y = X ** 2

# shuffle
random = np.random.permutation(X.shape[0])
X = X[random]
y = y[random]

# anfis
n = 2
model = anfis(n = 2, X = X)
model.fit(X, y)
yhat = model.predict(X)

# test
xx, yy = zip(*sorted(zip(X, yhat)))
plt.plot(xx, yy)
xx, yy = zip(*sorted(zip(X, y)))
plt.plot(xx, yy)
