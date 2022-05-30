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
plt.figure()
plt.plot(u)

y = np.zeros(N)
X_test = np.zeros(shape = (N - 6, 5))
for k in range(2, N - 1):
    x = np.array([y[k], y[k-1], y[k-2], u[k], u[k-1]])
    y[k + 1] = g(x)
    if k > 4:
        X_test[k - 5] = x
plt.figure()
plt.plot(y)




