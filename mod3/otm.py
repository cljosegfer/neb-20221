#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 14:24:22 2022

@author: jose
"""

import numpy as np

def grad(target, x0, delta = 0.001):
    grad = np.zeros(x0.size)
    for dim in range(x0.size):
        d = np.zeros(x0.size)
        d[dim] = delta
        x = x0 + d
        grad[dim] = (target(x) - target(x0)) / delta
    return grad

def metodo_gradiente(x0, MAX, target, alpha = 0.1, delta = 0.001):
    x = x0
    for epoca in range(MAX):
        g = grad(target, x, delta)
        # alpha = scipy.optimize.golden(lambda alpha: target(x0 - alpha * g))
        x = x - alpha * g
    return x, target(x)

def fobj(x):
    f = x[0]**2 + x[1]**2
    return f

x0 = np.array([2, 0])
delta = grad(fobj, x0)
otimo, minimo = metodo_gradiente(x0, 50, fobj)
