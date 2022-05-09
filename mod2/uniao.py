#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 13:40:55 2022

@author: jose
"""

import numpy as np

def maximo(a, b):
    return np.maximum(a, b)

def probabilistica(a, b):
    return a + b - a * b

def limitada(a, b):
    return np.minimum(1, a + b)

def drastica(a, b):
    mu = np.ones(shape = a.shape)
    # for i in range(len(a)):
    #     if a[i] == 0:
    #         mu[i] = b[i]
    #     elif b[i] == 0:
    #         mu[i] = a[i]
    #     else:
    #         continue
    azero = (a == 0)    
    bzero = (b == 0)
    mu[azero] = b[azero]
    mu[bzero] = a[bzero]
    
    return mu
