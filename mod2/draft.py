#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 13:45:51 2022

@author: jose
"""

import numpy as np

def composition(U, V):
    #max-min
    return np.amax(np.minimum(U[:, :, None], V[None, :, :]), axis = 1)

# R1 = np.array([[0.6, 0.5, 0.3], 
#               [0.1, 0.3, 0.7], 
#               [0.4, 0.5, 0.9]])
# R2 = np.array([[0.1, 0.7, 0.9], 
#               [0.25, 0.65, 0.1], 
#               [0.21, 0.72, 0.0]])

# # composicao
# mu = composition(R1, R2)

A1 = np.array([0.6, 0.1, 0.4])
A2 = np.array([0.8, 0.3, 0.2])
B1 = np.array([0.3, 0.8])
B2 = np.array([0.7, 0.7])
Alinha = np.array([0.5, 0.4, 0.8])

# primeiro i = 1










