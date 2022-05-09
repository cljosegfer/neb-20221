#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 15:44:19 2022

@author: jose
"""

import numpy as np
from pertinencia import trimf
import matplotlib.pyplot as plt

# param
li = -2
ls = 2
delta = 0.1
N = int((ls - li) / delta + 1)

# pertinencia
x = np.linspace(li, ls, N)
y = np.linspace(0, 4, N)
A1 = trimf(x, a = -3, b = -2, c = -1)
A2 = trimf(x, a = -2, b = -1, c = 0)
A3 = trimf(x, a = -1, b = 0, c = 1)
A4 = trimf(x, a = 0, b = 1, c = 2)
A5 = trimf(x, a = 1, b = 2, c = 3)
B1 = trimf(y, a = -1, b = 0, c = 1)
B2 = trimf(y, a = 0, b = 1, c = 2)
B3 = trimf(y, a = 1, b = 4, c = 0)

# plot pertinencia
fig, axs = plt.subplots(2, figsize = (8, 10))
axs[0].plot(x, A1)
axs[0].plot(x, A2)
axs[0].plot(x, A3)
axs[0].plot(x, A4)
axs[0].plot(x, A5)
axs[0].set_xlabel('A')
axs[1].plot(x, B1)
axs[1].plot(x, B2)
axs[1].plot(x, B3)
axs[1].set_xlabel('B')

# regras
Alinha = np.zeros()








