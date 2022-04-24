#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 13:13:11 2022

@author: jose
"""

import numpy as np
from pertinencia import gaussmf
from complemento import zadeh, yager, sugeno
import matplotlib.pyplot as plt

li = 0
ls = 10
delta = 0.1
N = int((ls - li) / delta + 1)

# pertinencia
x = np.linspace(li, ls, N)
mu = gaussmf(x, c = 5, sigma = 2)
mu_zadeh = zadeh(mu)
mu_yager = yager(mu, w = 3)
mu_sugeno = sugeno(mu, s = 2)

# plot
fig, axs = plt.subplots(4, figsize = (8, 10))
axs[0].plot(x, mu, color='b')
axs[0].set_ylabel('mf')
axs[1].plot(x, mu_zadeh, color='r')
axs[1].set_ylabel('zadeh')
axs[2].plot(x, mu_yager, color='r')
axs[2].set_ylabel('yager, m = 3')
axs[3].plot(x, mu_sugeno, color='r')
axs[3].set_ylabel('sugeno, s = 2')
plt.setp(axs, xticks=np.linspace(start = li, stop = ls, num = 11), 
         yticks=np.linspace(start = 0, stop = 1, num = 6));
plt.savefig('fig/ex1.png')

# superficie
x = np.linspace(0, 1, N)
x_zadeh = zadeh(x)
x_yager1 = yager(x, w = 3)
x_yager2 = yager(x, w = 1)
x_yager3 = yager(x, w = 0.7)
x_sugeno1 = sugeno(x, s = -0.7)
x_sugeno2 = sugeno(x, s = 0)
x_sugeno3 = sugeno(x, s = 20)

# plot
fig, axs = plt.subplots(3, figsize = (8, 12))
axs[0].plot(x, x_zadeh, color='b')
axs[0].set_ylabel('zadeh')
axs[1].plot(x, x_yager1, color='b')
axs[1].plot(x, x_yager2, color='r')
axs[1].plot(x, x_yager3, color='g')
axs[1].set_ylabel('yager, m = 3, 1, 0.7')
axs[2].plot(x, x_sugeno1, color='b')
axs[2].plot(x, x_sugeno2, color='r')
axs[2].plot(x, x_sugeno3, color='g')
axs[2].set_ylabel('sugeno, s = -0.7, 0, 20')
plt.setp(axs, xticks=np.linspace(start = 0, stop = 1, num = 11), 
         yticks=np.linspace(start = 0, stop = 1, num = 6));
plt.savefig('fig/ex11.png')
