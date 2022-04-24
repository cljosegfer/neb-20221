#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 13:41:31 2022

@author: jose
"""

import numpy as np
from pertinencia import trimf
from uniao import maximo, probabilistica, limitada, drastica
import matplotlib.pyplot as plt
from matplotlib import cm

li = 0
ls = 10
delta = 0.1
N = int((ls - li) / delta + 1)

# pertinencia
x = np.linspace(li, ls, N)
mu1 = trimf(x, a = 2, b = 3, c = 5)
mu2 = trimf(x, a = 4, b = 5, c = 9)
mu_maximo = maximo(mu1, mu2)
mu_probabilistica = probabilistica(mu1, mu2)
mu_limitada = limitada(mu1, mu2)
mu_drastica = drastica(mu1, mu2)

# plot
fig, axs = plt.subplots(5, figsize = (8, 10))
axs[0].plot(x, mu1, color='b')
axs[0].plot(x, mu2, color='g')
axs[0].set_ylabel('mfs')
axs[1].plot(x, mu_maximo, color='r')
axs[1].set_ylabel('maximo')
axs[2].plot(x, mu_probabilistica, color='r')
axs[2].set_ylabel('probabilistica')
axs[3].plot(x, mu_limitada, color='r')
axs[3].set_ylabel('limitada')
axs[4].plot(x, mu_drastica, color='r')
axs[4].set_ylabel('drastica')

plt.setp(axs, xticks=np.linspace(start = li, stop = ls, num = 11), 
         yticks=np.linspace(start = 0, stop = 1, num = 6));
plt.savefig('fig/ex2.png')

# superficie
a = np.linspace(0, 1, N)
b = np.linspace(0, 1, N)
A, B = np.meshgrid(a, b)
surf_maximo = maximo(A, B)
surf_probabilistica = probabilistica(A, B)
surf_limitada = limitada(A, B)
surf_drastica = drastica(A, B)

# plot
# fig = plt.figure()
# ax = fig.add_subplot(2, 1, 1, projection='3d')
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot_surface(A, B, surf_maximo, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
axs[0, 0].set_zlabel('maximo')

# ax = fig.add_subplot(2, 1, 2, projection='3d')
# surf = ax.plot_surface(A, B, surf_probabilistica, cmap=cm.coolwarm,
#                         linewidth=0, antialiased=False)
# ax.set_zlabel('probabilistica')

# ax = fig.add_subplot(2, 2, 1, projection='3d')
# surf = ax.plot_surface(A, B, surf_limitada, cmap=cm.coolwarm,
#                         linewidth=0, antialiased=False)
# ax.set_zlabel('limitada')

# ax = fig.add_subplot(2, 2, 2, projection='3d')
# surf = ax.plot_surface(A, B, surf_drastica, cmap=cm.coolwarm,
#                         linewidth=0, antialiased=False)
# ax.set_zlabel('drastica')

plt.savefig('fig/ex22.png')

