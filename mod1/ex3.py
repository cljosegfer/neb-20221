#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:16:01 2022

@author: jose
"""

import numpy as np
from pertinencia import trapmf, sigmf
from intersecao import minimo, algebrico, limitado, drastico
import matplotlib.pyplot as plt

li = 0
ls = 10
delta = 0.1
N = int((ls - li) / delta + 1)

# pertinencia
x = np.linspace(li, ls, N)
mu1 = trapmf(x, a = 2, b = 3, c = 6, d = 7)
mu2 = sigmf(x, a = 2, c = 6)
mu_minimo = minimo(mu1, mu2)
mu_algebrico = algebrico(mu1, mu2)
mu_limitado = limitado(mu1, mu2)
mu_drastico = drastico(mu1, mu2)

# plot
fig, axs = plt.subplots(5, figsize = (8, 10))
axs[0].plot(x, mu1, color='b')
axs[0].plot(x, mu2, color='g')
axs[0].set_ylabel('mfs')
axs[1].plot(x, mu_minimo, color='r')
axs[1].set_ylabel('minimo')
axs[2].plot(x, mu_algebrico, color='r')
axs[2].set_ylabel('algebrico')
axs[3].plot(x, mu_limitado, color='r')
axs[3].set_ylabel('limitado')
axs[4].plot(x, mu_drastico, color='r')
axs[4].set_ylabel('drastico')

plt.setp(axs, xticks=np.linspace(start = li, stop = ls, num = 11), 
          yticks=np.linspace(start = 0, stop = 1, num = 6));
plt.savefig('fig/ex3.png')

# superficie
a = np.linspace(0, 1, N)
b = np.linspace(0, 1, N)
A, B = np.meshgrid(a, b)
surf_maximo = minimo(A, B)
surf_probabilistica = algebrico(A, B)
surf_limitada = limitado(A, B)
surf_drastica = drastico(A, B)

# plot
fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(surf_maximo, cmap = 'gray')
axs[0, 0].set_title('minimo')
axs[0, 0].axes.get_xaxis().set_visible(False)
axs[0, 0].axes.get_yaxis().set_visible(False)

axs[0, 1].imshow(surf_probabilistica, cmap = 'gray')
axs[0, 1].set_title('algebrico')
axs[0, 1].axes.get_xaxis().set_visible(False)
axs[0, 1].axes.get_yaxis().set_visible(False)

axs[1, 0].imshow(surf_limitada, cmap = 'gray')
axs[1, 0].set_title('limitado')
axs[1, 0].axes.get_xaxis().set_visible(False)
axs[1, 0].axes.get_yaxis().set_visible(False)

axs[1, 1].imshow(surf_drastica, cmap = 'gray')
axs[1, 1].set_title('drastico')
axs[1, 1].axes.get_xaxis().set_visible(False)
axs[1, 1].axes.get_yaxis().set_visible(False)

plt.savefig('fig/ex33.png')
