#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 15:12:25 2022

@author: jose
"""

import simpful as sf
import numpy as np
import matplotlib.pyplot as plt

FS = sf.FuzzySystem()

# antecedent
S_1 = sf.FuzzySet(function = sf.Triangular_MF(a = -1, b = 0, c = 4.5), term = 'maximo')
# S_2 = sf.FuzzySet(function = sf.Triangular_MF(a = 0, b = np.pi, c = 4.5), term = 'rooti')
S_3 = sf.FuzzySet(function = sf.Triangular_MF(a = 0, b = 4.5, c = 2*np.pi), term = 'minimo')
# S_4 = sf.FuzzySet(function = sf.Triangular_MF(a = 4.5, b = 2*np.pi, c = 2*np.pi+1), term = 'roots')
# FS.add_linguistic_variable('x', sf.LinguisticVariable([S_1, S_2, S_3, S_4], universe_of_discourse = [0, 2*np.pi]))
FS.add_linguistic_variable('x', sf.LinguisticVariable([S_1, S_3], universe_of_discourse = [0, 2*np.pi]))

# consequent
T_1 = sf.FuzzySet(function = sf.Triangular_MF(a = -1.22, b = -0.22, c = 1), term = 'MINIMO')
# T_2 = sf.FuzzySet(function = sf.Triangular_MF(a = -0.22, b = 0, c = 1), term = 'ZERO')
T_3 = sf.FuzzySet(function = sf.Triangular_MF(a = -0.22, b = 1, c = 2), term = 'MAXIMO')
# FS.add_linguistic_variable('y', sf.LinguisticVariable([T_1, T_2, T_3], universe_of_discourse = [-0.22, 1]))
FS.add_linguistic_variable('y', sf.LinguisticVariable([T_1, T_3], universe_of_discourse = [-0.22, 1]))

# rules
RULE1 = 'IF (x IS maximo) THEN (y IS MAXIMO)'
# RULE2 = 'IF (x IS rooti) THEN (y IS ZERO)'
RULE3 = 'IF (x IS minimo) THEN (y IS MINIMO)'
# RULE4 = 'IF (x IS roots) THEN (y IS ZERO)'
# FS.add_rules([RULE1, RULE2, RULE3, RULE4])
FS.add_rules([RULE1, RULE3])

# sintese
N = 100
X = np.linspace(start = 1e-4, stop = 2 * np.pi, num = N)
y = np.sin(X) / X
yhat = []
for x in X:
    FS.set_variable('x', x)
    yhat.append(FS.inference().get('y'))

plt.plot(X, yhat)
plt.plot(X, y)
plt.savefig('fig/s_mamdani.png')
mse = np.sqrt(np.sum((y - yhat) ** 2)) / len(yhat)
print(mse)
