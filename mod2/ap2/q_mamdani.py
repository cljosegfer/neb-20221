#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 14:00:08 2022

@author: jose
"""

import simpful as sf
import numpy as np
import matplotlib.pyplot as plt

FS = sf.FuzzySystem()

# antecedent
S_1 = sf.FuzzySet(function = sf.Triangular_MF(a = -3, b = -2, c = -1), term = 'doisn')
S_2 = sf.FuzzySet(function = sf.Triangular_MF(a = -2, b = -1, c = 0), term = 'umn')
S_3 = sf.FuzzySet(function = sf.Triangular_MF(a = -1, b = 0, c = 1), term = 'zero')
S_4 = sf.FuzzySet(function = sf.Triangular_MF(a = 0, b = 1, c = 2), term = 'ump')
S_5 = sf.FuzzySet(function = sf.Triangular_MF(a = 1, b = 2, c = 3), term = 'doisp')
FS.add_linguistic_variable('x', sf.LinguisticVariable([S_1, S_2, S_3, S_4, S_5], universe_of_discourse = [-2, 2]))

# consequent
T_1 = sf.FuzzySet(function = sf.Triangular_MF(a = -1, b = 0, c = 1), term = 'ZERO')
T_2 = sf.FuzzySet(function = sf.Triangular_MF(a = 0, b = 1, c = 2), term = 'UM')
T_3 = sf.FuzzySet(function = sf.Triangular_MF(a = 1, b = 4, c = 5), term = 'QUATRO')
FS.add_linguistic_variable('y', sf.LinguisticVariable([T_1, T_2, T_3], universe_of_discourse = [0, 4]))

# rules
RULE1 = 'IF (x IS doisn) THEN (y IS QUATRO)'
RULE2 = 'IF (x IS umn) THEN (y IS UM)'
RULE3 = 'IF (x IS zero) THEN (y IS ZERO)'
RULE4 = 'IF (x IS ump) THEN (y IS UM)'
RULE5 = 'IF (x IS doisp) THEN (y IS QUATRO)'
FS.add_rules([RULE1, RULE2, RULE3, RULE4, RULE5])

# sintese
N = 100
X = np.linspace(start = -2, stop = 2, num = N)
y = X ** 2
yhat = []
for x in X:
    FS.set_variable('x', x)
    yhat.append(FS.inference().get('y'))

plt.plot(X, yhat)
plt.plot(X, y)
plt.savefig('fig/q_mamdani.png')
mse = np.sqrt(np.sum((y - yhat) ** 2)) / len(yhat)
print(mse)
