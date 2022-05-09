#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 14:12:52 2022

@author: jose
"""

import simpful as sf
import numpy as np
import matplotlib.pyplot as plt

FS = sf.FuzzySystem()

# antecedent
# S_1 = sf.FuzzySet(function = sf.Triangular_MF(a = -3, b = -2, c = 2), term = 'negativo')
# S_2 = sf.FuzzySet(function = sf.Triangular_MF(a = -2, b = 2, c = 3), term = 'positivo')
S_1 = sf.FuzzySet(function = sf.Gaussian_MF(mu = -2, sigma = 2), term = 'negativo')
S_2 = sf.FuzzySet(function = sf.Gaussian_MF(mu = 2, sigma = 2), term = 'positivo')
FS.add_linguistic_variable('x', sf.LinguisticVariable([S_1, S_2], universe_of_discourse = [-2, 2]))

# consequent
FS.set_output_function('decrescente', '-2*x')
FS.set_output_function('crescente', '2*x')
# FS.set_output_function('-2x', lambda x: -2*x)
# FS.set_output_function('2x', lambda x: 2*x)

# rules
RULE1 = 'IF (x IS negativo) THEN (y IS decrescente)'
RULE2 = 'IF (x IS positivo) THEN (y IS crescente)'
FS.add_rules([RULE1, RULE2])

# sintese
N = 100
X = np.linspace(start = -2, stop = 2, num = N)
y = X ** 2
yhat = []
for x in X:
    FS.set_variable('x', x)
    yhat.append(FS.Sugeno_inference(['y']).get('y'))

plt.plot(X, yhat)
plt.plot(X, y)
plt.savefig('fig/q_sugeno_linear.png')
mse = np.sqrt(np.sum((y - yhat) ** 2)) / len(yhat)
print(mse)
