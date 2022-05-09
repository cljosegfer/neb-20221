#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 15:48:06 2022

@author: jose
"""

import simpful as sf
import numpy as np
import matplotlib.pyplot as plt

FS = sf.FuzzySystem()

# antecedent
S_1 = sf.FuzzySet(function = sf.Triangular_MF(a = 0, b = 1.5, c = 5.2), term = 'queda')
S_2 = sf.FuzzySet(function = sf.Triangular_MF(a = 1.5, b = 5.2, c = 2*np.pi), term = 'subida')
FS.add_linguistic_variable('x', sf.LinguisticVariable([S_1, S_2], universe_of_discourse = [0, 2*np.pi]))

# consequent
FS.set_output_function('decrescente', '-0.4*x+1.26')
FS.set_output_function('crescente', '0.12*x-0.79')

RULE1 = 'IF (x IS queda) THEN (y IS decrescente)'
RULE2 = 'IF (x IS subida) THEN (y IS crescente)'
FS.add_rules([RULE1, RULE2])

# sintese
N = 100
X = np.linspace(start = 1e-4, stop = 2*np.pi-1e-4, num = N)
y = np.sin(X) / X
yhat = []
for x in X:
    FS.set_variable('x', x)
    yhat.append(FS.Sugeno_inference(['y']).get('y'))

plt.plot(X, yhat)
plt.plot(X, y)
plt.savefig('fig/s_sugeno_linear.png')
mse = np.sqrt(np.sum((y - yhat) ** 2)) / len(yhat)
print(mse)

