#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 14:28:23 2022

@author: jose
"""

import simpful as sf
import numpy as np
import matplotlib.pyplot as plt

FS = sf.FuzzySystem()

# antecedentes
sd = 1.5
S_1 = sf.FuzzySet(function = sf.Gaussian_MF(mu = -2, sigma = sd), term = 'menos_dois')
S_2 = sf.FuzzySet(function = sf.Gaussian_MF(mu = 2, sigma = sd), term = 'mais_dois')
FS.add_linguistic_variable('x', sf.LinguisticVariable([S_1, S_2], universe_of_discourse = [-2, 2]))

# consequentes
FS.set_crisp_output_value('MENOS_UM', -1)
FS.set_crisp_output_value('MAIS_UM', 1)

# regras
RULE1 = 'IF (x IS menos_dois) THEN (y IS MENOS_UM)'
RULE2 = 'IF (x IS mais_dois) THEN (y IS MAIS_UM)'
FS.add_rules([RULE1, RULE2])

# sintese
N = 100
X = np.linspace(start = -2, stop = 2, num = N)
y = np.tanh(X)
yhat = []
for x in X:
    FS.set_variable('x', x)
    yhat.append(FS.Sugeno_inference(['y']).get('y'))

plt.plot(X, yhat)
plt.plot(X, y)
mse = np.sqrt(np.sum((y - yhat) ** 2)) / len(yhat)
print(mse)
