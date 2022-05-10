#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 16:02:59 2022

@author: jose
"""

import simpful as sf
import numpy as np
import matplotlib.pyplot as plt

FS = sf.FuzzySystem()

# antecedents
# S_1 = sf.FuzzySet(function = sf.Gaussian_MF(mu = np.array([0, 0]), sigma = 0.2), term = 'azul')
# S_2 = sf.FuzzySet(function = sf.Gaussian_MF(mu = np.array([-1, 1]), sigma = 0.2), term = 'preto')
# S_3 = sf.FuzzySet(function = sf.Gaussian_MF(mu = np.array([1, 1]), sigma = 0.2), term = 'laranja')
# S_4 = sf.FuzzySet(function = sf.Gaussian_MF(mu = np.array([1, -1]), sigma = 0.2), term = 'roxo')
# S_5 = sf.FuzzySet(function = sf.Gaussian_MF(mu = np.array([-1, -1]), sigma = 0.2), term = 'amarelo')
# FS.add_linguistic_variable('x', sf.LinguisticVariable([S_1, S_2, S_3, S_4, S_5]))

S_1 = sf.FuzzySet(function = sf.Gaussian_MF(mu = 0, sigma = 0.2), term = 'xazul')
S_2 = sf.FuzzySet(function = sf.Gaussian_MF(mu = -1, sigma = 0.2), term = 'xpreto')
S_3 = sf.FuzzySet(function = sf.Gaussian_MF(mu = 1, sigma = 0.2), term = 'xlaranja')
S_4 = sf.FuzzySet(function = sf.Gaussian_MF(mu = 1, sigma = 0.2), term = 'xroxo')
S_5 = sf.FuzzySet(function = sf.Gaussian_MF(mu = -1, sigma = 0.2), term = 'xamarelo')
FS.add_linguistic_variable('x', sf.LinguisticVariable([S_1, S_2, S_3, S_4, S_5], universe_of_discourse = [-2, 2]))

T_1 = sf.FuzzySet(function = sf.Gaussian_MF(mu = 0, sigma = 0.2), term = 'yazul')
T_2 = sf.FuzzySet(function = sf.Gaussian_MF(mu = 1, sigma = 0.2), term = 'ypreto')
T_3 = sf.FuzzySet(function = sf.Gaussian_MF(mu = 1, sigma = 0.2), term = 'ylaranja')
T_4 = sf.FuzzySet(function = sf.Gaussian_MF(mu = -1, sigma = 0.2), term = 'yroxo')
T_5 = sf.FuzzySet(function = sf.Gaussian_MF(mu = -1, sigma = 0.2), term = 'yamarelo')
FS.add_linguistic_variable('y', sf.LinguisticVariable([T_1, T_2, T_3, T_4, T_5], universe_of_discourse = [-2, 2]))

# consequent
# FS.set_crisp_output_value('AZUL', np.array([0, 0]))
# FS.set_crisp_output_value('PRETO', np.array([-1, 1]))
# FS.set_crisp_output_value('LARANHA', np.array([1, 1]))
# FS.set_crisp_output_value('ROXO', np.array([1, -1]))
# FS.set_crisp_output_value('AMARELO', np.array([-1, -1]))

FS.set_crisp_output_value('AZUL', 1)
FS.set_crisp_output_value('PRETO', 2)
FS.set_crisp_output_value('LARANJA', 3)
FS.set_crisp_output_value('ROXO', 4)
FS.set_crisp_output_value('AMARELO', 5)

# rules
RULE1 = 'IF (x IS xazul) AND (y IS yazul) THEN (z IS AZUL)'
RULE2 = 'IF (x IS xpreto) AND (y IS ypreto) THEN (z IS PRETO)'
RULE3 = 'IF (x IS xlaranja) AND (y IS ylaranja) THEN (z IS LARANJA)'
RULE4 = 'IF (x IS xroxo) AND (y IS yroxo) THEN (z IS ROXO)'
RULE5 = 'IF (x IS xamarelo) AND (y IS yamarelo) THEN (z IS AMARELO)'
FS.add_rules([RULE1, RULE2, RULE3, RULE4, RULE5])

# contorno
N = 100
xx = np.linspace(-2, 2, N)
yy = np.linspace(-2, 2, N)
yy[::-1].sort()
XX, YY = np.meshgrid(xx, yy)
Z = np.zeros(shape = XX.shape)
for i in range(N):
    for j in range(N):
        x = XX[i, j]
        y = YY[i, j]
        FS.set_variable('x', x)
        FS.set_variable('y', y)
        # Z[i, j] = round(FS.Sugeno_inference(['z']).get('z'))
        Z[i, j] = FS.Sugeno_inference(['z']).get('z')
plt.imshow(Z)
plt.savefig('fig/classificacao.png')
