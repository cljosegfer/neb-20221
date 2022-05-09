#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 16:11:54 2022

@author: jose
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# antecendentes
delta = 0.1
x = ctrl.Antecedent(np.arange(-2, 2, delta), 'x')
y = ctrl.Consequent(np.arange(0, 4, delta), 'y')

# mf
x['menos dois'] = fuzz.trimf(x.universe, [-3, -2, -1])
x['menos um'] = fuzz.trimf(x.universe, [-2, -1, 0])
x['zero'] = fuzz.trimf(x.universe, [-1, 0, 1])
x['mais um'] = fuzz.trimf(x.universe, [0, 1, 2])
x['mais dois'] = fuzz.trimf(x.universe, [1, 2, 3])

# sd = 0.75
# c1 = 1
# x['menos dois'] = fuzz.gaussmf(x.universe, -2, sd)
# x['menos um'] = fuzz.gaussmf(x.universe, c1, sd)
# x['zero'] = fuzz.gaussmf(x.universe, 0, sd)
# x['mais um'] = fuzz.gaussmf(x.universe, c1, sd)
# x['mais dois'] = fuzz.gaussmf(x.universe, 2, sd)

y['zero'] = fuzz.trimf(y.universe, [-1, 0, 1])
y['um'] = fuzz.trimf(y.universe, [0, 1, 2])
y['quatro'] = fuzz.trimf(y.universe, [1, 4, 5])

# y['zero'] = fuzz.gaussmf(y.universe, 0, sd)
# y['um'] = fuzz.gaussmf(y.universe, c1**2, sd)
# y['quatro'] = fuzz.gaussmf(y.universe, 4, sd)

# n sei fazer tsk
# x['positivo'] = fuzz.trimf(x.universe, [-3, -2, 2])
# x['negativo'] = fuzz.trimf(x.universe, [-2, 2, 3])

# regras
rule1 = ctrl.Rule(x['menos dois'], y['quatro'])
rule2 = ctrl.Rule(x['mais dois'], y['quatro'])
rule3 = ctrl.Rule(x['zero'], y['zero'])
rule4 = ctrl.Rule(x['menos um'], y['um'])
rule5 = ctrl.Rule(x['mais um'], y['um'])

# controle
y_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
yy = ctrl.ControlSystemSimulation(y_ctrl)

ylog = []
for i in x.universe:
    yy.input['x'] = i
    yy.compute()
    ylog.append(yy.output['y'])

plt.plot(x.universe, ylog)

ytrue = x.universe ** 2
acc = np.sqrt(np.sum((ytrue - ylog) ** 2)) / len(ylog)
print(acc)

