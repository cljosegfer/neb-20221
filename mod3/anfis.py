#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 12:05:32 2022

@author: jose
"""

import numpy as np
import matplotlib.pyplot as plt

class anfis:
    def __init__(self, n, m):
        self.n = n
        self.c = np.random.randn(m, self.n)
        self.s = np.random.randn(m, self.n)
        self.P = np.random.randn(m, self.n)
        self.q = np.random.randn(self.n)
        self.log = None
    
    def _forward(self, x, regra):
        mu = np.zeros(shape = x.shape)
        y = self.q[regra]
        for j, x_j in enumerate(x):
            arg = (x_j - self.c[j, regra]) / self.s[j, regra]
            mu[j] = np.exp(-1 / 2 * arg**2)
            y += self.P[j, regra] * x_j
        w = np.product(mu, axis = 0)
        return w, y
    
    def forward(self, x):
        w = np.zeros(self.n)
        y = np.zeros(self.n)
        for regra in range(self.n):
            w[regra], y[regra] = self._forward(x, regra)
        b = np.sum(w)
        yhat = np.sum(y * w) / b
        return yhat, y, w, b
        
    def fit(self, X, y, max_epochs = 100, alpha = 0.01):
        self.log = []
        for epoch in range(max_epochs):
            for i, x_i in enumerate(X):
                yhat, Y, W, b = self.forward(x_i)
                de_dyhat = yhat - y[i]
                # print('erro: {}, y: {}, yhat: {}'.format(de_dyhat, y[i], yhat))
                
                dyhat_dW = np.zeros(self.n)
                dyhat_dY = np.zeros(self.n)
                for regra in range(self.n):
                    dyhat_dW[regra] = (Y[regra] - yhat) / b
                    dyhat_dY[regra] = W[regra] / b
                    
                    for j, x_j in enumerate(x_i):
                        dW_dc = W[regra] * (x_j - self.c[j, regra]) / self.s[j, regra]**2
                        dW_ds = W[regra] * (x_j - self.c[j, regra])**2 / self.s[j, regra]**3
                        dY_dP = x_j
                        
                        # update
                        self.c[j, regra] = self.c[j, regra] - alpha * de_dyhat * dyhat_dW[regra] * dW_dc
                        self.s[j, regra] = self.s[j, regra] - alpha * de_dyhat * dyhat_dW[regra] * dW_ds
                        self.P[j, regra] = self.P[j, regra] - alpha * de_dyhat * dyhat_dY[regra] * dY_dP
                    self.q[regra] = self.q[regra] - alpha * de_dyhat * dyhat_dY[regra]
            self.log.append(self.mse(X, y))
    
    def mse(self, X, y):
        yhat = self.predict(X).reshape(-1, 1)
        return np.square(y - yhat).mean()
        
    def predict(self, X):
        W = np.zeros(shape = (X.shape[0], self.n))
        Y = np.zeros(shape = (X.shape[0], self.n))
        for i, x_i in enumerate(X):
            for regra in range(self.n):
                W[i, regra], Y[i, regra] = self._forward(x_i, regra)
        yhat = np.sum(Y * W, axis = 1) / np.sum(W, axis = 1)
        return yhat