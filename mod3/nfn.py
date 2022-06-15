#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 15:01:56 2022

@author: jose
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.utils import shuffle

class nfn:
    def __init__(self, N = 100):
        self.w_i = None
        self.w_s = None
        self.N = N
        self.log = None
        self.delta = None
        self.minimo = None
        self.maximo = None
    
    def forward(self, x, index):
        y = 0
        mu = np.zeros(len(x))
        for j, x_j in enumerate(x):
            offset = (x_j // (2 * self.delta[j])) * 2 * self.delta[j]
            reta = (x_j - offset) / self.delta[j]
            if reta > 1:
                reta = reta - 1
            mu[j] = reta
            indice = index[j]
            try:
                w1 = self.w_i[j, indice]
            except IndexError:
                w1 = 0.5
            try:
                w2 = self.w_s[j, indice + 1]
            except IndexError:
                w2 = 0.5
            y += w1 * mu[j] + w2 * (1 - mu[j])
        return y, mu
            
    def fit(self, X, y, max_epochs = 100, alpha = 0.01):
        self.log = []
        m = X.shape[1]
        self.w_i = np.zeros(shape = (m, self.N))
        self.w_s = np.zeros(shape = (m, self.N))
        self.delta = np.zeros(m)
        self.minimo = np.zeros(m)
        self.maximo = np.zeros(m)
        for j in range(m):
            self.minimo[j] = np.min(X[:, j])
            self.maximo[j] = np.max(X[:, j])
            self.delta[j] = (self.maximo[j] - self.minimo[j]) / 2 / self.N
        for epoch in tqdm(range(max_epochs)):
            X, y = shuffle(X, y)
            for i, x_i in enumerate(X):
                index = []
                for j in range(m):
                    offset = (x_i[j] // (2 * self.delta[j])) * 2 * self.delta[j]
                    index.append(int((offset - self.minimo[j]) // (2 * self.delta[j])))
                yhat, mu = self.forward(x_i, index)
                de_dyhat = yhat - y[i]
                
                if alpha == 'auto':
                    den = 0
                    for j in range(m):
                        den += mu[j]**2 + (1 - mu[j])**2
                    alpha = 1/den
                
                for j in range(m):
                    dyhat_dw = mu[j]
                    # update
                    indice = index[j]
                    try:
                        self.w_i[j, indice] = self.w_i[j, indice] - alpha * de_dyhat * dyhat_dw
                    except IndexError:
                        pass
                    try:
                        self.w_s[j, indice + 1] = self.w_s[j, indice + 1] - alpha * de_dyhat * (1 - dyhat_dw)
                    except IndexError:
                        pass
            self.log.append(self.mse(X, y))
            
    def mse(self, X, y):
        yhat = self.predict(X).reshape(-1, 1)
        return np.square(y - yhat).mean()
    
    def predict(self, X):
        m = X.shape[1]
        yhat = []
        for i, x_i in enumerate(X):
            index = []
            for j in range(m):
                offset = (x_i[j] // (2 * self.delta[j])) * 2 * self.delta[j]
                index.append(int((offset - self.minimo[j]) // (2 * self.delta[j])))
            y, _ = self.forward(x_i, index)
            yhat.append(y)
        return np.array(yhat)