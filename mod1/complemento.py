#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 13:20:29 2022

@author: jose
"""

import numpy as np

def zadeh(a):
    return 1 - a

def yager(a, w):
    return (1 - a ** w) ** (1 / w)

def sugeno(a, s):
    return (1 - a) / (1 + s * a)
