#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 16:44:33 2022

@author: jose
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

# read
data_folder = 'data'

df = pd.read_csv('{}/old/cefala.tsv'.format(data_folder), sep = '\t')
files = pd.read_csv('{}/old/locutor.txt'.format(data_folder), header = None)

mfcc = []
for index, row in tqdm(files.iterrows()):
    file = '{}/old/q39/{}'.format(data_folder, row.values[0])
    q39 = np.load(file).reshape(-1)
    
    locutor = int(row.values[0][8:12])
    y = df[df.Locutor == locutor].Sotaque.values[0]
    mfcc.append(np.concatenate((q39, [y])))
data = pd.DataFrame(mfcc)

# analise
accent = data.groupby(39).size().sort_values()

# write
data.to_csv('{}/cefala.csv'.format(data_folder), index = None)
