#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 15:16:56 2022

@author: jose
"""

import pandas as pd

data_folder = 'data'

# read
df = pd.read_csv('{}/old/comvoi-en.tsv'.format(data_folder), sep = '\t')
mfcc = pd.read_csv('{}/old/comvoi-en-mfcc.tsv'.format(data_folder), sep = '\t')

# analise
gender = df.groupby('gender').size()
accent = df.groupby('accent').size().sort_values()

# subset
delete = accent[accent < 100].index
data = pd.concat([df, mfcc], axis = 1).drop(columns = ['wav_id', 'gender'])
data = data[data.accent.isin(delete) == False].reset_index(drop = True)

# write
data.to_csv('{}/comvoi-en.csv'.format(data_folder), index = None)
