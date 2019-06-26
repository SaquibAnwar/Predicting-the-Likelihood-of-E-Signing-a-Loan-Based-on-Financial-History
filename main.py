#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 20:17:05 2019

@author: kryptonite
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

dataset = pd.read_csv('financial_data.csv')

### EDA ###

dataset.head()
dataset.columns
dataset.describe()

### Cleaning the Data ###
dataset.isna().any()


## Histograms

dataset2 = dataset.drop(columns = ['entry_id', 'pay_schedule', 'e_signed'])

fig = plt.figure(figsize=(15, 20))
plt.suptitle('Histogram of Numerical Columns', fontsize = 20)
for i in range(dataset2.shape[1]):
    plt.subplot(6, 3, i+1)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i])
    
    vals = np.size(dataset2.iloc[:, i].unique())
    if vals >= 100:
        vals = 100
        
    plt.hist(dataset2.iloc[:, i], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

## Correlation with response Variable (Note: Models like RF are not linear like these)

dataset2.corrwith(dataset.e_signed).plt.bar(
        fig)