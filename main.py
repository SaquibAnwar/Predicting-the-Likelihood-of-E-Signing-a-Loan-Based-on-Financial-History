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