#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 14:48:47 2021

@author: dhruv
"""

#loading the dataset
import numpy as np 
import pandas as pd 
import scipy.io
from functions import *
from wrappers import *
import ipyparallel
import os, sys
import neuroseries as nts 
import time 
import matplotlib.pyplot as plt 
from Wavelets import MyMorlet as Morlet
from scipy.stats import kendalltau, pearsonr, sem, mannwhitneyu 

rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/UDStats.csv'

data = pd.read_csv(rwpath)

############################################################################################### 
    # WAKE FR v/s LATENCY
###############################################################################################  

#Cumulative plot 

t,pvalue = mannwhitneyu(data.FrcorrW_ex_UD.values,data.FrcorrW_FS_UD.values)
means_ex = np.nanmean(data.FrcorrW_ex_UD.values)
means_inh = np.nanmean(data.FrcorrW_FS_UD.values)

label = ['All sessions']
x = np.arange(len(label))  # the label locations
width = 0.35  # the width of the bars


fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, means_ex, width, label='Excitatory cells')
rects2 = ax.bar(x + width/2, means_inh, width, label='FS cells')

pval = np.vstack([(data.FrcorrW_ex_UD.values), (data.FrcorrW_FS_UD.values )])

x2 = [x-width/2, x+width/2]
plt.plot(x2, np.vstack(pval), 'o-', color = 'k')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Pearson r value')
ax.set_title('Wake FR v/s latency from last spike correlation')
ax.set_xticks(x)
ax.set_xticklabels(label)
ax.legend(loc = 'upper right')

fig.tight_layout()


############################################################################################### 
    # NREM FR v/s LATENCY
###############################################################################################  

#Cumulative plot 

t,pvalue = mannwhitneyu(data.FrcorrS_ex_UD.values,data.FrcorrS_FS_UD.values)
means_ex = np.nanmean(data.FrcorrS_ex_UD.values)
means_inh = np.nanmean(data.FrcorrS_FS_UD.values)

label = ['All sessions']
x = np.arange(len(label))  # the label locations
width = 0.35  # the width of the bars


fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, means_ex, width, label='Excitatory cells')
rects2 = ax.bar(x + width/2, means_inh, width, label='FS cells')

pval = np.vstack([(data.FrcorrS_ex_UD.values), (data.FrcorrS_FS_UD.values )])

x2 = [x-width/2, x+width/2]
plt.plot(x2, np.vstack(pval), 'o-', color = 'k')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Pearson r value')
ax.set_title('NREM FR v/s latency from last spike correlation')
ax.set_xticks(x)
ax.set_xticklabels(label)
ax.legend(loc = 'lower left')

fig.tight_layout()

############################################################################################### 
    # LATENCY v/s DEPTH
###############################################################################################  

#Cumulative plot 

t,pvalue = mannwhitneyu(data.nonparcorr_ex_UD.values,data.nonparcorr_FS_UD.values)
means_ex = np.nanmean(data.nonparcorr_ex_UD.values)
means_inh = np.nanmean(data.nonparcorr_FS_UD.values)

label = ['All sessions']
x = np.arange(len(label))  # the label locations
width = 0.35  # the width of the bars


fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, means_ex, width, label='Excitatory cells')
rects2 = ax.bar(x + width/2, means_inh, width, label='FS cells')

pval = np.vstack([(data.nonparcorr_ex_UD.values), (data.nonparcorr_FS_UD.values )])

x2 = [x-width/2, x+width/2]
plt.plot(x2, np.vstack(pval), 'o-', color = 'k')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Pearson r value')
ax.set_title('Latency from last spike v/s cell depth correlation')
ax.set_xticks(x)
ax.set_xticklabels(label)
ax.legend(loc = 'lower left')

fig.tight_layout()