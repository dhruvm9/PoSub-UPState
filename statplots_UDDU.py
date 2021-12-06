#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 15:07:58 2021

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

#Excitatory cells 
t,pvalue = mannwhitneyu(data.FrcorrW_ex_UD.values,data.FrcorrW_ex.values)
means_ud = np.nanmean(data.FrcorrW_ex_UD.values)
means_du = np.nanmean(data.FrcorrW_ex.values)

label = ['All sessions']
x = np.arange(len(label))  # the label locations
width = 0.35  # the width of the bars


fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, means_ud, width, label='UD')
rects2 = ax.bar(x + width/2, means_du, width, label='DU')

pval = np.vstack([(data.FrcorrW_ex_UD.values), (data.FrcorrW_ex.values )])

x2 = [x-width/2, x+width/2]
plt.plot(x2, np.vstack(pval), 'o-', color = 'k')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Pearson r value')
ax.set_title('Wake FR v/s latency correlation (excitatory cells)')
ax.set_xticks(x)
ax.set_xticklabels(label)
ax.legend(loc = 'upper right')
fig.tight_layout()

#FS cells 
t,pvalue = mannwhitneyu(data.FrcorrW_FS_UD.values,data.FrcorrW_FS.values)
means_ud = np.nanmean(data.FrcorrW_FS_UD.values)
means_du = np.nanmean(data.FrcorrW_FS.values)

label = ['All sessions']
x = np.arange(len(label))  # the label locations
width = 0.35  # the width of the bars


fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, means_ud, width, label='UD')
rects2 = ax.bar(x + width/2, means_du, width, label='DU')

pval = np.vstack([(data.FrcorrW_FS_UD.values), (data.FrcorrW_FS.values )])

x2 = [x-width/2, x+width/2]
plt.plot(x2, np.vstack(pval), 'o-', color = 'k')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Pearson r value')
ax.set_title('Wake FR v/s latency correlation (FS cells)')
ax.set_xticks(x)
ax.set_xticklabels(label)
ax.legend(loc = 'upper right')
fig.tight_layout()



############################################################################################### 
    # NREM FR v/s LATENCY
###############################################################################################  

#Excitatory cells 
t,pvalue = mannwhitneyu(data.FrcorrS_ex_UD.values,data.FrcorrS_ex.values)
means_ud = np.nanmean(data.FrcorrS_ex_UD.values)
means_du = np.nanmean(data.FrcorrS_ex.values)

label = ['All sessions']
x = np.arange(len(label))  # the label locations
width = 0.35  # the width of the bars


fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, means_ud, width, label='UD')
rects2 = ax.bar(x + width/2, means_du, width, label='DU')

pval = np.vstack([(data.FrcorrS_ex_UD.values), (data.FrcorrS_ex.values )])

x2 = [x-width/2, x+width/2]
plt.plot(x2, np.vstack(pval), 'o-', color = 'k')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Pearson r value')
ax.set_title('NREM FR v/s latency correlation (excitatory cells)')
ax.set_xticks(x)
ax.set_xticklabels(label)
ax.legend(loc = 'lower left')
fig.tight_layout()

#FS cells 
t,pvalue = mannwhitneyu(data.FrcorrS_FS_UD.values,data.FrcorrS_FS.values)
means_ud = np.nanmean(data.FrcorrS_FS_UD.values)
means_du = np.nanmean(data.FrcorrS_FS.values)

label = ['All sessions']
x = np.arange(len(label))  # the label locations
width = 0.35  # the width of the bars


fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, means_ud, width, label='UD')
rects2 = ax.bar(x + width/2, means_du, width, label='DU')

pval = np.vstack([(data.FrcorrS_FS_UD.values), (data.FrcorrS_FS.values )])

x2 = [x-width/2, x+width/2]
plt.plot(x2, np.vstack(pval), 'o-', color = 'k')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Pearson r value')
ax.set_title('NREM FR v/s latency correlation (FS cells)')
ax.set_xticks(x)
ax.set_xticklabels(label)
ax.legend(loc = 'lower left')
fig.tight_layout()

############################################################################################### 
    # LATENCY v/s DEPTH
###############################################################################################  

#Excitatory cells 
t,pvalue = mannwhitneyu(data.nonparcorr_ex_UD.values,data.nonparcorr_ex.values)
means_ud = np.nanmean(data.nonparcorr_ex_UD.values)
means_du = np.nanmean(data.nonparcorr_ex.values)

label = ['All sessions']
x = np.arange(len(label))  # the label locations
width = 0.35  # the width of the bars


fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, means_ud, width, label='UD')
rects2 = ax.bar(x + width/2, means_du, width, label='DU')

pval = np.vstack([(data.nonparcorr_ex_UD.values), (data.nonparcorr_ex.values )])

x2 = [x-width/2, x+width/2]
plt.plot(x2, np.vstack(pval), 'o-', color = 'k')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Pearson r value')
ax.set_title('Latency v/s cell depth correlation (excitatory cells)')
ax.set_xticks(x)
ax.set_xticklabels(label)
ax.legend(loc = 'lower left')
fig.tight_layout()

#FS cells 
t,pvalue = mannwhitneyu(data.nonparcorr_FS_UD.values,data.nonparcorr_FS.values)
means_ud = np.nanmean(data.nonparcorr_FS_UD.values)
means_du = np.nanmean(data.nonparcorr_FS.values)

label = ['All sessions']
x = np.arange(len(label))  # the label locations
width = 0.35  # the width of the bars


fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, means_ud, width, label='UD')
rects2 = ax.bar(x + width/2, means_du, width, label='DU')

pval = np.vstack([(data.nonparcorr_FS_UD.values), (data.nonparcorr_FS.values )])

x2 = [x-width/2, x+width/2]
plt.plot(x2, np.vstack(pval), 'o-', color = 'k')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Pearson r value')
ax.set_title('Latency v/s cell depth correlation (FS cells)')
ax.set_xticks(x)
ax.set_xticklabels(label)
ax.legend(loc = 'lower left')
fig.tight_layout()
