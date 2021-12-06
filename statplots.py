#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 12:06:16 2021

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

rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Stats.csv'

data = pd.read_csv(rwpath)

############################################################################################### 
    # WAKE v/s NREM FR
###############################################################################################  

t23, p23 = mannwhitneyu(data[data['Layer'] == 23 ].NREMwake_ex.values, data[data['Layer'] == 23 ].NREMwake_inh.values) 
t3, p3 = mannwhitneyu(data[data['Layer'] == 3 ].NREMwake_ex.values, data[data['Layer'] == 3 ].NREMwake_inh.values) 
t34, p34 = mannwhitneyu(data[data['Layer'] == 34 ].NREMwake_ex.values, data[data['Layer'] == 34 ].NREMwake_inh.values)  

labels = ['Layer 2/3', 'Layer 3', 'Layer 3/4']
wakeNREM_means_ex = [np.mean(data[data['Layer'] == 23].NREMwake_ex.values), np.mean(data[data['Layer'] == 3].NREMwake_ex.values), np.mean(data[data['Layer'] == 34].NREMwake_ex.values)]
wakeNREM_means_inh = [np.mean(data[data['Layer'] == 23].NREMwake_inh.values), np.mean(data[data['Layer'] == 3].NREMwake_inh.values), np.mean(data[data['Layer'] == 34].NREMwake_inh.values)]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars


fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, wakeNREM_means_ex, width, label='Excitatory cells')
rects2 = ax.bar(x + width/2, wakeNREM_means_inh, width, label='FS cells')

p = {}
p[0] = np.vstack([(data[data['Layer'] == 23 ].NREMwake_ex.values), (data[data['Layer'] == 23 ].NREMwake_inh.values )])
p[1] = np.vstack([(data[data['Layer'] == 3 ].NREMwake_ex.values), (data[data['Layer'] == 3 ].NREMwake_inh.values )])
p[2] = np.vstack([(data[data['Layer'] == 34 ].NREMwake_ex.values), (data[data['Layer'] == 34 ].NREMwake_inh.values )])



for i in range(3):
    x2 = [x[i]-width/2, x[i]+width/2]
    plt.plot(x2, np.vstack(p[i]), 'o-', color = 'k')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Pearson r value')
ax.set_title('NREM v/s wake FR correlation')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc = 'upper left')

fig.tight_layout()


#Cumulative plot 

t,pvalue = mannwhitneyu(data.NREMwake_ex.values,data.NREMwake_inh.values)
means_ex = np.nanmean(data.NREMwake_ex.values)
means_inh = np.nanmean(data.NREMwake_inh.values)

label = ['All sessions']
x = np.arange(len(label))  # the label locations
width = 0.35  # the width of the bars


fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, means_ex, width, label='Excitatory cells')
rects2 = ax.bar(x + width/2, means_inh, width, label='FS cells')

pval = np.vstack([(data.NREMwake_ex.values), (data.NREMwake_inh.values )])

x2 = [x-width/2, x+width/2]
plt.plot(x2, np.vstack(pval), 'o-', color = 'k')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Pearson r value')
ax.set_title('NREM v/s wake FR correlation')
ax.set_xticks(x)
ax.set_xticklabels(label)
ax.legend(loc = 'upper left')

fig.tight_layout()


############################################################################################### 
    # NREM FR v/s DEPTH
###############################################################################################  

t23, p23 = mannwhitneyu(data[data['Layer'] == 23 ].NREM_FRDepth_ex.values, data[data['Layer'] == 23 ].NREM_FRDepth_inh.values) 
t3, p3 = mannwhitneyu(data[data['Layer'] == 3 ].NREM_FRDepth_ex.values, data[data['Layer'] == 3 ].NREM_FRDepth_inh.values) 
t34, p34 = mannwhitneyu(data[data['Layer'] == 34 ].NREM_FRDepth_ex.values, data[data['Layer'] == 34 ].NREM_FRDepth_inh.values)  

labels = ['Layer 2/3', 'Layer 3', 'Layer 3/4']
NREM_means_ex = [np.mean(data[data['Layer'] == 23].NREM_FRDepth_ex.values), np.mean(data[data['Layer'] == 3].NREM_FRDepth_ex.values), np.mean(data[data['Layer'] == 34].NREM_FRDepth_ex.values)]
NREM_means_inh = [np.mean(data[data['Layer'] == 23].NREM_FRDepth_inh.values), np.mean(data[data['Layer'] == 3].NREM_FRDepth_inh.values), np.mean(data[data['Layer'] == 34].NREM_FRDepth_inh.values)]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars


fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, NREM_means_ex, width, label='Excitatory cells')
rects2 = ax.bar(x + width/2, NREM_means_inh, width, label='FS cells')

p = {}
p[0] = np.vstack([(data[data['Layer'] == 23 ].NREM_FRDepth_ex.values), (data[data['Layer'] == 23 ].NREM_FRDepth_inh.values )])
p[1] = np.vstack([(data[data['Layer'] == 3 ].NREM_FRDepth_ex.values), (data[data['Layer'] == 3 ].NREM_FRDepth_inh.values )])
p[2] = np.vstack([(data[data['Layer'] == 34 ].NREM_FRDepth_ex.values), (data[data['Layer'] == 34 ].NREM_FRDepth_inh.values )])



for i in range(3):
    x2 = [x[i]-width/2, x[i]+width/2]
    plt.plot(x2, np.vstack(p[i]), 'o-', color = 'k')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Pearson r value')
ax.set_title('NREM FR v/s cell depth correlation')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc = 'upper left')

fig.tight_layout()

#Cumulative plot 

t,pvalue = mannwhitneyu(data.NREM_FRDepth_ex.values,data.NREM_FRDepth_inh.values)
means_ex = np.nanmean(data.NREM_FRDepth_ex.values)
means_inh = np.nanmean(data.NREM_FRDepth_inh.values)

label = ['All sessions']
x = np.arange(len(label))  # the label locations
width = 0.35  # the width of the bars


fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, means_ex, width, label='Excitatory cells')
rects2 = ax.bar(x + width/2, means_inh, width, label='FS cells')

pval = np.vstack([(data.NREM_FRDepth_ex.values), (data.NREM_FRDepth_inh.values )])

x2 = [x-width/2, x+width/2]
plt.plot(x2, np.vstack(pval), 'o-', color = 'k')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Pearson r value')
ax.set_title('NREM FR v/s cell depth correlation')
ax.set_xticks(x)
ax.set_xticklabels(label)
ax.legend(loc = 'upper left')

fig.tight_layout()

############################################################################################### 
    # WAKE FR v/s DEPTH
###############################################################################################  

t23, p23 = mannwhitneyu(data[data['Layer'] == 23 ].Wake_FRDepth_ex.values, data[data['Layer'] == 23 ].Wake_FRDepth_inh.values) 
t3, p3 = mannwhitneyu(data[data['Layer'] == 3 ].Wake_FRDepth_ex.values, data[data['Layer'] == 3 ].Wake_FRDepth_inh.values) 
t34, p34 = mannwhitneyu(data[data['Layer'] == 34 ].Wake_FRDepth_ex.values, data[data['Layer'] == 34 ].Wake_FRDepth_inh.values)  

labels = ['Layer 2/3', 'Layer 3', 'Layer 3/4']
Wake_means_ex = [np.mean(data[data['Layer'] == 23].Wake_FRDepth_ex.values), np.mean(data[data['Layer'] == 3].Wake_FRDepth_ex.values), np.mean(data[data['Layer'] == 34].Wake_FRDepth_ex.values)]
Wake_means_inh = [np.mean(data[data['Layer'] == 23].Wake_FRDepth_inh.values), np.mean(data[data['Layer'] == 3].Wake_FRDepth_inh.values), np.mean(data[data['Layer'] == 34].Wake_FRDepth_inh.values)]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars


fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, Wake_means_ex, width, label='Excitatory cells')
rects2 = ax.bar(x + width/2, Wake_means_inh, width, label='FS cells')

p = {}
p[0] = np.vstack([(data[data['Layer'] == 23 ].Wake_FRDepth_ex.values), (data[data['Layer'] == 23 ].Wake_FRDepth_inh.values )])
p[1] = np.vstack([(data[data['Layer'] == 3 ].Wake_FRDepth_ex.values), (data[data['Layer'] == 3 ].Wake_FRDepth_inh.values )])
p[2] = np.vstack([(data[data['Layer'] == 34 ].Wake_FRDepth_ex.values), (data[data['Layer'] == 34 ].Wake_FRDepth_inh.values )])



for i in range(3):
    x2 = [x[i]-width/2, x[i]+width/2]
    plt.plot(x2, np.vstack(p[i]), 'o-', color = 'k')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Pearson r value')
ax.set_title('Wake FR v/s cell depth correlation')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc = 'upper left')

fig.tight_layout()

#Cumulative plot 

t,pvalue = mannwhitneyu(data.Wake_FRDepth_ex.values,data.Wake_FRDepth_inh.values)
means_ex = np.nanmean(data.Wake_FRDepth_ex.values)
means_inh = np.nanmean(data.Wake_FRDepth_inh.values)

label = ['All sessions']
x = np.arange(len(label))  # the label locations
width = 0.35  # the width of the bars


fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, means_ex, width, label='Excitatory cells')
rects2 = ax.bar(x + width/2, means_inh, width, label='FS cells')

pval = np.vstack([(data.Wake_FRDepth_ex.values), (data.Wake_FRDepth_inh.values )])

x2 = [x-width/2, x+width/2]
plt.plot(x2, np.vstack(pval), 'o-', color = 'k')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Pearson r value')
ax.set_title('WakeFR v/s cell depth correlation')
ax.set_xticks(x)
ax.set_xticklabels(label)
ax.legend(loc = 'upper left')

fig.tight_layout()

############################################################################################### 
    # WAKE FR v/s LATENCY
###############################################################################################  

t23, p23 = mannwhitneyu(data[data['Layer'] == 23 ].Wake_frcorr_ex.values, data[data['Layer'] == 23 ].Wake_frcorr_inh.values) 
t3, p3 = mannwhitneyu(data[data['Layer'] == 3 ].Wake_frcorr_ex.values, data[data['Layer'] == 3 ].Wake_frcorr_inh.values) 
t34, p34 = mannwhitneyu(data[data['Layer'] == 34 ].Wake_frcorr_ex.values, data[data['Layer'] == 34 ].Wake_frcorr_inh.values)  

labels = ['Layer 2/3', 'Layer 3', 'Layer 3/4']
Wake_means_ex = [np.nanmean(data[data['Layer'] == 23].Wake_frcorr_ex.values), np.nanmean(data[data['Layer'] == 3].Wake_frcorr_ex.values), np.nanmean(data[data['Layer'] == 34].Wake_frcorr_ex.values)]
Wake_means_inh = [np.nanmean(data[data['Layer'] == 23].Wake_frcorr_inh.values), np.nanmean(data[data['Layer'] == 3].Wake_frcorr_inh.values), np.nanmean(data[data['Layer'] == 34].Wake_frcorr_inh.values)]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars


fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, Wake_means_ex, width, label='Excitatory cells')
rects2 = ax.bar(x + width/2, Wake_means_inh, width, label='FS cells')

p = {}
p[0] = np.vstack([(data[data['Layer'] == 23 ].Wake_frcorr_ex.values), (data[data['Layer'] == 23 ].Wake_frcorr_inh.values )])
p[1] = np.vstack([(data[data['Layer'] == 3 ].Wake_frcorr_ex.values), (data[data['Layer'] == 3 ].Wake_frcorr_inh.values )])
p[2] = np.vstack([(data[data['Layer'] == 34 ].Wake_frcorr_ex.values), (data[data['Layer'] == 34 ].Wake_frcorr_inh.values )])



for i in range(3):
    x2 = [x[i]-width/2, x[i]+width/2]
    plt.plot(x2, np.vstack(p[i]), 'o-', color = 'k')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Pearson r value')
ax.set_title('Wake FR v/s latency to first spike correlation')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc = 'upper left')

fig.tight_layout()

#Cumulative plot 

t,pvalue = mannwhitneyu(data.Wake_frcorr_ex.values,data.Wake_frcorr_inh.values)
means_ex = np.nanmean(data.Wake_frcorr_ex.values)
means_inh = np.nanmean(data.Wake_frcorr_inh.values)

label = ['All sessions']
x = np.arange(len(label))  # the label locations
width = 0.35  # the width of the bars


fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, means_ex, width, label='Excitatory cells')
rects2 = ax.bar(x + width/2, means_inh, width, label='FS cells')

pval = np.vstack([(data.Wake_frcorr_ex.values), (data.Wake_frcorr_inh.values )])

x2 = [x-width/2, x+width/2]
plt.plot(x2, np.vstack(pval), 'o-', color = 'k')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Pearson r value')
ax.set_title('Wake FR v/s latency to first spike correlation')
ax.set_xticks(x)
ax.set_xticklabels(label)
ax.legend(loc = 'upper left')

fig.tight_layout()


############################################################################################### 
    # NREM FR v/s LATENCY
###############################################################################################  

t23, p23 = mannwhitneyu(data[data['Layer'] == 23 ].NREM_frcorr_ex.values, data[data['Layer'] == 23 ].NREM_frcorr_inh.values) 
t3, p3 = mannwhitneyu(data[data['Layer'] == 3 ].NREM_frcorr_ex.values, data[data['Layer'] == 3 ].NREM_frcorr_inh.values) 
t34, p34 = mannwhitneyu(data[data['Layer'] == 34 ].NREM_frcorr_ex.values, data[data['Layer'] == 34 ].NREM_frcorr_inh.values)  

labels = ['Layer 2/3', 'Layer 3', 'Layer 3/4']
NREM_means_ex = [np.nanmean(data[data['Layer'] == 23].NREM_frcorr_ex.values), np.nanmean(data[data['Layer'] == 3].NREM_frcorr_ex.values), np.nanmean(data[data['Layer'] == 34].NREM_frcorr_ex.values)]
NREM_means_inh = [np.nanmean(data[data['Layer'] == 23].NREM_frcorr_inh.values), np.nanmean(data[data['Layer'] == 3].NREM_frcorr_inh.values), np.nanmean(data[data['Layer'] == 34].NREM_frcorr_inh.values)]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars


fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, NREM_means_ex, width, label='Excitatory cells')
rects2 = ax.bar(x + width/2, NREM_means_inh, width, label='FS cells')

p = {}
p[0] = np.vstack([(data[data['Layer'] == 23 ].NREM_frcorr_ex.values), (data[data['Layer'] == 23 ].NREM_frcorr_inh.values )])
p[1] = np.vstack([(data[data['Layer'] == 3 ].NREM_frcorr_ex.values), (data[data['Layer'] == 3 ].NREM_frcorr_inh.values )])
p[2] = np.vstack([(data[data['Layer'] == 34 ].NREM_frcorr_ex.values), (data[data['Layer'] == 34 ].NREM_frcorr_inh.values )])



for i in range(3):
    x2 = [x[i]-width/2, x[i]+width/2]
    plt.plot(x2, np.vstack(p[i]), 'o-', color = 'k')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Pearson r value')
ax.set_title('NREM FR v/s latency to first spike correlation')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc = 'lower left')

fig.tight_layout()

#Cumulative plot 

t,pvalue = mannwhitneyu(data.NREM_frcorr_ex.values,data.NREM_frcorr_inh.values)
means_ex = np.nanmean(data.NREM_frcorr_ex.values)
means_inh = np.nanmean(data.NREM_frcorr_inh.values)

label = ['All sessions']
x = np.arange(len(label))  # the label locations
width = 0.35  # the width of the bars


fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, means_ex, width, label='Excitatory cells')
rects2 = ax.bar(x + width/2, means_inh, width, label='FS cells')

pval = np.vstack([(data.NREM_frcorr_ex.values), (data.NREM_frcorr_inh.values )])

x2 = [x-width/2, x+width/2]
plt.plot(x2, np.vstack(pval), 'o-', color = 'k')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Pearson r value')
ax.set_title('NREM FR v/s latency to first spike correlation')
ax.set_xticks(x)
ax.set_xticklabels(label)
ax.legend(loc = 'lower left')

fig.tight_layout()

############################################################################################### 
    # LATENCY v/s DEPTH
###############################################################################################  

t23, p23 = mannwhitneyu(data[data['Layer'] == 23 ].nonparcorr_ex.values, data[data['Layer'] == 23 ].nonparcorr_inh.values) 
t3, p3 = mannwhitneyu(data[data['Layer'] == 3 ].nonparcorr_ex.values, data[data['Layer'] == 3 ].nonparcorr_inh.values) 
t34, p34 = mannwhitneyu(data[data['Layer'] == 34 ].nonparcorr_ex.values, data[data['Layer'] == 34 ].nonparcorr_inh.values)  

labels = ['Layer 2/3', 'Layer 3', 'Layer 3/4']
means_ex = [np.nanmean(data[data['Layer'] == 23].nonparcorr_ex.values), np.nanmean(data[data['Layer'] == 3].nonparcorr_ex.values), np.nanmean(data[data['Layer'] == 34].nonparcorr_ex.values)]
means_inh = [np.nanmean(data[data['Layer'] == 23].nonparcorr_inh.values), np.nanmean(data[data['Layer'] == 3].nonparcorr_inh.values), np.nanmean(data[data['Layer'] == 34].nonparcorr_inh.values)]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars


fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, means_ex, width, label='Excitatory cells')
rects2 = ax.bar(x + width/2, means_inh, width, label='FS cells')

p = {}
p[0] = np.vstack([(data[data['Layer'] == 23 ].nonparcorr_ex.values), (data[data['Layer'] == 23 ].nonparcorr_inh.values )])
p[1] = np.vstack([(data[data['Layer'] == 3 ].nonparcorr_ex.values), (data[data['Layer'] == 3 ].nonparcorr_inh.values )])
p[2] = np.vstack([(data[data['Layer'] == 34 ].nonparcorr_ex.values), (data[data['Layer'] == 34 ].nonparcorr_inh.values )])



for i in range(3):
    x2 = [x[i]-width/2, x[i]+width/2]
    plt.plot(x2, np.vstack(p[i]), 'o-', color = 'k')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Pearson r value')
ax.set_title('Latency to first spike v/s cell depth correlation')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc = 'upper left')

fig.tight_layout()

#Cumulative plot 

t,pvalue = mannwhitneyu(data.nonparcorr_ex.values,data.nonparcorr_inh.values)
means_ex = np.nanmean(data.nonparcorr_ex.values)
means_inh = np.nanmean(data.nonparcorr_inh.values)

label = ['All sessions']
x = np.arange(len(label))  # the label locations
width = 0.35  # the width of the bars


fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, means_ex, width, label='Excitatory cells')
rects2 = ax.bar(x + width/2, means_inh, width, label='FS cells')

pval = np.vstack([(data.nonparcorr_ex.values), (data.nonparcorr_inh.values )])

x2 = [x-width/2, x+width/2]
plt.plot(x2, np.vstack(pval), 'o-', color = 'k')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Pearson r value')
ax.set_title('Latency to first spike v/s cell depth correlation')
ax.set_xticks(x)
ax.set_xticklabels(label)
ax.legend(loc = 'upper left')

fig.tight_layout()