#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 17:32:28 2022

@author: dhruv
"""
import numpy as np 
import pandas as pd 
import scipy.io
import pynapple as nap 
import os, sys
import time 
import matplotlib.pyplot as plt 
from scipy.stats import kendalltau, pearsonr, wilcoxon, mannwhitneyu
import seaborn as sns

#%% 

corrs = []
pvals = []

data_directory = '/media/DataDhruv/Recordings/BWatson'

listdir    = os.listdir(data_directory)
file = [f for f in listdir if 'spikes' in f]
spikedata = scipy.io.loadmat(os.path.join(data_directory,file[0]))  

listdir    = os.listdir(data_directory)
file = [f for f in listdir if 'events' in f]
events = scipy.io.loadmat(os.path.join(data_directory,file[0]))  

listdir    = os.listdir(data_directory)
file = [f for f in listdir if 'states' in f]
states = scipy.io.loadmat(os.path.join(data_directory,file[0]))  


#%% 

#Load Spikes 

spks = spikedata['spikes']
spk = {}
for i in range(len(spks[0][0][1][0])):
    spk[i] = nap.Ts(spks[0][0][1][0][i])
spikes = nap.TsGroup(spk)

#Load Sleep States 

sleepstate = states['SleepState']
wake_ep = nap.IntervalSet(start = sleepstate[0][0][0][0][0][0][:,0], end = sleepstate[0][0][0][0][0][0][:,1])
nrem_ep = nap.IntervalSet(start = sleepstate[0][0][0][0][0][1][:,0], end = sleepstate[0][0][0][0][0][1][:,1])
rem_ep = nap.IntervalSet(start = sleepstate[0][0][0][0][0][2][:,0], end = sleepstate[0][0][0][0][0][2][:,1])


#Load UP and DOWN states 

slowwaves = events['SlowWaves']
up_ep = nap.IntervalSet( start = slowwaves[0][0][2][0][0][0][:,0], end = slowwaves[0][0][2][0][0][0][:,1])
down_ep = nap.IntervalSet( start = slowwaves[0][0][2][0][0][1][:,0], end = slowwaves[0][0][2][0][0][1][:,1])

#%% 

# COMPUTE EVENT CROSS CORRS
## Peak firing       

cc = nap.compute_eventcorrelogram(spikes, nap.Tsd(up_ep['start'].values), binsize = 0.005, windowsize = 0.255, ep = up_ep, norm = True)
tmp = pd.DataFrame(cc)
tmp = tmp.rolling(window=4, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
dd = tmp[0:0.105]

if len(dd.columns) > 0:
                
    indexplot = []
    peaks_keeping = []
            
        
    for i in range(len(dd.columns)):
        a = np.where(dd.iloc[:,i] > 0.5)
        if len(a[0]) > 0:
            peaks_keeping.append(dd.iloc[:,i].max())
            res = dd.iloc[:,i].index[a]
            indexplot.append(res[0])

corr, p = kendalltau(indexplot, peaks_keeping)
corrs.append(corr)
pvals.append(p)

plt.figure()
plt.rc('font', size = 15)
plt.title('Peak/ mean FR v/s UP onset')
plt.scatter(indexplot,peaks_keeping, label = 'R = ' + str((round(corr,2))), color = 'cornflowerblue')
plt.xlabel('Time from UP onset (s)')
plt.ylabel('Peak/mean FR')
plt.legend(loc = 'upper right')
