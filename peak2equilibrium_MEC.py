#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 15:58:14 2022

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

#On Nibelungen 

data_directory = '/mnt/DataNibelungen/Dhruv/'
rwpath = '/mnt/DataNibelungen/Dhruv/MEC-UPState'
datasets = np.genfromtxt(os.path.join(rwpath,'MEC_dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')

corrs = []
pvals = []

depthcorrs = []
depthpvals = []

for s in datasets:
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    rawpath = os.path.join(rwpath,s)

    data = nap.load_session(path, 'neurosuite')
    data.load_neurosuite_xml(path)
    channelorder = data.group_to_channel[0]
    spikes = data.spikes
    epochs = data.epochs
    
# ############################################################################################### 
#     # LOAD UP AND DOWN STATE, NEW SWS AND NEW WAKE EPOCHS
# ###############################################################################################   
    
    file = os.path.join(path, name +'.sws.evt')
    new_sws_ep = data.read_neuroscope_intervals(name = 'SWS', path2file = file)
    
    file = os.path.join(path, name +'.evt.py.dow')
    down_ep = data.read_neuroscope_intervals(name = 'DOWN', path2file = file)
    
    file = os.path.join(path, name +'.evt.py.upp')
    up_ep = data.read_neuroscope_intervals(name = 'UP', path2file = file)


#%% 


############################################################################################### 
    # COMPUTE EVENT CROSS CORRS
###############################################################################################  
      
## Peak firing       
    cc2 = nap.compute_eventcorrelogram(spikes, nap.Tsd(up_ep['start'].values), binsize = 0.005, windowsize = 0.255, ep = new_sws_ep, norm = True)
    tmp = pd.DataFrame(cc2)
    tmp = tmp.rolling(window=4, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
    dd2 = tmp[0:0.105]
    
    if len(dd2.columns) > 0:
                    
        indexplot = []
        peaks_keeping = []
                
            
        for i in range(len(dd2.columns)):
            a = np.where(dd2.iloc[:,i] > 0.5)
            if len(a[0]) > 0:
                peaks_keeping.append(dd2.iloc[:,i].max())
                res = dd2.iloc[:,i].index[a]
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

#%% 

##Summary plot 
summary = pd.DataFrame()
summary['corr'] = corrs
summary['p'] = pvals
summary['depthcorr'] = depthcorrs
summary['depthp'] = depthpvals 

plt.figure()
plt.boxplot(corrs, positions=[0], showfliers=False, patch_artist=True, boxprops=dict(facecolor='royalblue', color='royalblue'),
              capprops=dict(color='royalblue'),
              whiskerprops=dict(color='royalblue'),
              medianprops=dict(color='white', linewidth = 2))
plt.boxplot(depthcorrs, positions=[0.3], showfliers=False, patch_artist=True, boxprops=dict(facecolor='lightsteelblue', color='lightsteelblue'),
              capprops=dict(color='lightsteelblue'),
              whiskerprops=dict(color='lightsteelblue'),
              medianprops=dict(color='white', linewidth = 2))

x1 = np.random.normal(0, 0.01, size=len(summary['corr'][summary['p'] < 0.05]))
x2 = np.random.normal(0.3, 0.01, size=len(summary['depthcorr'][summary['depthp'] < 0.05]))
x3 = np.random.normal(0.3, 0.01, size=len(summary['depthcorr'][summary['depthp'] >= 0.05]))

plt.plot(x1, summary['corr'][summary['p'] < 0.05], '.', color = 'k', fillstyle = 'none', markersize = 6, zorder =3)
plt.plot(x2, summary['depthcorr'][summary['depthp'] < 0.05], '.', color = 'k', fillstyle = 'none', markersize = 6, zorder =3, label = 'p < 0.05')
plt.plot(x3, summary['depthcorr'][summary['depthp'] >= 0.05], 'x', color = 'k', fillstyle = 'none', markersize = 6, zorder =3, label = 'p >= 0.05')
plt.axhline(0, color = 'silver')
plt.xticks([0, 0.3],['vs delay', 'vs depth'])
plt.title('Peak/mean FR v/s UP onset - Summary')
plt.legend(loc = 'upper right')
plt.ylabel('Tau value')




    
        





