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


#%% On Lab PC
data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
# datasets = np.loadtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
datasets = np.loadtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'

corrs = []
pvals = []

depthcorrs = []
depthpvals = []

for s in datasets:
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    rawpath = os.path.join(rwpath,s)

    data = nap.load_session(rawpath, 'neurosuite')
    data.load_neurosuite_xml(rawpath)
    spikes = data.spikes  
    epochs = data.epochs
    
    # ############################################################################################### 
    #     # LOAD MAT FILES
    # ############################################################################################### 
    filepath = os.path.join(path, 'Analysis')
    listdir    = os.listdir(filepath)
    file = [f for f in listdir if 'BehavEpochs' in f]
    behepochs = scipy.io.loadmat(os.path.join(filepath,file[0]))  

    file = [f for f in listdir if 'CellDepth' in f]
    celldepth = scipy.io.loadmat(os.path.join(filepath,file[0]))
    depth = celldepth['cellDep']

    file = [f for f in listdir if 'MeanFR' in f]
    mfr = scipy.io.loadmat(os.path.join(filepath,file[0]))
    r_wake = mfr['rateS']


    file = [f for f in listdir if 'CellTypes' in f]
    celltype = scipy.io.loadmat(os.path.join(filepath,file[0]))
    
    pyr = []
    interneuron = []
    hd = []
        
    for i in range(len(spikes)):
        if celltype['ex'][i] == 1 and celltype['gd'][i] == 1:
            pyr.append(i)
            
    for i in range(len(spikes)):
        if celltype['fs'][i] == 1 and celltype['gd'][i] == 1:
            interneuron.append(i)
            
    for i in range(len(spikes)):
        if celltype['hd'][i] == 1 and celltype['gd'][i] == 1:
            hd.append(i)

# ############################################################################################### 
#     # LOAD UP AND DOWN STATE, NEW SWS AND NEW WAKE EPOCHS
# ###############################################################################################   

    
    file = os.path.join(rawpath, name +'.evt.py.dow')
    if os.path.exists(file):
        tmp = np.genfromtxt(file)[:,0]
        tmp = tmp.reshape(len(tmp)//2,2)/1000
        down_ep = nap.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
    
    file = os.path.join(rawpath, name +'.evt.py.upp')
    if os.path.exists(file):
        tmp = np.genfromtxt(file)[:,0]
        tmp = tmp.reshape(len(tmp)//2,2)/1000
        up_ep = nap.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
    
    file = os.path.join(rawpath, name +'.DM.new_sws.evt')
    if os.path.exists(file):
        tmp = np.genfromtxt(file)[:,0]
        tmp = tmp.reshape(len(tmp)//2,2)/1000
        new_sws_ep = nap.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
    
    file = os.path.join(rawpath, name +'.DM.new_wake.evt')
    if os.path.exists(file):
        tmp = np.genfromtxt(file)[:,0]
        tmp = tmp.reshape(len(tmp)//2,2)/1000
        new_wake_ep = nap.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
    
        
#%% 

#On Nibelungen 

# data_directory = '/mnt/DataNibelungen/Dhruv/'
# rwpath = '/mnt/DataNibelungen/Dhruv/MEC-UPState'
# datasets = np.genfromtxt(os.path.join(rwpath,'MEC_dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')

# corrs = []
# pvals = []

# depthcorrs = []
# depthpvals = []

# for s in datasets:
#     print(s)
#     name = s.split('/')[-1]
#     path = os.path.join(data_directory, s)
#     rawpath = os.path.join(rwpath,s)

#     data = nap.load_session(path, 'neurosuite')
#     data.load_neurosuite_xml(path)
#     channelorder = data.group_to_channel[0]
#     spikes = data.spikes
#     epochs = data.epochs
    
# # ############################################################################################### 
# #     # LOAD UP AND DOWN STATE, NEW SWS AND NEW WAKE EPOCHS
# # ###############################################################################################   
    
#     file = os.path.join(path, name +'.sws.evt')
#     new_sws_ep = data.read_neuroscope_intervals(name = 'SWS', path2file = file)
    
#     file = os.path.join(path, name +'.evt.py.dow')
#     down_ep = data.read_neuroscope_intervals(name = 'DOWN', path2file = file)
    
#     file = os.path.join(path, name +'.evt.py.upp')
#     up_ep = data.read_neuroscope_intervals(name = 'UP', path2file = file)


#%% 


############################################################################################### 
    # COMPUTE EVENT CROSS CORRS
###############################################################################################  
      
## Peak firing       
    cc2 = nap.compute_eventcorrelogram(spikes, nap.Tsd(up_ep['start'].values), binsize = 0.005, windowsize = 0.255, ep = new_sws_ep, norm = False)
    tmp = pd.DataFrame(cc2)
    tmp = tmp.rolling(window=4, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
    dd2 = tmp[-0.05:0.25]
    
    peaks = dd2.max()
    peaklocs = dd2.idxmax()
    
    peak_mag_above_mean = peaks.values / spikes.restrict(up_ep)._metadata['freq'].values
    
    corr, p = kendalltau(peaklocs, peak_mag_above_mean)
    corrs.append(corr)
    pvals.append(p)
    
    plt.figure()
    plt.title('Peak/ mean FR v/s UP onset_' + s)
    plt.scatter(peaklocs,peak_mag_above_mean, label = 'R = ' + str((round(corr,2))))
    plt.xlabel('Time from UP onset (s)')
    plt.ylabel('Peak/mean FR')
    plt.legend(loc = 'upper right')

    depthcorr, depthp = kendalltau(peak_mag_above_mean, depth)
    depthcorrs.append(depthcorr)
    depthpvals.append(depthp)

    # plt.figure()
    # plt.title('Peak/ mean FR v/s Depth_' + s)
    # plt.scatter(peak_mag_above_mean,depth, label = 'R = ' + str((round(depthcorr,2))))
    # plt.xlabel('Peak/mean FR')
    # plt.ylabel('Depth from top of probe (um)')
    # plt.legend(loc = 'upper right')

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




    
        





