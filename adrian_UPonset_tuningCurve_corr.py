#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 11:28:58 2022

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
from functions import * 

#%% 

corrs_onset = []
corrs_fr = []
pvals_onset = []
pvals_fr = []

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
# datasets = np.loadtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
datasets = np.loadtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'

peak_above_mean = []
uponset = []

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
    
#Load Position
    filepath = os.path.join(path, 'Analysis')
    data = pd.read_csv(filepath + '/Tracking_data.csv', header = None)
    position = pd.DataFrame(index = data[0].values, data = data[[1,2,3]].values, columns=['x', 'y', 'ang'])
    position = position.loc[~position.index.duplicated(keep='first')]
    position['ang'] = position['ang'] *(np.pi/180) #convert degrees to radian
    position['ang'] = (position['ang'] + 2*np.pi) % (2*np.pi) #convert [-pi, pi] to [0, 2pi]
    position = nap.TsdFrame(position)     
    
#%% 

#Compute tuning curves 

    tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ang'], 360, ep = new_wake_ep)
    hdcurves = tuning_curves[hd]
    hdcurves = smoothAngularTuningCurves(hdcurves)

#%%
        
#Compute event cross corrs 
    cc2 = nap.compute_eventcorrelogram(spikes, nap.Tsd(up_ep['start'].values), binsize = 0.005, windowsize = 0.255, ep = up_ep, norm = True)
    tmp = pd.DataFrame(cc2)
    tmp = tmp.rolling(window=4, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
    dd2 = tmp[0:0.105]
    ee = dd2[hd] 

#%% 

    tcwidth = []
    indexplot = []
    depths_keeping = []
    peaks_keeping = []
 
       
    for i in hdcurves: 
        
       idxpeak = np.where(hdcurves[i].index.values == hdcurves[i].idxmax())[0]
       maxIx = np.where(hdcurves[i].index.values == hdcurves[i].idxmax())[0]
       tc_shifted = np.roll(hdcurves[i], -maxIx[0]+180)
       idx_shifted = np.roll(hdcurves[i].index.values, -maxIx[0]+180)
       tc_shifted = tc_shifted/max(tc_shifted)
       tc = pd.Series(data = tc_shifted, index = idx_shifted)
       w0 = tc[tc > 0.5].index[0]
       w1 = tc[tc > 0.5].index[-1]
       width = np.minimum((2*np.pi - abs(w1-w0)), abs(w1-w0))
       
       a = np.where(ee[i] > 0.5)
       
       if width > 0.2 and width < 1.5 and len(a[0]) > 0:
           tcwidth.append(width)
             
           peaks_keeping.append(ee[i].max())
           peak_above_mean.append(ee[i].max())
           depths_keeping.append(depth.flatten()[i])
           
           res = ee[i].index[a]
           indexplot.append(res[0])
           uponset.append(res[0])
    
    # hdcurves.columns[tc_keeping]
    
    
#%% 
    
    r_onset, p_onset = kendalltau(indexplot,tcwidth)
    corrs_onset.append(r_onset)
    pvals_onset.append(p_onset)
    
    plt.figure()
    plt.title('UP onset v/s TC Width_' + s)
    plt.scatter(np.array(indexplot),np.array(tcwidth), label = 'R = ' + str(round(r_onset,2)))
    plt.xlabel('UP onset (ms)')
    plt.ylabel('TC Width')
    plt.legend(loc = 'upper right')

    r_fr, p_fr = kendalltau(peaks_keeping,tcwidth)
    corrs_fr.append(r_fr)
    pvals_fr.append(p_fr)
    
    plt.figure()
    plt.title('Peak/mean FR v/s TC Width_' + s)
    plt.scatter(np.array(peaks_keeping),np.array(tcwidth), label = 'R = ' + str(round(r_fr,2)))
    plt.xlabel('Peak/mean FR')
    plt.ylabel('TC Width')
    plt.legend(loc = 'upper right')
    
#%% 

summary = pd.DataFrame()
summary['corr_onset'] = corrs_onset
summary['p_onset'] = pvals_onset
summary['corr_fr'] = corrs_fr
summary['p_fr'] = pvals_fr

plt.figure()
plt.boxplot(corrs_onset, positions=[0], showfliers=False, patch_artist=True, boxprops=dict(facecolor='royalblue', color='royalblue'),
              capprops=dict(color='royalblue'),
              whiskerprops=dict(color='royalblue'),
              medianprops=dict(color='white', linewidth = 2))
plt.boxplot(corrs_fr, positions=[0.3], showfliers=False, patch_artist=True, boxprops=dict(facecolor='lightsteelblue', color='lightsteelblue'),
              capprops=dict(color='lightsteelblue'),
              whiskerprops=dict(color='lightsteelblue'),
              medianprops=dict(color='white', linewidth = 2))

x1 = np.random.normal(0, 0.01, size=len(summary['corr_onset'][summary['p_onset'] < 0.05]))
x2 = np.random.normal(0, 0.01, size=len(summary['corr_onset'][summary['p_onset'] >= 0.05]))
x3 = np.random.normal(0.3, 0.01, size=len(summary['corr_fr'][summary['p_fr'] < 0.05]))
x4 = np.random.normal(0.3, 0.01, size=len(summary['corr_fr'][summary['p_fr'] >= 0.05]))

plt.plot(x1, summary['corr_onset'][summary['p_onset'] < 0.05], 'x', color = 'k', fillstyle = 'none', markersize = 6, zorder =3, label = 'p < 0.05')
plt.plot(x2, summary['corr_onset'][summary['p_onset'] >= 0.05], '.', color = 'k', fillstyle = 'none', markersize = 6, zorder =3,  label = 'p >= 0.05')
plt.plot(x3, summary['corr_fr'][summary['p_fr'] < 0.05], 'x', color = 'k', fillstyle = 'none', markersize = 6, zorder =3)
plt.plot(x4, summary['corr_fr'][summary['p_fr'] >= 0.05], '.', color = 'k', fillstyle = 'none', markersize = 6, zorder =3)
plt.axhline(0, color = 'silver')
plt.xticks([0, 0.3],['vs UP onset', 'vs peak/mean FR'])
# plt.xticks([])
plt.title('Tuning Curve Width Correlations')
plt.legend(loc = 'upper right')


