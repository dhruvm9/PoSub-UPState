#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:57:02 2024

@author: dhruv
"""

import numpy as np 
import pandas as pd 
import pynapple as nap 
import nwbmatic as ntm
import scipy.io
import os, sys
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import kendalltau, pearsonr, wilcoxon, mannwhitneyu
from sklearn.linear_model import LinearRegression
from matplotlib.colors import hsv_to_rgb

#%% 

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
# datasets = np.loadtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'

FR = []
Depth = []
onsets = []

for s in datasets:
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    rawpath = os.path.join(rwpath,s)

    data = ntm.load_session(rawpath, 'neurosuite')
    data.load_neurosuite_xml(rawpath)
    spikes = data.spikes  
    epochs = data.epochs
    
#%% LOAD MAT FILES

    filepath = os.path.join(path, 'Analysis')
    listdir    = os.listdir(filepath)
    
    file = [f for f in listdir if 'CellDepth' in f]
    celldepth = scipy.io.loadmat(os.path.join(filepath,file[0]))
    depth = celldepth['cellDep']
    
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
            
#%% LOAD UP AND DOWN STATES 

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
    
#%% COMPUTE FIRING RATE IN NREM

    NREM_fr = spikes.restrict(nap.IntervalSet(new_sws_ep))._metadata['rate']
 
#%% COMPUTE EVENT CROSS CORRS 

    cc2 = nap.compute_eventcorrelogram(spikes, nap.Ts(up_ep['start'].values), binsize = 0.005, windowsize = 0.255, ep = up_ep, norm = True)
    tmp = pd.DataFrame(cc2)
    tmp = tmp.rolling(window=8, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
    dd2 = tmp[0:0.155]  
    ee = dd2[pyr] 
   
    if len(ee.columns) > 0:
                    
        tokeep = []
        depths_keeping_ex = []
        sess_uponset = []
        NREM_fr_ex = []    
        
        for i in range(len(ee.columns)):
            a = np.where(ee.iloc[:,i] > 0.5)
            if len(a[0]) > 0:
                tokeep.append(ee.columns[i])  
                depths_keeping_ex.append(depth.flatten()[ee.columns[i]])
                
                res = ee.iloc[:,i].index[a]
                sess_uponset.append(res[0])
                
                NREM_fr_ex.append(NREM_fr[ee.columns[i]])
                
        Depth.extend(depths_keeping_ex)
        FR.extend(NREM_fr_ex)
        onsets.extend(sess_uponset)
                
#%% Organize firingrate, UP onset delay and depth 

    H = 1 - depths_keeping_ex/(min(depths_keeping_ex))
    cmap = plt.cm.inferno
    
    plt.figure()
    plt.title(s)
    plt.scatter(sess_uponset, NREM_fr_ex, c = cmap(H))
    
#%% Pooled plot 

rf, pf = pearsonr(onsets, FR)
rd, pd = pearsonr(onsets, Depth)

Hpool = 1 - Depth/(min(Depth))
cmap = plt.cm.inferno

plt.figure()
plt.title('Pooled plot')
plt.scatter(onsets, FR, c = cmap(Hpool))