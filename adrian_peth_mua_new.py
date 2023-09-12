#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 17:17:16 2023

@author: dhruv
"""

import numpy as np 
import pandas as pd 
import scipy.io
from functions import *
from wrappers import *
import os, sys
import pynapple as nap
import time 
import matplotlib.pyplot as plt 
from scipy.stats import pearsonr 
from scipy.stats import wilcoxon, mannwhitneyu
import seaborn as sns

#%% 

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
# datasets = np.loadtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'

dur_D = []
dur_V = []
pmeans = []
diff = []
sess = []

for s in datasets:
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    rawpath = os.path.join(rwpath,s)

    spikes, shank = loadSpikeData(path)
    n_channels, fs, shank_to_channel = loadXML(rawpath)
  
    filepath = os.path.join(path, 'Analysis')
    listdir    = os.listdir(filepath)  
    file = [f for f in listdir if 'CellDepth' in f]
    celldepth = scipy.io.loadmat(os.path.join(filepath,file[0]))
    depth = celldepth['cellDep']
    
    file = [f for f in listdir if 'CellTypes' in f]
    celltype = scipy.io.loadmat(os.path.join(filepath,file[0]))
    
    data = pd.DataFrame()   
    data['depth'] = np.reshape(depth,(len(spikes.keys())),)
    data['level'] = pd.cut(data['depth'],2, precision=0, labels=[1,0]) #0 is dorsal, 1 is ventral
    data['celltype'] = np.nan
    data['gd'] = 0
    
    for i in range(len(spikes)):
        if celltype['gd'][i] == 1:
            data.loc[i,'gd'] = 1
            
    data = data[data['gd'] == 1]
    
    for i in range(len(spikes)):
        if celltype['ex'][i] == 1 and celltype['gd'][i] == 1:
            data.loc[i,'celltype'] = 'ex' #0 for excitatory
        elif celltype['fs'][i] == 1 and celltype['gd'][i] == 1:
            data.loc[i,'celltype'] = 'fs' #1 for inhibitory
    
    bin_size = 0.01 #seconds            
    
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
    
#%% Build dorsal and ventral MUA 

    mua = {}
    
    latency_dorsal = []
    latency_ventral = []
    
    # define mua for dorsal and ventral
    for i in range(2):
        mua[i] = []        
        for n in data[data['level'] == i].index:            
            mua[i].append(spikes[n].index.values)
        mua[i] = nap.Ts(t = np.sort(np.hstack(mua[i])))
        
#%% Generate PETH 


