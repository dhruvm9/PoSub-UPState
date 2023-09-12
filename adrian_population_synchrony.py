#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 17:31:20 2023

@author: dhruv
"""

import numpy as np 
import pandas as pd 
import scipy.io
from functions import *
from wrappers2 import *
import os, sys
import pynapple as nap
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import wilcoxon

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')
#datasets = np.loadtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'

#%% 

sth = []

for s in datasets:
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    rawpath = os.path.join(rwpath,s)
    
    spikes, shank = loadSpikeData(path)
    n_channels, fs, shank_to_channel = loadXML(rawpath)
    
    file = os.path.join(rawpath, name +'.sws.evt')
    if os.path.exists(file):
        tmp = np.genfromtxt(file)[:,0]
        tmp = tmp.reshape(len(tmp)//2,2)/1000
        sws_ep = nap.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
        
#%% 

    bin_size = 0.1 #s
    smoothing_window = 0.2 #0.02
    rates = spikes.count(bin_size, sws_ep) 
    
    total2 = rates.as_dataframe().rolling(window = 100 ,win_type='gaussian',
                                          center=True,min_periods=1, 
                                          axis = 0).mean(std= int(smoothing_window/bin_size))
    
    total2 = total2.sum(axis = 1)
    threshold = np.percentile(total2.values,20)
    
    synchrony = rates.astype(bool).sum(axis=1)/len(spikes)
    
    counts, bins = np.histogram(synchrony)
    
    bincenter = 0.5 * (bins[1:] + bins[:-1])
    
    m, b = np.polyfit(total2, synchrony, 1)
    
    synch_threshold = (m*threshold) + b
    sth.append(synch_threshold)
    
#%% 
    plt.figure()
    plt.suptitle(s)
    plt.subplot(121)
    plt.plot(bincenter, counts/sum(counts))
    plt.axvline(synch_threshold, color = 'k')
    plt.subplot(122)
    plt.scatter(total2,synchrony)
    

    
    