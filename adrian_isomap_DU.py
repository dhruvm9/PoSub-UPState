#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 13:05:05 2023

@author: dhruv
"""

import numpy as np 
import pandas as pd 
import scipy.io
import pynapple as nap 
import os, sys
import time 
import matplotlib.pyplot as plt 
from random import sample
from matplotlib.colors import hsv_to_rgb
from scipy.stats import kendalltau, pearsonr, wilcoxon, mannwhitneyu
from itertools import combinations
import seaborn as sns
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

#%% 

def full_ang(ep,position, nb_bins):
    
    starts = []
    ends = []
    count = np.zeros(nb_bins-1)
    bins = np.linspace(0,2*np.pi,nb_bins)
    ang_pos = position.restrict(ep)
    ang_time = ang_pos.times()

    idx = np.digitize(ang_pos,bins)-1
    
    start = 0
    for i,j in enumerate(idx):
        count[j] += 1
        if np.all(count >= 1):
            starts.append(start)
            ends.append(i)
            count = np.zeros(nb_bins-1)
            start = i+1
            
    t_start = ang_time[starts]
    t_end = ang_time[ends]
    
    full_ang_ep = nap.IntervalSet(start = t_start, end = t_end)
    
    return full_ang_ep

#%% 

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'

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
    
    file = [f for f in listdir if 'Layers' in f]
    lyr = scipy.io.loadmat(os.path.join(filepath,file[0]))
    layer = lyr['l']
    
    file = [f for f in listdir if 'Velocity' in f]
    vl = scipy.io.loadmat(os.path.join(filepath,file[0]), simplify_cells = True)
    vel = nap.Tsd(t = vl['vel']['t'], d = vl['vel']['data']  )
    
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


#%% 

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

    filepath = os.path.join(path, 'Analysis')
    data = pd.read_csv(filepath + '/Tracking_data.csv', header = None)
    position = pd.DataFrame(index = data[0].values, data = data[[1,2,3]].values, columns=['x', 'y', 'ang'])
    position = position.loc[~position.index.duplicated(keep='first')]
    position['ang'] = position['ang'] *(np.pi/180) #convert degrees to radian
    position['ang'] = (position['ang'] + 2*np.pi) % (2*np.pi) #convert [-pi, pi] to [0, 2pi]
    position = nap.TsdFrame(position) 
    position = position.restrict(epochs['wake'])
    
    angle = position['ang'].loc[vel.index.values]
    
#%%  Visualization
    
### Wake binning 

    wake_bin = 0.2 #200ms binwidth
    
    ep = full_ang(angle.time_support, angle, 60)
      
    dx = position['x'].bin_average(wake_bin,ep)
    dy = position['y'].bin_average(wake_bin,ep)
    ang = angle.bin_average(wake_bin, ep)
    v = vel.bin_average(wake_bin,ep)

    v = v.threshold(2)
       
    ang = ang.loc[v.index.values]
    
    wake_count = spikes[hd].count(wake_bin, ep) 
    wake_count = wake_count.loc[v.index.values]
           
    wake_count = wake_count.as_dataframe()
    wake_rate = np.sqrt(wake_count/wake_bin)
    wake_rate = wake_rate.rolling(window = 50, win_type = 'gaussian', center = True, min_periods = 1, axis = 0).mean(std = 3)
        
    samples = sample(list(np.arange(0,len(ep)-1)), round(len(ep)/5))
    samples = np.sort(samples)
    
    sub_ep = nap.IntervalSet(start = ep.iloc[samples]['start'], end = ep.iloc[samples]['end'])    
    
    ang = ang.restrict(sub_ep)
    ang = ang.dropna()
      
    wake_rate = wake_rate.loc[ang.index.values]

### Sleep binning 

    du = nap.IntervalSet(start = up_ep['start'] - 0.025, end = up_ep['start'] + 0.15) 
    
    sleep_dt = 0.025 
    sleep_binwidth = 0.1

    num_overlapping_bins = int(sleep_binwidth/sleep_dt)    
    
    sleep_count = spikes[hd].count(sleep_dt, du.loc[[0]]) 
    sleep_count = sleep_count.as_dataframe()
    sleep_rate = np.sqrt(sleep_count/sleep_dt)
    sleep_rate = sleep_rate.rolling(window = num_overlapping_bins, win_type = 'gaussian', center = True, min_periods = 1, axis = 0).mean(std = 3)
        
    rate = np.vstack([wake_rate.values, sleep_rate.values])
               
    projection = Isomap(n_components = 2, n_neighbors = 20).fit_transform(rate)
        
    H = ang.values/(2*np.pi)
    HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T
    RGB = hsv_to_rgb(HSV)    
       
    cmap = plt.colormaps['Greys']
    col = np.linspace(0,1,len(projection[:,0][len(wake_rate)+1:]))
    
    
    dx = np.diff(projection[:,0][len(wake_rate)+1:])
    dy = np.diff(projection[:,1][len(wake_rate)+1:])
    
    plt.figure(figsize = (8,8))
    plt.scatter(projection[:,0][0:len(wake_rate)], projection[:,1][0:len(wake_rate)], c = RGB)
    
    for i in range(len(projection[:,0][len(wake_rate)+1:])):
        plt.plot(projection[:,0][i], projection[:,1][i], 'o-', color = cmap(col)[i])
        
    
    
    plt.xticks([])
    plt.yticks([])
    
    
#%% 

