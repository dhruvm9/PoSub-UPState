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

def perievent_Tsd(data, tref,  minmax):
    peth = {}
    
    a = data.index[data.index.get_indexer(tref.index.values, method='nearest')]
    
    tmp = nap.compute_perievent(data, nap.Ts(a.values) , minmax = minmax, time_unit = 's')
    peth_all = []
    for j in range(len(tmp)):
        #if len(tmp[j]) >= 400: #TODO: Fix this - don't hard code
        peth_all.append(tmp[j].as_series())
    peth['all'] = pd.concat(peth_all, axis = 1, join = 'outer')
    peth['mean'] = peth['all'].mean(axis = 1)
    return peth


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

    # wake_bin = 0.2 #200ms binwidth
    
    # ep = full_ang(angle.time_support, angle, 60)
      
    # dx = position['x'].bin_average(wake_bin,ep)
    # dy = position['y'].bin_average(wake_bin,ep)
    # ang = angle.bin_average(wake_bin, ep)
    # v = vel.bin_average(wake_bin,ep)

    # v = v.threshold(2)
       
    # ang = ang.loc[v.index.values]
    
    # wake_count = spikes[hd].count(wake_bin, ep) 
    # wake_count = wake_count.loc[v.index.values]
           
    # wake_count = wake_count.as_dataframe()
    # wake_rate = np.sqrt(wake_count/wake_bin)
    # wake_rate = wake_rate.rolling(window = 50, win_type = 'gaussian', center = True, min_periods = 1, axis = 0).mean(std = 3)
        
    # samples = sample(list(np.arange(0,len(ep)-1)), round(len(ep)/5))
    # samples = np.sort(samples)
    
    # sub_ep = nap.IntervalSet(start = ep.iloc[samples]['start'], end = ep.iloc[samples]['end'])    
    
    # ang = ang.restrict(sub_ep)
    # ang = ang.dropna()
      
    # wake_rate = wake_rate.loc[ang.index.values]

### Sleep binning 

    # du = nap.IntervalSet(start = up_ep['start'] - 0.025, end = up_ep['start'] + 0.15) 
    
    # sleep_dt = 0.025 
    # sleep_binwidth = 0.1

    # num_overlapping_bins = int(sleep_binwidth/sleep_dt)    
    
    # sleep_count = spikes[hd].count(sleep_dt, du.loc[[0]]) 
    # sleep_count = sleep_count.as_dataframe()
    # sleep_rate = np.sqrt(sleep_count/sleep_dt)
    # sleep_rate = sleep_rate.rolling(window = num_overlapping_bins, win_type = 'gaussian', center = True, min_periods = 1, axis = 0).mean(std = 3)
        
    # rate = np.vstack([wake_rate.values, sleep_rate.values])
               
    # projection = Isomap(n_components = 2, n_neighbors = 20).fit_transform(rate)
        
    # H = ang.values/(2*np.pi)
    # HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T
    # RGB = hsv_to_rgb(HSV)    
       
    # cmap = plt.colormaps['Greys']
    # col = np.linspace(0,1,len(projection[:,0][len(wake_rate)+1:]))
    
    
    # dx = np.diff(projection[:,0][len(wake_rate)+1:])
    # dy = np.diff(projection[:,1][len(wake_rate)+1:])
    
    # plt.figure(figsize = (8,8))
    # plt.scatter(projection[:,0][0:len(wake_rate)], projection[:,1][0:len(wake_rate)], c = RGB)
    
    # for i in range(len(projection[:,0][len(wake_rate)+1:])):
    #     plt.plot(projection[:,0][i], projection[:,1][i], 'o-', color = cmap(col)[i])
        
    
    
    # plt.xticks([])
    # plt.yticks([])
    
    
#%% Analysis of radial and angular velocity 
   
### Wake binning 

    wake_bin =  0.2 #0.4 #400ms binwidth
    
    ep = full_ang(angle.time_support, angle, 60)
      
    dx = position['x'].bin_average(wake_bin,ep)
    dy = position['y'].bin_average(wake_bin,ep)
    ang = angle.bin_average(wake_bin, ep)
    v = vel.bin_average(wake_bin,ep)

    v = v.threshold(2)
       
    ang = ang.loc[v.index.values]
    
    wake_count = spikes.count(wake_bin, ep) 
    wake_count = wake_count.loc[v.index.values]
           
    wake_count = wake_count.as_dataframe()
    wake_rate = np.sqrt(wake_count/wake_bin)
    wake_rate = wake_rate.rolling(window = 50, win_type = 'gaussian', center = True, min_periods = 1, axis = 0).mean(std = 3)
        
    ang = ang.dropna()
      
    wake_rate = wake_rate.loc[ang.index.values]
    
    wake_rate = wake_rate.iloc[0:3000]

### Sleep binning 

    tokeep = np.where((up_ep['end'] - up_ep['start']) > 0.5)
    
    du = nap.IntervalSet(start = up_ep.iloc[tokeep]['start'] - 0.025, end = up_ep.iloc[tokeep]['start'] + 0.5) 
    
    sleep_dt = 0.01 #0.015 #0.025 #0.015 
    sleep_binwidth = 0.05 #0.1 #0.03

    num_overlapping_bins = int(sleep_binwidth/sleep_dt)    
    
    sleep_count = spikes.count(sleep_dt,du) 
    sleep_count = sleep_count.as_dataframe()
    sleep_rate = np.sqrt(sleep_count/sleep_dt)
    sleep_rate = sleep_rate.rolling(window = num_overlapping_bins, win_type = 'gaussian', center = True, min_periods = 1, axis = 0).mean(std = 3)
        
    rate = np.vstack([wake_rate.values, sleep_rate.iloc[0:100].values])
               
    fit = Isomap(n_components = 2, n_neighbors = 200).fit(rate) 
    p_wake = fit.transform(wake_rate)    
    p_sleep = fit.transform(sleep_rate)

    H = ang.values/(2*np.pi)
    HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T
    RGB = hsv_to_rgb(HSV)    
    
    truex = p_sleep[np.where(sleep_rate.sum(axis=1)==0)][0][0]
    truey = p_sleep[np.where(sleep_rate.sum(axis=1)==0)][0][1]
    
    p_wake = p_wake - [truex, truey]
    p_sleep = p_sleep - [truex, truey]
                
    projection = nap.TsdFrame(t = sleep_rate.index.values, d = p_sleep, columns = ['x', 'y'])
    
##Angular direction, radius and velocity     
    angdir = nap.Tsd(t = projection.index.values, d = np.arctan2(projection['y'].values, projection['x'].values))
    radius = nap.Tsd(t = projection.index.values, 
                     d = np.sqrt((projection['x'].values**2) 
                     + (projection['y'].values**2)))
    
    angdiff = (angdir + 2*np.pi)%(2*np.pi)
    angdiff = np.unwrap(angdiff)
    
    angs = pd.Series(index = projection.index.values, data = angdiff)
    angs.rolling(window = 20, win_type='gaussian', center=True, min_periods=1).mean(std=1)
    
    # angdiff = np.minimum((2*np.pi - abs(angdiff.values)), abs(angdiff.values))
    
    angvel = nap.Tsd(t = projection.index.values, d = np.abs(np.gradient(angs.values)))    
    # radvel = nap.Tsd(t = projection.index.values[:-1], d = (np.diff(radius.values)))
    
    peth_radius = perievent_Tsd(radius, nap.Tsd(up_ep.iloc[tokeep]['start'].values), (-0.025, 0.5))
    peth_radius['all'] = peth_radius['all'][0:0.48]
    peth_radius['mean'] = peth_radius['mean'][0:0.48]
        
    peth_angvel = perievent_Tsd(angvel, nap.Tsd(up_ep.iloc[tokeep]['start'].values), (-0.025, 0.5))
    peth_angvel['all'] = peth_angvel['all'][0:0.48]
    peth_angvel['mean'] = peth_angvel['mean'][0:0.48]
   
   #%%  
   
    plt.figure(figsize = (20,10))
    plt.subplot(131)
    plt.scatter(p_wake[:,0], p_wake[:,1], c = RGB[0:3000], zorder = 2)
    plt.scatter(p_sleep[:,0], p_sleep[:,1], c = 'silver') 
   
    plt.subplot(132)
    plt.plot(peth_radius['all'], color = 'silver', linewidth = 0.5) 
    plt.plot(peth_radius['mean'], color = 'k', linewidth = 2)
   
    plt.subplot(133)
    plt.plot(peth_angvel['all'], color = 'silver', linewidth = 0.5) 
    plt.plot(peth_angvel['mean'], color = 'k', linewidth = 2)

    #%% 
    
    
    cmap = plt.colormaps['Greys']

### Select Individual examples and plot them 
    
    trajmatrix_x = pd.DataFrame()
    trajmatrix_y = pd.DataFrame()
        
    examples = [0,1,4,7,9,69,420, 616, 666, 786]
    tokeep = []
        
    for k in examples: #range(len(du)):  

        traj = projection.restrict(nap.IntervalSet(start = du.loc[[k]]['start'], 
                                                   end = du.loc[[k]]['end']))
        traj.index = traj.index.values - (du.loc[[k]]['start'].values + 0.025)
        traj.index = [round (j,4) for j in traj.index.values]                               
                                   
        traj = traj.as_dataframe()
                                
        trajmatrix_x = pd.concat([trajmatrix_x, traj['x']],axis = 1)
        trajmatrix_y = pd.concat([trajmatrix_y, traj['y']],axis = 1)
  
    
  
    
#%% 
        ### Plotting 
        exl = len(traj)
        col = np.linspace(0,1, exl)
        dx = np.diff(traj['x'].values)
        dy = np.diff(traj['y'].values)
        
        tuning_curves = nap.compute_1d_tuning_curves(group=spikes[hd], 
                                                     feature=position['ang'],                                              
                                                     nb_bins=31, 
                                                     minmax=(0, 2*np.pi))
        pref_ang = []
         
        for i in tuning_curves.columns:
           pref_ang.append(tuning_curves.loc[:,i].idxmax())

        norm = plt.Normalize()        
        color = plt.cm.hsv(norm([i/(2*np.pi) for i in pref_ang]))
        
    
        

        plt.figure(figsize = (8,8))
        plt.subplot(141)
        plt.scatter(p_wake[:,0], p_wake[:,1], c = RGB[0:3000], zorder = 2)
        plt.scatter(p_sleep[:,0], p_sleep[:,1], c = 'silver') 
        
        for i in range(exl-1):
            plt.plot(traj['x'].iloc[i], traj['y'].iloc[i], 'o-', color = cmap(col)[i])
        
        for i in range(exl-2):
            plt.arrow(traj['x'].iloc[i], traj['y'].iloc[i],
                  dx[i], dy[i], color = cmap(col)[i],
                  head_width = 0.1, head_length = 0.1, linewidth = 4)
        
            
        plt.subplot(142)
        plt.plot(peth_radius['all'][k], color = 'silver') 
               
        plt.subplot(143)
        plt.plot(peth_angvel['all'][k], color = 'silver')
        
        plt.subplot(144)
        for i,n in enumerate(spikes[hd].keys()):
            plt.plot(spikes[hd][n].restrict(du.loc[[k]]).fillna(pref_ang[i]), '|',color = color[i])
            plt.axvline(du.loc[[k]]['start'][0] + 0.025, color = 'r')
        
        
        
        