#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 13:44:57 2023

@author: dhruv
"""

import numpy as np 
import pandas as pd 
import scipy.io
import pynapple as nap 
import os, sys
import matplotlib.pyplot as plt 
import scipy.signal
from math import sin, cos
from matplotlib.colors import hsv_to_rgb
from random import uniform, sample

#%% 

data_directory = '/media/dhruv/LaCie1/PoSub-UPState/Data/###AllPoSub'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')
# datasets = np.genfromtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')

rwpath = '/media/dhruv/LaCie1/PoSub-UPState/Project/Data'

crosstimes = []
alldiffs = []
angvars = []
stats = []
UDang_all = [] 
DUang_all = [] 
relang_all = []
minvals_all = pd.DataFrame()
maxvals_all =  pd.DataFrame()

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
            
#%% LOAD UP AND DOWN STATE
        
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

#%% 

    RGB = np.load(rawpath + '/' + s + '_RGB.npy')
    p_wake = np.load(rawpath + '/' + s + '_pwake.npy')
    
    projection = nap.TsdFrame(pd.read_pickle(rawpath + '/' + s + '_projection.pkl'))
    
    wake_radius = np.sqrt(p_wake[:,0]**2 + p_wake[:,1]**2)
    ringradius = np.mean(wake_radius)
    
    # plt.figure()
    # plt.scatter(p_wake[:,0], p_wake[:,1], color = RGB)

#%% 
   
    angdir = nap.Tsd(t = projection.index.values, d = np.arctan2(projection['y'].values, projection['x'].values))
    
    radius = nap.Tsd(t = projection.index.values, 
                        d = np.sqrt((projection['x'].values**2) 
                      + (projection['y'].values**2)))
    
    bins = np.arange(0, np.ceil(radius.max()), 1)
    
    counts, bins = np.histogram(radius,bins)
    # radius_index = scipy.signal.argrelmin(counts)
    
    # if len(radius_index) > 1:
    #     ringradius = int(bins[radius_index][0])
    # else: ringradius = 5
           
    # plt.figure()
    # plt.plot(counts)
    # plt.axvline(ringradius/3, color = 'k')
    
#%%
    
    tokeep = np.where((up_ep['end'] - up_ep['start']) > 0.5)
    longdu = nap.IntervalSet(start = up_ep.iloc[tokeep]['start'] - 0.25, end = up_ep.iloc[tokeep]['start'] + 0.5) 
    
    
    goodeps = []
    sess_angdiffs = []
    DUang = []
    UDang = []
    
    # plt.figure()    
    # plt.scatter(p_wake[:,0], p_wake[:,1], c = RGB)
    
    examples = [48, 580, 12, 264, 445] 
    
    for k in range(len(longdu)): #examples:
    
        traj = projection.restrict(nap.IntervalSet(start = longdu.loc[[k]]['start'], 
                                                    end = longdu.loc[[k]]['end']))
        traj.index = traj.index.values - (longdu.loc[[k]]['start'].values + 0.25)
        traj.index = [round (j,4) for j in traj.index.values]                               
                                   
        traj = traj.as_dataframe()
                  
        vec = traj[-0.25:0.5]
        vx = vec[0:0.15]
                
        tr = nap.Tsd(t = vec. index. values, d = np.sqrt((vec['x'].values**2) + (vec['y'].values**2)))
                      
        ix = np.where(tr[0:0.1].values > (ringradius/3))[0]
        iy = np.where(tr[-0.25:0].values > (ringradius/3))[0]
        
        if (len(ix) > 0) and (len(iy) > 0):
                       
            ix1 = ix[0]
            crosstimes.append(vx.index[ix1])
            winlength = len(vx[vx.index[ix1]:0.15])
            
            goodeps.append(k)     
            
            theta = np.arctan2(vx['y'].iloc[ix1], vx['x'].iloc[ix1]) 
            DUang.append(theta)
            DUang_all.append(theta)
            UDang.append(np.arctan2(vec['y'][-0.25:0].iloc[iy[-1]], vec['x'][-0.25:0].iloc[iy[-1]]))
            UDang_all.append(np.arctan2(vec['y'][-0.25:0].iloc[iy[-1]], vec['x'][-0.25:0].iloc[iy[-1]]))
        
            angs = pd.DataFrame(data = [UDang, DUang], index = ['UD', 'DU']).T
            diffs = np.abs(angs['UD'] - angs['DU'])
            relang = np.minimum((2*np.pi - abs(diffs)), abs(diffs))
            
            
#%% Bootstrapping
            
    bins = np.linspace(0, np.pi, 30)
    xcenters = np.mean(np.vstack([bins[:-1], bins[1:]]), axis = 0)
    
    allcounts = pd.DataFrame()

    for i in range(1000):
        DUshu = sample(DUang, len(DUang))
        shuffdiff = np.abs(angs['UD'] - DUshu)
        shuff_rel = np.minimum((2*np.pi - abs(shuffdiff)), abs(shuffdiff))
        
        counts, bins = np.histogram(shuff_rel,bins)
        p_shu = counts/sum(counts)
        allcounts = pd.concat([allcounts, pd.Series(p_shu)], axis = 1)
        
    maxvals = allcounts.max(axis = 1)
    maxvals_all = pd.concat([maxvals_all, maxvals], axis = 1)
    minvals = allcounts.min(axis = 1)
    minvals_all = pd.concat([minvals_all, minvals], axis = 1)
    
    relcounts,_ = np.histogram(relang, bins)     
    p_rel = relcounts/sum(relcounts)
    
    # plt.figure()
    # plt.title(s)                       
    # plt.stairs(p_rel,bins, linewidth =  2, color = 'mediumorchid')
    # plt.hlines(minvals, xmin = bins[0:-1], xmax = bins[1:], color = 'k')
    # plt.hlines(maxvals, xmin = bins[0:-1], xmax = bins[1:],  color = 'r')
            
#%%

angs = pd.DataFrame(data = [UDang_all, DUang_all], index = ['UD', 'DU']).T
diffs = np.abs(angs['UD'] - angs['DU'])
relang = np.minimum((2*np.pi - abs(diffs)), abs(diffs))
relang_all.append(relang)

bins = np.linspace(0, np.pi, 30)

allcounts = pd.DataFrame()

# for i in range(1000):
#     DUshu = sample(DUang_all, len(DUang_all))
#     shuffdiff = np.abs(angs['UD'] - DUshu)
#     shuff_rel = np.minimum((2*np.pi - abs(shuffdiff)), abs(shuffdiff))
    
#     counts, bins = np.histogram(shuff_rel,bins)
#     p_counts = counts/sum(counts)
#     allcounts = pd.concat([allcounts, pd.Series(p_counts)], axis = 1)
    
maxvals = maxvals_all.mean(axis = 1)
minvals = minvals_all.mean(axis = 1)    

relcounts_all,_ = np.histogram(relang_all, bins)     
p_rel = relcounts_all/sum(relcounts_all)
     
plt.figure()
plt.stairs(p_rel,bins, linewidth =  2, color = 'rosybrown')
plt.fill_between(0.5 * (bins[1:] + bins[:-1]), p_rel- minvals, p_rel + (maxvals - p_rel), color = 'rosybrown', alpha = 0.2)
plt.xlabel('Ang diff between DU and UD (rad)')
plt.ylabel('% events')
             
            
                
             
          