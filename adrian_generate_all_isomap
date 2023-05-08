#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 09:21:03 2023

@author: Dhruv
"""

import numpy as np 
import pandas as pd 
import scipy.io
import pynapple as nap 
import os, sys
import matplotlib.pyplot as plt 
from matplotlib.colors import hsv_to_rgb
import scipy.signal
from sklearn.manifold import Isomap

#%% 

# data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
data_directory = '/media/adrien/LaCie/PoSub-UPState/Data/###AllPoSub'
# datasets = np.genfromtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')

rwpath = '/media/adrien/LaCie/PoSub-UPState/Project/Data'

alltruex = []
alltruey = []

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
    
#%%  Wake binning 
   
    wake_bin =  0.2 #0.4 #400ms binwidth
    
    ep = epochs['wake']
      
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
        
    ang = ang.dropna()
    ang = ang.rename('ang')
      
    wake_rate = wake_rate.loc[ang.index.values]
        
    wake_rate = pd.concat([wake_rate, pd.DataFrame(ang)], axis = 1)
    wake_rate = wake_rate.sample(frac = 0.5).sort_index()

#%% Sleep binning 
    
    sleep_dt = 0.01 #0.015 #0.025 #0.015 
    sleep_binwidth = 0.05 #0.1 #0.03

    tokeep = np.where((up_ep['end'] - up_ep['start']) > 0.5)
    
    du = nap.IntervalSet(start = up_ep.iloc[tokeep]['start'] - 0.25, end = up_ep.iloc[tokeep]['start'] + 0.25) 
    longdu = nap.IntervalSet(start = up_ep.iloc[tokeep]['start'] - 0.25, end = up_ep.iloc[tokeep]['start'] + 0.5) 
        
    num_overlapping_bins = int(sleep_binwidth/sleep_dt)    
        
    sleep_count = spikes[hd].count(sleep_dt,longdu) 
    sleep_count = sleep_count.as_dataframe()
    sleep_rate = np.sqrt(sleep_count/sleep_dt)
    sleep_rate = sleep_rate.rolling(window = num_overlapping_bins, win_type = 'gaussian', center = True, min_periods = 1, axis = 0).mean(std = 3)
    
    fit_rate = nap.TsdFrame(sleep_rate).restrict(du)
       
    rate = np.vstack([wake_rate.loc[:, wake_rate.columns != 'ang'].values, fit_rate.as_dataframe().sample(frac = 0.01).values]) #Take 1000 random values 
               
    fit = Isomap(n_components = 2, n_neighbors = 200).fit(rate) 
    p_wake = fit.transform(wake_rate.loc[:, wake_rate.columns != 'ang'])    
    p_sleep = fit.transform(sleep_rate)
    

    H = wake_rate['ang'].values/(2*np.pi)
    HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T
    RGB = hsv_to_rgb(HSV)    
    
    truex = p_sleep[np.where(sleep_rate.sum(axis=1)==0)][0][0]
    truey = p_sleep[np.where(sleep_rate.sum(axis=1)==0)][0][1]
    
    alltruex.append(truex)
    alltruey.append(truey)
    
    p_wake = p_wake - [truex, truey]
    p_sleep = p_sleep - [truex, truey]
                    
    projection = nap.TsdFrame(t = sleep_rate.index.values, d = p_sleep, columns = ['x', 'y'])
    projection.to_pickle(rawpath + '/' + s + '_projection.pkl')  
    
    np.save(rawpath + '/' + s + '_pwake.npy',p_wake)
    np.save(rawpath + '/' + s + '_RGB.npy', RGB)      
    
#%% 

np.save(rwpath + '/alltruex.npy' , alltruex)
np.save(rwpath + '/alltruey.npy' , alltruey)

