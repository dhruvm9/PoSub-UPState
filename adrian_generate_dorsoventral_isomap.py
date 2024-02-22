#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:26:17 2024

@author: Dhruv
"""

import numpy as np 
import pandas as pd 
import scipy.io
import pynapple as nap 
import os, sys
import nwbmatic as ntm
import matplotlib.pyplot as plt 
from matplotlib.colors import hsv_to_rgb
import scipy.signal
from sklearn.manifold import Isomap

#%% 

# data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
data_directory = '/media/adrien/LaCie/PoSub-UPState/Data/###AllPoSub'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')
# datasets = np.genfromtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')

rwpath = '/media/adrien/LaCie/PoSub-UPState/Project/Data'

alltruex_D = []
alltruex_V = []

alltruey_D = []
alltruey_V = []

for s in datasets[11:]:
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    rawpath = os.path.join(rwpath,s)

    data = ntm.load_session(rawpath, 'neurosuite')
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

#%% Split into dorsal and ventral population

    spkdata = pd.DataFrame()   
    spkdata['depth'] = np.reshape(depth,(len(spikes.keys())),)
    spkdata['level'] = pd.cut(spkdata['depth'],2, precision=0, labels=[1,0]) #0 is dorsal, 1 is ventral
    spkdata['gd'] = 0
    
    for i in range(len(spikes)):
        if celltype['gd'][i] == 1:
            spkdata.loc[i,'gd'] = 1
            
    data = spkdata[spkdata['gd'] == 1]

    dorsal_spikes = spikes[spkdata.index[spkdata['level'] == 0]]
    ventral_spikes = spikes[spkdata.index[spkdata['level'] == 1]]

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
    position = position.restrict(nap.IntervalSet(epochs['wake']))
    
    angle = position['ang'].as_series().loc[vel.index.values]
    
#%%  Wake binning 
   
    wake_bin =  0.2 #0.4 #400ms binwidth
    
    dorsal_hd = np.intersect1d(hd, dorsal_spikes.index)
    ventral_hd = np.intersect1d(hd, ventral_spikes.index)
    
    ep = epochs['wake']
      
    dx = position['x'].bin_average(wake_bin,ep)
    dy = position['y'].bin_average(wake_bin,ep)
    ang = nap.Tsd(angle).bin_average(wake_bin, ep)
    v = vel.bin_average(wake_bin,ep)

    v = v.threshold(2)
       
    ang = ang.as_series().loc[v.index.values]
    
    wake_count_dorsal = dorsal_spikes[dorsal_hd].count(wake_bin, ep) 
    wake_count_ventral = ventral_spikes[ventral_hd].count(wake_bin, ep) 
        
    wake_count_dorsal = wake_count_dorsal.as_dataframe().loc[v.index.values]
    wake_count_ventral = wake_count_ventral.as_dataframe().loc[v.index.values]
           
   
    wake_rate_D = np.sqrt(wake_count_dorsal/wake_bin)
    wake_rate_V = np.sqrt(wake_count_ventral/wake_bin)
    
    
    wake_rate_D = wake_rate_D.rolling(window = 50, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 3)
    wake_rate_V = wake_rate_V.rolling(window = 50, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 3)
        
    ang = ang.dropna()
    ang = ang.rename('ang')
      
    wake_rate_D = wake_rate_D.loc[ang.index.values]
    wake_rate_V = wake_rate_V.loc[ang.index.values]
    
        
    wake_rate_D = pd.concat([wake_rate_D, pd.DataFrame(ang)], axis = 1)
    wake_rate_V = pd.concat([wake_rate_V, pd.DataFrame(ang)], axis = 1)
    
    wake_rate_D = wake_rate_D.sample(frac = 0.5).sort_index()
    wake_rate_V = wake_rate_V.sample(frac = 0.5).sort_index()
    

#%% Sleep binning 
    
    sleep_dt = 0.01 #0.015 #0.025 #0.015 
    sleep_binwidth = 0.05 #0.1 #0.03

    tokeep = np.where((up_ep['end'] - up_ep['start']) > 0.5)
    
    du = nap.IntervalSet(start = up_ep.iloc[tokeep]['start'] - 0.25, end = up_ep.iloc[tokeep]['start'] + 0.25) 
    longdu = nap.IntervalSet(start = up_ep.iloc[tokeep]['start'] - 0.25, end = up_ep.iloc[tokeep]['start'] + 0.5) 
        
    num_overlapping_bins = int(sleep_binwidth/sleep_dt)    
        
    sleep_count_D = dorsal_spikes[dorsal_hd].count(sleep_dt,longdu) 
    sleep_count_V = ventral_spikes[ventral_hd].count(sleep_dt,longdu) 
    
    sleep_count_D = sleep_count_D.as_dataframe()
    sleep_count_V = sleep_count_V.as_dataframe()
        
    sleep_rate_D = np.sqrt(sleep_count_D/sleep_dt)
    sleep_rate_V = np.sqrt(sleep_count_V/sleep_dt)
        
    sleep_rate_D = sleep_rate_D.rolling(window = num_overlapping_bins, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 3)
    sleep_rate_V = sleep_rate_V.rolling(window = num_overlapping_bins, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 3)
    
    fit_rate_D = nap.TsdFrame(sleep_rate_D).restrict(du)
    fit_rate_V = nap.TsdFrame(sleep_rate_V).restrict(du)
       
    rate_D = np.vstack([wake_rate_D.loc[:, wake_rate_D.columns != 'ang'].values, fit_rate_D.as_dataframe().sample(frac = 0.01).values]) #Take 1000 random values 
    rate_V = np.vstack([wake_rate_V.loc[:, wake_rate_V.columns != 'ang'].values, fit_rate_V.as_dataframe().sample(frac = 0.01).values]) #Take 1000 random values 
               
    fit_D = Isomap(n_components = 2, n_neighbors = 200).fit(rate_D) 
    fit_V = Isomap(n_components = 2, n_neighbors = 200).fit(rate_V) 
        
    p_wake_D = fit_D.transform(wake_rate_D.loc[:, wake_rate_D.columns != 'ang'])    
    p_wake_V = fit_V.transform(wake_rate_V.loc[:, wake_rate_V.columns != 'ang'])    
        
    p_sleep_D = fit_D.transform(sleep_rate_D)
    p_sleep_V = fit_V.transform(sleep_rate_V)
    
    H_D = wake_rate_D['ang'].values/(2*np.pi)
    H_V = wake_rate_V['ang'].values/(2*np.pi)
        
    HSV_D = np.vstack((H_D, np.ones_like(H_D), np.ones_like(H_D))).T
    HSV_V = np.vstack((H_V, np.ones_like(H_V), np.ones_like(H_V))).T    
    
    RGB_D = hsv_to_rgb(HSV_D)    
    RGB_V = hsv_to_rgb(HSV_V)    
    
    truex_D = p_sleep_D[np.where(sleep_rate_D.sum(axis=1)==0)][0][0]
    truex_V = p_sleep_V[np.where(sleep_rate_V.sum(axis=1)==0)][0][0]
        
    truey_D = p_sleep_D[np.where(sleep_rate_D.sum(axis=1)==0)][0][1]
    truey_V = p_sleep_V[np.where(sleep_rate_V.sum(axis=1)==0)][0][1]
    
    alltruex_D.append(truex_D)
    alltruex_V.append(truex_V)
    
    alltruey_D.append(truey_D)
    alltruey_V.append(truey_V)
    
    p_wake_D = p_wake_D - [truex_D, truey_D]
    p_wake_V = p_wake_V - [truex_V, truey_V]
    
    p_sleep_D = p_sleep_D - [truex_D, truey_D]
    p_sleep_V = p_sleep_V - [truex_V, truey_V]
                    
    # plt.figure()
    # plt.suptitle(s)
    # plt.subplot(121)
    # plt.title('Dorsal')
    # plt.scatter(p_wake_D[:,0], p_wake_D[:,1], color = RGB_D)
    # plt.subplot(122)
    # plt.title('Ventral')
    # plt.scatter(p_wake_V[:,0], p_wake_V[:,1], color = RGB_V)
    
    projection_D = nap.TsdFrame(t = sleep_rate_D.index.values, d = p_sleep_D, columns = ['x', 'y'])
    projection_V = nap.TsdFrame(t = sleep_rate_V.index.values, d = p_sleep_V, columns = ['x', 'y'])
    
    projection_D.as_dataframe().to_pickle(rawpath + '/' + s + '_projection_D.pkl')  
    projection_V.as_dataframe().to_pickle(rawpath + '/' + s + '_projection_V.pkl')  
    
    np.save(rawpath + '/' + s + '_pwake_D.npy',p_wake_D)
    np.save(rawpath + '/' + s + '_pwake_V.npy',p_wake_V)
    
    np.save(rawpath + '/' + s + '_RGB_D.npy', RGB_D)      
    np.save(rawpath + '/' + s + '_RGB_V.npy', RGB_V)      
    
#%% 

np.save(rwpath + '/alltruex_D.npy' , alltruex_D)
np.save(rwpath + '/alltruex_V.npy' , alltruex_V)

np.save(rwpath + '/alltruey_D.npy' , alltruey_D)
np.save(rwpath + '/alltruey_V.npy' , alltruey_V)

