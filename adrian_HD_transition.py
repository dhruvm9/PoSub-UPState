#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 17:29:03 2022

@author: dhruv
"""
#loading the dataset
import numpy as np 
import pandas as pd 
import scipy.io
import os, sys
import pynapple as nap
import time 
import matplotlib.pyplot as plt 
from scipy.stats import kendalltau, pearsonr, wilcoxon
from functions import *
from wrappers import *
from matplotlib.colors import hsv_to_rgb

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
datasets = np.loadtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
# datasets = np.loadtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'

n_pyr = []
n_int = []

for s in datasets:
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    rawpath = os.path.join(rwpath,s)
   
    data = nap.load_session(rawpath, 'neurosuite')
    data.load_neurosuite_xml(rawpath)
    
#%%    
# ############################################################################################### 
#     # LOAD DATA
# ############################################################################################### 
    spikes = data.spikes  
    epochs = data.epochs    

    filepath = os.path.join(path, 'Analysis')
    listdir    = os.listdir(filepath)

    file = [f for f in listdir if 'CellDepth' in f]
    celldepth = scipy.io.loadmat(os.path.join(filepath,file[0]))
    depth = celldepth['cellDep']

    file = [f for f in listdir if 'MeanFR' in f]
    mfr = scipy.io.loadmat(os.path.join(filepath,file[0]))
    r_wake = mfr['rateS']

    filepath = os.path.join(path, 'Analysis')
    data = pd.read_csv(filepath + '/Tracking_data.csv', header = None)
    position = pd.DataFrame(index = data[0].values, data = data[[1,2,3]].values, columns=['x', 'y', 'ang'])
    position = position.loc[~position.index.duplicated(keep='first')]
    position['ang'] = position['ang'] *(np.pi/180) #convert degrees to radian
    position['ang'] = (position['ang'] + 2*np.pi) % (2*np.pi) #convert [-pi, pi] to [0, 2pi]
    position = nap.TsdFrame(position) 
    
    w1 = position.restrict(epochs['wake'].loc[[0]])
    w2 = position.restrict(epochs['wake'].loc[[1]])
    
    starts = [w1.index[0], w2.index[1]]
    ends = [w1.index[-1], w2.index[-1]]
    
    wake_ep = nap.IntervalSet(start = starts, end = ends)
    
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

    n_pyr.append(len(pyr))
    n_int.append(len(interneuron))
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
      
# ############################################################################################### 
#     # PLOT RASTER FOR DU TRANSITIONS
# ###############################################################################################   

    tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ang'], 60, ep = wake_ep)
    hdcurves = tuning_curves[hd]
    hdcurves = smoothAngularTuningCurves(hdcurves)
         
    
    plt.figure()
    for j, n in enumerate(hdcurves.columns):
        plt.subplot(2,10,j+1, projection = 'polar')
        plt.plot(hdcurves[n])
        # plt.subplots_adjust(wspace=0.4, hspace=1, top = 0.85)
    plt.show()
    
    pref_ang = []
    
    for i in hdcurves.columns:
        pref_ang.append(hdcurves.loc[:,i].idxmax())
    
    hd_data = pd.DataFrame(columns = ['cell', 'prefang'], data = list(zip(hd, pref_ang)))
            
    HSV = np.vstack((hd_data['prefang']/(2*np.pi), np.ones_like(hd_data['prefang']), np.ones_like(hd_data['prefang']))).T
    RGB = hsv_to_rgb(HSV)
          
    cols = []
    for i in hd_data.index.values:
       tmp = RGB[i]
       cols.append(tmp)
    hd_data['color'] = cols
    
    hd_data['depth'] = depth[hd]
    hd_data = hd_data.sort_values(by=['depth'], ascending = False)
    
    plt.figure(figsize = (20,20))
    for j, n in enumerate(hd_data['cell'].values):
        plt.subplot(12,10,j+1, projection = 'polar')
        plt.plot(hdcurves[n], color = hd_data.loc[hd_data['cell'] == n, 'color'].values[0])
        # plt.subplots_adjust(wspace=0.4, hspace=1, top = 0.85)
    plt.show()
    
    per = nap.IntervalSet(start = 1251.288000, end = 1251.688000) #20min 51s 288ms
    fig, ax = plt.subplots()
    
    for i in hd_data['cell'].values:
        print(i)
        a = spikes[i].restrict(per)
        plt.plot(a.fillna(i), '|', color = hd_data.loc[hd_data['cell'] == i, 'color'].values[0], markersize = 10)
        
   
    
    
    for i in range(3):
        ep = nap.IntervalSet(start = up_ep['start'][i] - 0.5, end = up_ep['start'][i] + 1)
        plt.figure()
        
        for j in hd_data['cell'].values:
            # print(i)
            a = spikes[j].restrict(ep)
            plt.plot(a.fillna(j), '|', color = hd_data.loc[hd_data['cell'] == j, 'color'].values[0], markersize = 10)
        plt.axvline(up_ep['start'][i], color = 'k')
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
               
        plt.pause(5)
        
        
    