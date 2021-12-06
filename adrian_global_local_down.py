#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 11:05:02 2021

@author: dhruv
"""
import numpy as np 
import pandas as pd 
import scipy.io
from functions import *
from wrappers import *
import ipyparallel
import os, sys
import neuroseries as nts 
import time 
import matplotlib.pyplot as plt 
from Wavelets import MyMorlet as Morlet
from scipy.stats import pearsonr 
from scipy.stats import wilcoxon

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
datasets = np.loadtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
# datasets = np.loadtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'

means = []
# means_shu = []

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
    
     
############################################################################################### 
############################################################################################### 
    data = pd.DataFrame()   
        
       
    data['depth'] = np.reshape(depth,(len(spikes.keys())),)
    data['level'] = pd.cut(data['depth'],2, precision=0, labels=[1,0]) #0 is dorsal, 1 is ventral
    data['celltype'] = np.nan
    data['gd'] = 0
    
    for i in range(len(spikes)):
        if celltype['gd'][i] == 1:
            data.loc[i,'gd'] = 1
            
    data = data[data['gd'] == 1]
    
    #CONTROL: Use every other cell
    # data = data.iloc[::2,:]
    
    # for i in range(len(spikes)):
    #     if celltype['ex'][i] == 1 and celltype['gd'][i] == 1:
    #         data.loc[i,'celltype'] = 'ex' #0 for excitatory
    #     elif celltype['fs'][i] == 1 and celltype['gd'][i] == 1:
    #         data.loc[i,'celltype'] = 'fs' #1 for inhibitory
    
    # data = data[data['celltype'] == 'ex'] #Doing it for each cell type separately 
    # data = data[data['celltype'] == 'fs']
    
    epoch = pd.read_pickle(rawpath + '/' + name + '_epoch.pkl')
    bin_size = 10000 #us    

########################################################################################################   
#FIND MUA THRESHOLD CROSSING FOR DORSAL AND VENTRAL        
########################################################################################################  
    
    mua = {}
    
    latency_dorsal = []
    latency_ventral = []
    
    # define mua for dorsal and ventral
    for i in range(2):
        mua[i] = []        
        for n in data[data['level'] == i].index:            
            mua[i].append(spikes[n].index.values)
        mua[i] = nts.Ts(t = np.sort(np.hstack(mua[i])))
    
    adt = []
    idx = []
    for j in epoch.index.values:                
        ep = epoch.loc[[j]]
        bins = np.arange(ep.loc[j,'start'], ep.loc[j,'end'], bin_size)        
        r = np.array([np.histogram(mua[i].restrict(ep).index.values, bins)[0] for i in range(2)])
        r = pd.DataFrame(index = bins[0:-1] + np.diff(bins)/2, data = r.T)            
        r2 = r.rolling(window=30,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2)
        r3 = r2.values
        ix = r3 > np.percentile(r3, 20, 0)
        if (~ix).all(0).sum() == 0:            
            t = np.array([r2.index.values[np.where(ix[:,i])[0][0]] for i in range(2)])
            dt = t - ep.loc[j,'start']
            adt.append(dt)
            idx.append(j)
        
    adt = pd.DataFrame(index = idx, data = adt)
    
    diff = np.zeros((len(adt),1))
    
    for i in range(len(adt)): 
        diff[i] = (adt.iloc[i][1] - adt.iloc[i][0])/1000        
        
    z_statistic, p_value = wilcoxon(diff.flatten()-0)    
    means.append(np.mean(diff))
    
    # plt.figure()
    # bin_size = 40
    # plt.subplot(121)
    # plt.title('Histogram of latencies_' + s)
    # plt.xlabel('Latency from epoch start (ms)')
    # plt.ylabel('Number of threshold crossings')
    # x = np.arange(0, np.maximum(max(adt[0]/1000), max(adt[1]/1000)), bin_size)
    # d = np.histogram(adt[0]/1000, x)[0]
    # v = np.histogram(adt[1]/1000, x)[0]
    # plt.bar(x[0:-1], d, width = bin_size/2, label = 'dorsal')
    # plt.bar(x[0:-1] + np.diff(x)/2, v, width = bin_size/2, label = 'ventral')
    # plt.legend()
    # plt.subplot(122)
    # plt.title('Cumulative plot_' + s)
    # plt.xlabel('Latency from epoch start (ms)')
    # plt.ylabel('Number of threshold crossings')
    # x2 = x[0:-1]
    # plt.plot(x2, np.cumsum(d), label = 'dorsal')
    # plt.plot(x2, np.cumsum(v), label = 'ventral')
    # plt.legend()    
    
    rge = np.arange(min(diff)[0],max(diff)[0],20)
    plt.figure()
    plt.title('(Ventral - Dorsal) latency (ms)_' + s)
    plt.xlabel('Difference (ms)')
    plt.ylabel('Number of epochs')
    plt.hist(diff,rge, label ='Mean =' +  str(np.mean(diff)))
    plt.legend()
    
    
        
        
    
########################################################################################################   
#SHUFFLE      
########################################################################################################   
    # dshu = data.copy()
    # np.random.shuffle(dshu['level'])
    
    # mua_shu = {}
    
    # ld_shu = []
    # lv_shu = []
    
    # # define mua for dorsal and ventral
    # for i in range(2):
    #     mua_shu[i] = []        
    #     for n in dshu[dshu['level'] == i].index:            
    #         mua_shu[i].append(spikes[n].index.values)
    #     mua_shu[i] = nts.Ts(t = np.sort(np.hstack(mua_shu[i])))
    
  
    # dt_shu = []
    # ix_shu = []
    
    # for j in epoch.index.values:                
    #     ep = epoch.loc[[j]]
    #     bins = np.arange(ep.loc[j,'start'], ep.loc[j,'end'], bin_size)        
    #     r = np.array([np.histogram(mua_shu[i].restrict(ep).index.values, bins)[0] for i in range(2)])
    #     r = pd.DataFrame(index = bins[0:-1] + np.diff(bins)/2, data = r.T)            
    #     r2 = r.rolling(window=30,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2)
    #     r3 = r2.values
    #     ix = r3 > np.percentile(r3, 20, 0)
    #     if (~ix).all(0).sum() == 0:            
    #         t = np.array([r2.index.values[np.where(ix[:,i])[0][0]] for i in range(2)])
    #         dt = t - ep.loc[j,'start']
    #         dt_shu.append(dt)
    #         ix_shu.append(j)
        
    # dt_shu = pd.DataFrame(index = ix_shu, data = dt_shu)
    
    # diff_shu = np.zeros((len(dt_shu),1))
    
    # for i in range(len(dt_shu)): 
    #     diff_shu[i] = (dt_shu.iloc[i][1] - dt_shu.iloc[i][0])/1000        
        
    # z_shu, p_shu = wilcoxon(diff_shu.flatten()-0)    
    # means_shu.append(np.mean(diff_shu))
    
    # plt.figure()
    # bin_size = 40
    # plt.subplot(121)
    # plt.title('Histogram of shuffled latencies_' + s)
    # plt.xlabel('Latency from epoch start (ms)')
    # plt.ylabel('Number of threshold crossings')
    # x = np.arange(0, np.maximum(max(dt_shu[0]/1000), max(dt_shu[1]/1000)), bin_size)
    # d = np.histogram(dt_shu[0]/1000, x)[0]
    # v = np.histogram(dt_shu[1]/1000, x)[0]
    # plt.bar(x[0:-1], d, width = bin_size/2, label = 'dorsal')
    # plt.bar(x[0:-1] + np.diff(x)/2, v, width = bin_size/2, label = 'ventral')
    # plt.legend()
    # plt.subplot(122)
    # plt.title('Cumulative plot_' + s)
    # plt.xlabel('Latency from epoch start (ms)')
    # plt.ylabel('Number of threshold crossings')
    # x2 = x[0:-1]
    # plt.plot(x2, np.cumsum(d), label = 'dorsal')
    # plt.plot(x2, np.cumsum(v), label = 'ventral')
    # plt.legend()    
    
    # rge = np.arange(min(diff_shu)[0],max(diff_shu)[0],20)
    # plt.figure()
    # plt.title('(Ventral - Dorsal) shuffled latency (ms)_' + s)
    # plt.xlabel('Difference (ms)')
    # plt.ylabel('Number of epochs')
    # plt.hist(diff_shu,rge, label ='Mean =' +  str(np.mean(diff_shu)))
    # plt.legend()