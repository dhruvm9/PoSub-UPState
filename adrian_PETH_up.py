#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 10:12:29 2021

@author: dhruv
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:21:34 2020

@author: Dhruv
"""

#loading the dataset
import numpy as np 
import pandas as pd 
import scipy.io
from functions import *
from wrappers import *
import os, sys
import neuroseries as nts 
import time 
import matplotlib.pyplot as plt 
from Wavelets import MyMorlet as Morlet
from scipy.stats import kendalltau, pearsonr, wilcoxon

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
#datasets = np.loadtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
datasets = np.loadtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'

allcoefs_up = []
allcoefs_dn = []
alldurs = []
updurs = []

for s in datasets:
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    rawpath = os.path.join(rwpath,s)

    spikes, shank = loadSpikeData(path)
    n_channels, fs, shank_to_channel = loadXML(rawpath)

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

# ############################################################################################### 
#     # LOAD UP AND DOWN STATE, NEW SWS AND NEW WAKE EPOCHS
# ###############################################################################################   

    
    file = os.path.join(rawpath, name +'.evt.py.dow')
    if os.path.exists(file):
        tmp = np.genfromtxt(file)[:,0]
        tmp = tmp.reshape(len(tmp)//2,2)/1000
        down_ep = nts.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
    
    file = os.path.join(rawpath, name +'.evt.py.upp')
    if os.path.exists(file):
        tmp = np.genfromtxt(file)[:,0]
        tmp = tmp.reshape(len(tmp)//2,2)/1000
        up_ep = nts.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
    
    file = os.path.join(rawpath, name +'.DM.new_sws.evt')
    if os.path.exists(file):
        tmp = np.genfromtxt(file)[:,0]
        tmp = tmp.reshape(len(tmp)//2,2)/1000
        new_sws_ep = nts.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
    
    file = os.path.join(rawpath, name +'.DM.new_wake.evt')
    if os.path.exists(file):
        tmp = np.genfromtxt(file)[:,0]
        tmp = tmp.reshape(len(tmp)//2,2)/1000
        new_wake_ep = nts.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')

############################################################################################### 
    # COMPUTE EVENT CROSS CORRS
###############################################################################################  
                 
    binsize = 5
    nbins = 1000        
    neurons = list(spikes.keys())
    times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2        
    cc = pd.DataFrame(index = times, columns = neurons)
    tsd_up = up_ep.as_units('ms').start.values
   
#UP State    
    ep_U = nts.IntervalSet(start = up_ep.start[0], end = up_ep.end.values[-1])
                   
    rates = []
    
    
    for i in neurons:
        # spk2 = spikes[i].restrict(ep_U).as_units('ms').index.values
        spk2 = spikes[i].restrict(new_sws_ep).as_units('ms').index.values
        tmp = crossCorr(tsd_up, spk2, binsize, nbins)
       
        tmp = pd.DataFrame(tmp)
        tmp = tmp.rolling(window=4, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
        # fr = len(spk2)/ep_U.tot_length('s')
        fr = len(spk2)/new_sws_ep.tot_length('s')
        rates.append(fr)
    
        cc[i] = tmp.values
        cc[i] = tmp.values/fr
     
        dd = cc[-50:250]
    
    #Cell types 
    # ee = dd[pyr]
    # ee = dd[interneuron]
    
    n = len(depth)
    # tmp = np.argsort(depth[pyr].flatten())
    # tmp = np.argsort(depth[interneuron].flatten())
    tmp = np.argsort(depth.flatten())
    desc = tmp[::-1][:n]
    
    order = []
    # for i in range(len(pyr)): 
        # order.append(pyr[desc[i]])
    
    # for i in range(len(interneuron)): 
        # order.append(interneuron[desc[i]])
    
        
    # finalRates = ee[order]
    finalRates = dd[desc]

    
    # if len(ee.columns) > 5:
    if len(dd.columns) > 5:
        
        fig, ax = plt.subplots()
        #cax = ax.imshow(finalRates.T,extent=[-250 , 150, len(interneuron) , 1],aspect = 'auto', cmap = 'hot')
        cax = ax.imshow(finalRates.T,extent=[-50 , 250, len(neurons) , 1],aspect = 'auto', cmap = 'inferno')
        # plt.imshow(finalRates.T,extent=[-250 , 250, len(neurons) , 1],aspect = 'auto', cmap = 'hot')        
        # plt.imshow(finalRates.T,extent=[-250 , 250, len(pyr) , 1],aspect = 'auto', cmap = 'hot')        
        cbar = fig.colorbar(cax, ticks=[0, finalRates.values.max()], label = 'Norm. Firing Rate')
        cbar.ax.set_yticklabels(['0', str(round(finalRates.values.max(),2))])
        plt.title('Event-related Xcorr, aligned to UP state onset_' + s)
        ax.set_ylabel('Neuron number')
        ax.set_xlabel('Lag (ms)')
        
        
        dur = (down_ep['end'] - down_ep['start']) / 1000
        updur = (up_ep['end'] - up_ep['start']) / 1000
        updurs.append(updur)
        alldurs.append(dur)
    # sys.exit()   
alldurs = np.concatenate(alldurs)