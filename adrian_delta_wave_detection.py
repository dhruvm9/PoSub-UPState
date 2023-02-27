#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 10:00:04 2022

@author: dhruv
"""
import numpy as np 
import pandas as pd 
import scipy.io
from functions import *
from wrappers import *
import os, sys
import neuroseries as nts 
import time 
import matplotlib.pyplot as plt 
import pynapple as nap
from Wavelets import MyMorlet as Morlet
import seaborn as sns
from scipy.stats import wilcoxon
import matplotlib.cm as cm

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
datasets = np.loadtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')
# datasets = np.loadtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
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
    
############################################################################################### 
    # LOADING DATA
###############################################################################################
    spikes = data.spikes  
    epochs = data.epochs
    channelorder = data.group_to_channel[0]
    seq = channelorder[::8].tolist()
    
    filepath = os.path.join(path, 'Analysis')
    mouse_pos = pd.read_csv(filepath + '/Tracking_data.csv', header = None)
    position = pd.DataFrame(index = mouse_pos[0].values*1e6, data = mouse_pos[[1,2,3]].values, columns=['x', 'y', 'ang'])
    position = position.loc[~position.index.duplicated(keep='first')]
    position['ang'] = position['ang'] *(np.pi/180) #convert degrees to radian
    position['ang'] = (position['ang'] + 2*np.pi) % (2*np.pi) #convert [-pi, pi] to [0, 2pi]
    position = nts.TsdFrame(position) 
    
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

    filepath = os.path.join(path, 'Analysis')
    position = pd.read_csv(filepath + '/Tracking_data.csv', header = None)

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
    file = os.path.join(rawpath, name +'.DM.new_sws.evt')
    new_sws_ep  = data.read_neuroscope_intervals(name = 'new_sws', path2file = file)
    
    file = os.path.join(rawpath, name +'.DM.new_wake.evt')
    new_wake_ep  = data.read_neuroscope_intervals(name = 'new_wake', path2file = file)
    
    filepath = os.path.join(rwpath, name)
    listdir    = os.listdir(filepath)
    
    file = [f for f in listdir if  name + '.eeg' in f]
    filename = [name + '.eeg']
    matches = set(filename).intersection(set(file))
    
    #Channels to seq[0], seq[int(len(seq)/2)], seq[-1]
    
    if (matches == set()) is False:
        lfp = nap.load_eeg(filepath + '/' + name + '.eeg' , channel = seq[int(len(seq)/2)], n_channels = data.nChannels, frequency = 1250, precision ='int16', bytes_size = 2) 
    else: 
        lfp = nap.load_eeg(filepath + '/' + name + '.lfp' , channel = seq[int(len(seq)/2)] , n_channels = data.nChannels, frequency = 1250, precision ='int16', bytes_size = 2) 
    
    downsample =  5
    lfp = lfp[::downsample]
    lfp_filt_delta = nap.Tsd(lfp.index.values, butter_bandpass_filter(lfp, 0.5, 4, 1250/5, 2))
        
    
    # a,_ = scipy.signal.find_peaks(lfp_filt_delta.restrict(new_sws_ep).values, 2.5*np.std(lfp_filt_delta.restrict(new_sws_ep).values))
    a,_ = scipy.signal.find_peaks(lfp_filt_delta.restrict(new_sws_ep).values, 3*np.std(lfp_filt_delta.restrict(new_sws_ep).values))
        
    # plt.figure()
    # plt.title('LFP trace')
    # plt.plot(lfp.restrict(new_sws_ep))
    # plt.plot(lfp_filt_delta.restrict(new_sws_ep), color = 'k')
    # plt.axhline(3*np.std(lfp_filt_delta.restrict(new_sws_ep).values))
    # plt.plot(lfp_filt_delta.restrict(new_sws_ep).index.values[a],lfp_filt_delta.restrict(new_sws_ep).values[a],'o')
    
    #Example is at 1063.8s (A3707)
    
# ############################################################################################### 
#     # PETH ALIGNED TO PEAK OF DELTA WAVE
# ###############################################################################################   
    sleep_lfp = nap.Tsd(lfp_filt_delta.restrict(new_sws_ep))   
    peaks = nap.Tsd(sleep_lfp.index.values[a])
    peth0 = nap.compute_perievent(sleep_lfp, peaks, minmax = (-1, 1), time_unit = 's')
    
    # peaks.to_pickle(rawpath + '/' + s + '_LFP_peaks.pkl')
    peaks.to_pickle(rawpath + '/' + s + '_LFP_peaks1.pkl')
    
    lfpmag = pd.DataFrame(index = peth0[0].index.values)
   
    for i in range(len(peth0)):
        if len(peth0[i].values) == len(lfpmag):
            lfpmag = pd.concat([lfpmag, peth0[i]], axis = 1)
            
    lfpmag = lfpmag.mean(axis = 1)      
    # plt.plot(lfpmag) 

# ############################################################################################### 
#     # LOOP OVER ALL LFP CHANNELS 
# ###############################################################################################  
    
    lfpsigs = {}
 
            
    for j in range(data.nChannels):
        
        if (matches == set()) is False:
            lfpsig = nap.load_eeg(filepath + '/' + name + '.eeg' , channel = j, n_channels = data.nChannels, frequency = 1250, precision ='int16', bytes_size = 2) 
        else: 
            lfpsig = nap.load_eeg(filepath + '/' + name + '.lfp' , channel = j, n_channels = data.nChannels, frequency = 1250, precision ='int16', bytes_size = 2) 
                  
        
        lfpsig = lfpsig[::downsample]
                        
        spatialPETH = nap.compute_perievent(lfpsig, peaks, minmax = (-1, 1), time_unit = 's')
        tmp = []
                
        for i in range(len(spatialPETH)):
            tmp.append(spatialPETH[i].as_series())
                                    
        tmp = pd.concat(tmp, axis = 1, join = 'inner')
        tmp = tmp.mean(axis = 1)     
               
        lfpsigs[j] = pd.Series(data = tmp, name = j)
            
    lfp_all = pd.DataFrame(index = lfpsigs[0].index.values)
   
    for i in channelorder:
        lfp_all = pd.concat([lfp_all, lfpsigs[i]], axis = 1)
        
    # lfp_all.to_pickle(rawpath + '/' + s + '_LFP_all.pkl') 
    lfp_all.to_pickle(rawpath + '/' + s + '_LFP_all1.pkl') 
            
    fig, ax = plt.subplots()
    cax = ax.imshow(lfp_all[-0.75:0.75].T,extent=[-0.75 , 0.75, data.nChannels , 1],aspect = 'auto', cmap = 'inferno')
    plt.xlabel('lag (s)')
    plt.ylabel('Channel number')
    plt.title('Delta PETH_' + s)
    cbar = fig.colorbar(cax, ticks=[lfp_all.values.min(), lfp_all.values.max()], label = 'LFP magnitude')
    
    plt.figure()
    plt.title('Delta PETH_' + s)
    plt.xlabel('lag (s)')
    plt.ylabel('LFP magnitude')
    j = 0
    for i in seq:
        plt.plot(lfp_all[-0.75:0.75][i], color=cm.inferno(j/8), label = j)
        plt.legend(loc = 'upper right')
        j+=1


    
    
    
    
    
    
    
   
                            
    


    
        
    
    

    




    
    
    