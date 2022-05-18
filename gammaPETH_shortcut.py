#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 12:58:35 2022

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
    
    peaks = pd.read_pickle(rawpath + '/' + s + '_LFP_peaks.pkl')
    gamma_all = pd.read_pickle(rawpath + '/' + s + '_gamma_all.pkl')
    
    
    
           
    fig, ax = plt.subplots()
    cax = ax.imshow(gamma_all[-0.75:0.75].T,extent=[-0.75 , 0.75, data.nChannels , 1],aspect = 'auto', cmap = 'inferno')
    plt.xlabel('lag (s)')
    plt.ylabel('Channel number')
    plt.title('Delta PETH_' + s)
    cbar = fig.colorbar(cax, ticks=[gamma_all[-0.75:0.75].values.min(), lfp_all[0.75:0.75].values.max()], label = 'LFP magnitude')
    
    # plt.figure()
    # plt.title('Delta PETH_' + s)
    # plt.xlabel('lag (s)')
    # plt.ylabel('LFP magnitude')
    # j = 0
    # for i in seq:
    #     plt.plot(lfp_all[-0.75:0.75][i], color=cm.inferno(j/8), label = j)
    #     plt.legend(loc = 'upper right')
    #     j+=1


    
    
    
    
    
    
    
   
                            
    


    
        
    
    

    




    
    
    