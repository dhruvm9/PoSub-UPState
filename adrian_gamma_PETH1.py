#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 14:07:59 2022

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
from scipy.stats import wilcoxon, pearsonr
from scipy.signal import hilbert
import matplotlib.cm as cm

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
# datasets = np.loadtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'
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

mediancorr = []
medianp = []

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
    depth = np.arange(0, -800, -12.5)
    
# ############################################################################################### 
#     # LOAD UP AND DOWN STATE, NEW SWS AND NEW WAKE EPOCHS
# ###############################################################################################   
    file = os.path.join(rawpath, name +'.DM.new_sws.evt')
    new_sws_ep  = data.read_neuroscope_intervals(name = 'new_sws', path2file = file)
    
    file = os.path.join(rawpath, name +'.DM.new_wake.evt')
    new_wake_ep  = data.read_neuroscope_intervals(name = 'new_wake', path2file = file)
    
    # peaks = pd.read_pickle(rawpath + '/' + s + '_LFP_peaks.pkl')
    # lfp_all = pd.read_pickle(rawpath + '/' + s + '_LFP_all.pkl')
    
    peaks = pd.read_pickle(rawpath + '/' + s + '_LFP_peaks1.pkl')
    lfp_all = pd.read_pickle(rawpath + '/' + s + '_LFP_all1.pkl')
    
    filepath = os.path.join(rwpath, name)
    listdir    = os.listdir(filepath)
    
# ############################################################################################### 
#     #MEAN GAMMA POWER 
# ###############################################################################################  

    gammapows = {}
    med = {}
    file = [f for f in listdir if  name + '.eeg' in f]
    filename = [name + '.eeg']
    matches = set(filename).intersection(set(file))
    
            
    for j in range(data.nChannels):
        if (matches == set()) is False:
            lfpsig = nap.load_eeg(filepath + '/' + name + '.eeg' , channel = j, n_channels = data.nChannels, frequency = 1250, precision ='int16', bytes_size = 2) 
        else: 
            lfpsig = nap.load_eeg(filepath + '/' + name + '.lfp' , channel = j, n_channels = data.nChannels, frequency = 1250, precision ='int16', bytes_size = 2) 
                 
        downsample = 2
        #lfpsig = downsample(lfpsig, 1, 2)
        lfpsig = lfpsig[::downsample]
        
        # lfp_filt_gamma = nap.Tsd(lfpsig.index.values, butter_bandpass_filter(lfpsig, 30, 50, 1250/2, 3))
        lfp_filt_gamma = nap.Tsd(lfpsig.index.values, butter_bandpass_filter(lfpsig, 70, 150, 1250/downsample, 3))

        del lfpsig
        
        power_gamma = nap.Tsd(lfp_filt_gamma.index.values, np.abs(hilbert(lfp_filt_gamma.values)))
        med[j] = np.median(power_gamma.restrict(new_sws_ep).values)
        
        del lfp_filt_gamma
        
        pgs = power_gamma.as_series()
        pgs = pgs.rolling(window=20,win_type='gaussian',center=True,min_periods=1).mean(std=80)
        
        # power_gamma.to_pickle(rawpath + '/' + s + '_power_gamma.pkl')        
        # power_gamma.to_pickle(rawpath + '/' + s + '_power_highgamma.pkl')        
        
                
        #%%
        
        gammaPETH = perievent_Tsd(nap.Tsd(pgs),nap.Ts(peaks.index.values), minmax = (1,1))
        
        gammapows[j] = pd.Series(data = gammaPETH['mean'], name = j)
    
    gamma_all = pd.DataFrame(index = gammapows[0].index.values)
    chanmed = []
   
    for i in channelorder:
        gamma_all = pd.concat([gamma_all, gammapows[i]], axis = 1)
        chanmed.append(med[i])
         
      

    corr, p = pearsonr(chanmed, depth)
    mediancorr.append(corr)
    medianp.append(p)
    
    # plt.figure()
    # plt.title('Median Gamma power v/s depth_' + s)
    # plt.xlabel('Median Gamma Power')
    # plt.ylabel('Depth (um)')
    # plt.scatter(chanmed, depth, label = 'R = ' + str(round(corr,4)))
    # plt.legend(loc = 'upper right')
    # plt.show()
    
    # gamma_all.to_pickle(rawpath + '/' + s + '_gamma_all.pkl')
    # gamma_all.to_pickle(rawpath + '/' + s + '_highgamma_all.pkl')
    gamma_all.to_pickle(rawpath + '/' + s + '_highgamma_all1.pkl')
    
# np.save(rwpath + '/' + 'mediancorr_gamma_3.npy', np.array(mediancorr))
# np.save(rwpath + '/' + 'medianp_gamma_3.npy', np.array(medianp))
      
    bounds = [-0.2,0.3]
    fig, ax = plt.subplots()
    cax = ax.imshow(gamma_all[bounds[0]:bounds[1]].T,extent=[bounds[0] , bounds[1], data.nChannels , 1],aspect = 'auto', cmap = 'OrRd')
    cbar = fig.colorbar(cax, ticks=[gamma_all[bounds[0]:bounds[1]].values.min(), gamma_all[bounds[0]:bounds[1]].values.max()], label = 'Gamma power')
    plt.xlabel('Lag (s)')
    plt.ylabel('Channel number')
    plt.yticks([1,64])
    # plt.title('Gamma PETH_' + s)
    plt.plot(lfp_all[bounds[0]:bounds[1]][seq[1]].index.values, 
              (-0.003* lfp_all[bounds[0]:bounds[1]][seq[1]].values)+10, color = 'white')
    plt.plot(lfp_all[bounds[0]:bounds[1]][seq[3]].index.values, 
              (-0.003* lfp_all[bounds[0]:bounds[1]][seq[3]].values)+25, color = 'white') #'gainsboro')
    plt.plot(lfp_all[bounds[0]:bounds[1]][seq[5]].index.values, (-0.003* lfp_all[bounds[0]:bounds[1]][seq[5]].values)+40, color = 'white') #'silver')
    plt.plot(lfp_all[bounds[0]:bounds[1]][seq[7]].index.values, (-0.003* lfp_all[bounds[0]:bounds[1]][seq[7]].values)+55, color = 'white') #'grey')
    plt.axvline(0, color = 'white', linestyle = '--')
    ax.set_box_aspect(1)
   
    
    w = pd.DataFrame(index = gamma_all.index.values[0:-1], columns = seq) #Drop end values
    peakval = []
    troughs = []
        
  #%% Quantification example plot 
    
    for i in seq:
        w[i] = np.diff(gamma_all[i].values)
        w[i] = w[i].rolling(window=60,win_type='gaussian',center=True,min_periods=1).mean(std=80)
        peakval.append(w[0:1][i].idxmax())
        troughs.append(w[-1:0][i].idxmin())
              
    plt.figure()
    plt.xlabel('lag (s)')
    plt.ylabel('Gamma Power')
    plt.title('Gamma PETH_' + s)
    j = 0
    for i in seq:
        plt.plot(gamma_all[-0.2:0.3][i], color=cm.gist_heat(j/8), label = j)
        plt.plot(peakval[j], gamma_all[i].loc[peakval[j]], 'o', color = 'r', markersize =4)
        plt.ylim(55,None)
        plt.axvline(0, color = 'k', linestyle = '--')
        plt.legend(loc = 'lower left')
        j+=1
    plt.vlines(peakval[0],55, gamma_all[seq[0]].loc[peakval[0]], color = 'r', linestyle = 'dashed')
    plt.vlines(peakval[-1],55, gamma_all[seq[-1]].loc[peakval[-1]], color = 'r', linestyle = 'dashed')
    plt.gca().set_box_aspect(1)

#%% 

    ###Showing threshold 
    plt.figure()
    plt.plot(gamma_all[-0.2:0.3][seq[4]], color = 'r')
    plt.plot(w[-0.2:0.3][seq[4]]*50, color = 'k')
    plt.axvline(w[-0.2:0.3][seq[4]].idxmax(),linestyle = '--', color = 'silver')
    plt.axvline(0,linestyle = '--', color = 'silver')
    plt.xticks(visible = True)
    plt.yticks(visible = True)
    plt.gca().set_box_aspect(1)
    
    
    
    #%%
# ############################################################################################### 
#     #SINGLE TRIAL GAMMA POWER 
# ############################################################################################### 
    
    
    # period = nap.IntervalSet(start = 1063.6, end = 1064.2, time_units = 's')
    
    # gp = {}
    # lfpchan = {}

    # for j in seq:
        
    #     file = [f for f in listdir if  name + '.eeg' in f]
          
    #     if file == name + '.eeg':
    #         lfpsig = nap.load_eeg(filepath + '/' + name + '.eeg' , channel = j , n_channels = data.nChannels, frequency = 1250, precision ='int16', bytes_size = 2) 
    #     else: 
    #         file = name + '.lfp'
    #         lfpsig = nap.load_eeg(filepath + '/' + name + '.lfp' , channel = j , n_channels = data.nChannels, frequency = 1250, precision ='int16', bytes_size = 2) 
     
    #     downsample = 2
    #     lfpsig = lfpsig[::downsample]
    #     lfpchan[j] = lfpsig 

        
    #     lfp_filt_gamma = nap.Tsd(lfpsig.index.values, butter_bandpass_filter(lfpsig, 70, 150, 1250/2, 2))
    #     power_gamma = nap.Tsd(lfp_filt_gamma.index.values, np.abs(hilbert(lfp_filt_gamma.values)))
    #     power_gamma = power_gamma.restrict(period)
    #     power_gamma = pd.Series(index=power_gamma.index.values - 1063.816, data = power_gamma.values )
    #     power_gamma = power_gamma.rolling(window=20,win_type='gaussian',center=True,min_periods=1).mean(std=80)
               
                
    #     gp[j] = power_gamma
        
    # gpdf = pd.DataFrame(index = gp[seq[0]].index.values)
        
    # for i in seq:
    #     gp[i] = pd.Series(index =gp[seq[0]].index.values, data = gp[i], name = i)
    #     gpdf = pd.concat([gpdf, gp[i]], axis = 1)
      
         
    
    # fig, ax = plt.subplots()
    # cax = ax.imshow(gpdf.T,extent=[period['start'][0] , period['end'][0], data.nChannels , 1],aspect = 'auto', cmap = 'inferno')
    # cbar = fig.colorbar(cax, ticks=[gpdf.values.min(), gpdf.values.max()], label = 'Gamma power',)
    # plt.plot(lfpchan[seq[1]].restrict(period).index.values, (-0.0025* lfpchan[seq[1]].restrict(period).values)+10, color = 'white')
    # plt.plot(lfpchan[seq[3]].restrict(period).index.values, (-0.0025* lfpchan[seq[3]].restrict(period).values)+25, color = 'white') #'gainsboro')
    # plt.plot(lfpchan[seq[5]].restrict(period).index.values, (-0.0025* lfpchan[seq[5]].restrict(period).values)+40, color = 'white') #'silver')
    # plt.plot(lfpchan[seq[7]].restrict(period).index.values, (-0.0025* lfpchan[seq[7]].restrict(period).values)+55, color = 'white') #'grey')
    # plt.yticks([1,64])
    # plt.axvline(1063.816, color = 'white', linestyle = '--')
    # ax.set_box_aspect(1)