#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:13:12 2023

@author: dhruv
"""

import numpy as np 
import pandas as pd 
import scipy.io
from functions import *
from wrappers import *
import os, sys
import nwbmatic as ntm
import pynapple as nap
import time 
import matplotlib.pyplot as plt 
from Wavelets import MyMorlet as Morlet
from scipy.stats import kendalltau, pearsonr, wilcoxon, mannwhitneyu
import seaborn as sns 

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
# datasets = np.loadtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'

s = 'A3707-200317'


#%% 

name = s.split('/')[-1]
path = os.path.join(data_directory, s)
rawpath = os.path.join(rwpath,s)

data = ntm.load_session(rawpath, 'neurosuite')
data.load_neurosuite_xml(rawpath)
spikes = data.spikes  
epochs = data.epochs

per = nap.IntervalSet(start = 1247.288, end = 1255.288)

#%% Load UP and DOWN states

      
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
    
sws_ep = data.read_neuroscope_intervals(name = 'SWS', path2file = file)

filepath = os.path.join(path, 'Analysis')
listdir    = os.listdir(filepath)
file = [f for f in listdir if 'CellTypes' in f]
celltype = scipy.io.loadmat(os.path.join(filepath,file[0]))

gd = [] 

for i in range(len(spikes)):
    if celltype['gd'][i] == 1:
         gd.append(i)

#%% EEG 

channelorder = data.group_to_channel[0]
seq = channelorder[::8].tolist()

filepath = os.path.join(rwpath, name)
listdir    = os.listdir(filepath)
    
file = [f for f in listdir if  name + '.eeg' in f]
filename = [name + '.eeg']
matches = set(filename).intersection(set(file))

if (matches == set()) is False:
    lfpsig = nap.load_eeg(filepath + '/' + name + '.eeg' , channel = seq[int(len(seq)/2)], n_channels = data.nChannels, frequency = 1250, precision ='int16', bytes_size = 2) 
else: 
    lfpsig = nap.load_eeg(filepath + '/' + name + '.lfp' , channel = seq[int(len(seq)/2)], n_channels = data.nChannels, frequency = 1250, precision ='int16', bytes_size = 2) 

downsample =  5
lfpsig = lfpsig[::downsample]
lfp_filt_delta = nap.Tsd(lfpsig.index.values, butter_bandpass_filter(lfpsig, 0.5, 4, 1250/5, 2))

peaks = pd.read_pickle(rawpath + '/' + s + '_LFP_peaks1.pkl')


#%% Population Rate 

bin_size = 0.01 #s
smoothing_window = 0.02

rates = spikes.count(bin_size, sws_ep)
   
total2 = rates.as_dataframe().rolling(window = 100 ,win_type='gaussian',
                                      center=True,min_periods=1, 
                                      axis = 0).mean(std= int(smoothing_window/bin_size))

total2 = total2.sum(axis =1)
total2 = nap.Tsd(total2)
idx = total2.threshold(np.percentile(total2.values,20),'below')

#%% 

plt.figure() 
plt.subplot(311)       
plt.plot(lfpsig.restrict(per))
[plt.axvspan(x,y, facecolor = 'g', alpha = 0.5) for x,y in zip(down_ep['start'][691:696], down_ep['end'][691:696])]
plt.subplot(312)
for i,n in enumerate(spikes.keys()):
    plt.plot(spikes[n].restrict(per).fillna(i+35), '|', color = 'k')
[plt.axvspan(x,y, facecolor = 'g', alpha = 0.5) for x,y in zip(down_ep['start'][691:696], down_ep['end'][691:696])]
plt.plot(total2.restrict(per), color = 'dimgray', alpha = 0.5)
plt.axhline(np.percentile(total2.values,20), zorder = 5, color = 'k')
plt.fill_between(total2.restrict(per).index.values, total2.restrict(per), color = 'dimgray', alpha = 0.5)
plt.subplot(313)
plt.plot((lfp_filt_delta.restrict(per)*0.01) + 55, zorder = 5, linewidth = 2)
plt.plot(total2.restrict(per), color = 'dimgray', alpha = 0.5)
plt.fill_between(total2.restrict(per).index.values, total2.restrict(per), color = 'dimgray', alpha = 0.5)
plt.plot((lfpsig.restrict(per)*0.01) + 55, color = 'k')
[plt.axvspan(x,y, facecolor = 'g', alpha = 0.5) for x,y in zip(down_ep['start'][691:696], down_ep['end'][691:696])]

# plt.plot(lfp_filt_delta[peaks.restrict(per).index.values],'o')
# [plt.axvspan(x,y, facecolor = 'g', alpha = 0.5) for x,y in zip(down_ep.loc[691:696]['start'].values, down_ep.loc[691:696]['end'].values)]