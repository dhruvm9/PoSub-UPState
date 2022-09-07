#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 15:58:14 2022

@author: dhruv
"""
import numpy as np 
import pandas as pd 
import scipy.io
import pynapple as nap 
import os, sys
import time 
import matplotlib.pyplot as plt 
from scipy.stats import kendalltau, pearsonr, wilcoxon, mannwhitneyu
import seaborn as sns

#%% On Lab PC
# data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
# datasets = np.loadtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
# # datasets = np.loadtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

# rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'

# for s in datasets:
#     print(s)
#     name = s.split('/')[-1]
#     path = os.path.join(data_directory, s)
#     rawpath = os.path.join(rwpath,s)

#     data = nap.load_session(rawpath, 'neurosuite')
#     data.load_neurosuite_xml(rawpath)
#     spikes = data.spikes  
#     epochs = data.epochs
    
#     # ############################################################################################### 
#     #     # LOAD MAT FILES
#     # ############################################################################################### 
#         filepath = os.path.join(path, 'Analysis')
#         listdir    = os.listdir(filepath)
#         file = [f for f in listdir if 'BehavEpochs' in f]
#         behepochs = scipy.io.loadmat(os.path.join(filepath,file[0]))  

#         file = [f for f in listdir if 'CellDepth' in f]
#         celldepth = scipy.io.loadmat(os.path.join(filepath,file[0]))
#         depth = celldepth['cellDep']

#         file = [f for f in listdir if 'MeanFR' in f]
#         mfr = scipy.io.loadmat(os.path.join(filepath,file[0]))
#         r_wake = mfr['rateS']


#         file = [f for f in listdir if 'CellTypes' in f]
#         celltype = scipy.io.loadmat(os.path.join(filepath,file[0]))
        
#         pyr = []
#         interneuron = []
#         hd = []
            
#         for i in range(len(spikes)):
#             if celltype['ex'][i] == 1 and celltype['gd'][i] == 1:
#                 pyr.append(i)
                
#         for i in range(len(spikes)):
#             if celltype['fs'][i] == 1 and celltype['gd'][i] == 1:
#                 interneuron.append(i)
                
#         for i in range(len(spikes)):
#             if celltype['hd'][i] == 1 and celltype['gd'][i] == 1:
#                 hd.append(i)

#     # ############################################################################################### 
#     #     # LOAD UP AND DOWN STATE, NEW SWS AND NEW WAKE EPOCHS
#     # ###############################################################################################   

        
#         file = os.path.join(rawpath, name +'.evt.py.dow')
#         if os.path.exists(file):
#             tmp = np.genfromtxt(file)[:,0]
#             tmp = tmp.reshape(len(tmp)//2,2)/1000
#             down_ep = nap.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
        
#         file = os.path.join(rawpath, name +'.evt.py.upp')
#         if os.path.exists(file):
#             tmp = np.genfromtxt(file)[:,0]
#             tmp = tmp.reshape(len(tmp)//2,2)/1000
#             up_ep = nap.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
        
#         file = os.path.join(rawpath, name +'.DM.new_sws.evt')
#         if os.path.exists(file):
#             tmp = np.genfromtxt(file)[:,0]
#             tmp = tmp.reshape(len(tmp)//2,2)/1000
#             new_sws_ep = nap.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
        
#         file = os.path.join(rawpath, name +'.DM.new_wake.evt')
#         if os.path.exists(file):
#             tmp = np.genfromtxt(file)[:,0]
#             tmp = tmp.reshape(len(tmp)//2,2)/1000
#             new_wake_ep = nap.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
            
#     ############################################################################################### 
#         # COMPUTE EVENT CROSS CORRS
#     ###############################################################################################  



#         cc = nap.compute_perievent(spikes, nap.Ts(up_ep['start'].values) ,minmax = (-0.25, 0.25), time_unit = 's')
    
#%% On Nibelungen 

data_directory = '/mnt/DataNibelungen/Dhruv/A3707-200317'
rwpath = '/mnt/DataNibelungen/Dhruv/'
data = nap.load_session(data_directory, 'neurosuite')
s = 'A3707-200317'

spikes = data.spikes
epochs = data.epochs
file = os.path.join(data_directory, s +'.DM.new_sws.evt')

filepath = os.path.join(data_directory, 'Analysis')
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
#     #     # LOAD UP AND DOWN STATE, NEW SWS AND NEW WAKE EPOCHS
# ###############################################################################################  

new_sws_ep  = data.read_neuroscope_intervals(name = 'new_sws', path2file = file)

file = os.path.join(data_directory, s +'.evt.py.dow')
if os.path.exists(file):
    tmp = np.genfromtxt(file)[:,0]
    tmp = tmp.reshape(len(tmp)//2,2)/1000
    down_ep = nap.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')

file = os.path.join(data_directory, s +'.evt.py.upp')
if os.path.exists(file):
    tmp = np.genfromtxt(file)[:,0]
    tmp = tmp.reshape(len(tmp)//2,2)/1000
    up_ep = nap.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')

# ############################################################################################### 
#          # COMPUTE EVENT CROSS CORRS
# ###############################################################################################  

cc = nap.compute_eventcorrelogram(spikes, nap.Tsd(up_ep['start'].values), binsize = 0.005, windowsize = 0.25, ep = new_sws_ep)
tmp = pd.DataFrame(cc)
tmp = tmp.rolling(window=4, win_type='gaussian',center=True,min_periods=1).mean(std = 2)

dd = tmp[-0.05:0.25]
n = len(depth)
tmp = np.argsort(depth.flatten())
desc = tmp[::-1][:n]

