#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 12:30:43 2022

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

#Parameters
##Ex cells, norm = True, period = 0 - 100 ms, UP onset ---> 9/16 significant 


#%% On Lab PC
data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
# datasets = np.loadtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
datasets = np.loadtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')
# datasets = np.loadtxt(os.path.join(data_directory,'dataset_Ver_DM.list'), delimiter = '\n', dtype = str, comments = '#')

rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'

corrs = []
pvals = []

corrs_sws = []
pvals_sws = []


peak_above_mean = []
uponset = []

depthcorrs = []
depthpvals = []


s = 'A3707-200317a'
print(s)
name = 'A3707-200317'
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

# plt.title('Firing rates_' + s)
# plt.scatter(spikes[pyr].restrict(new_sws_ep)._metadata['freq'].values,spikes[pyr].restrict(up_ep)._metadata['freq'].values)
# plt.xlabel('SWS rate')
# plt.ylabel('UP rate')
############################################################################################### 
# COMPUTE EVENT CROSS CORRS
###############################################################################################  
  
## Peak firing       
cc2 = nap.compute_eventcorrelogram(spikes, nap.Tsd(up_ep['start'].values), binsize = 0.005, windowsize = 0.255, ep = up_ep, norm = True)
tmp = pd.DataFrame(cc2)
tmp = tmp.rolling(window=4, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
# dd2 = tmp[0:0.255]
dd2 = tmp[0:0.105]  
        
# #Excitatory cells only 
ee = dd2[pyr] 

if len(ee.columns) > 0:
                
    indexplot_ex = []
    depths_keeping_ex = []
    peaks_keeping_ex = []
                       
        
    for i in range(len(ee.columns)):
        a = np.where(ee.iloc[:,i] > 0.5)
        if len(a[0]) > 0:
          peaks_keeping_ex.append(ee.iloc[:,i].max())
          peak_above_mean.append(ee.iloc[:,i].max())
          depths_keeping_ex.append(depth.flatten()[ee.columns[i]])
            
          res = ee.iloc[:,i].index[a]
          indexplot_ex.append(res[0])
          uponset.append(res[0])
  


            
# corr, p = kendalltau(indexplot_ex, peaks_keeping_ex)

corr, p = kendalltau(indexplot_ex, peaks_keeping_ex)
corrs.append(corr)
pvals.append(p)
    
     
plt.figure()
plt.rc('font', size = 15)
plt.title('Peak/ mean FR v/s UP onset_' + s)
plt.scatter(indexplot_ex, peaks_keeping_ex, label = 'R = ' + str((round(corr,2))), color = 'cornflowerblue')
plt.xlabel('Time from UP onset (s)')
plt.ylabel('Peak/mean FR')
plt.legend(loc = 'upper right')

