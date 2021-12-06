#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 14:07:30 2021

@author: dhruv
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 09:59:42 2021

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
#datasets = np.loadtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
datasets = np.loadtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'

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
#     # LOAD GLOBAL UP AND DOWN STATE
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
############################################################################################### 
    data = pd.DataFrame()   
        
       
    data['depth'] = np.reshape(depth,(len(spikes.keys())),)
    data['level'] = pd.cut(data['depth'],2, precision=0, labels=[1,0])
         
############################################################################################### 
    # CREATE NEW EPOCHS CONTAINING DOWN+UP
############################################################################################### 

    bin_size = 10000 #us
    epoch = []
    down = []

    for e in new_sws_ep.index:
        rates = []
        ep = new_sws_ep.loc[[e]]
        bins = np.arange(ep.iloc[0,0], ep.iloc[0,1], bin_size)       
        r = np.zeros((len(bins)-1))
        
        for n in spikes.keys(): 
            tmp = np.histogram(spikes[n].restrict(ep).index.values, bins)[0]
            r = r + tmp
        rates.append(pd.Series(index = bins[0:-1] + np.diff(bins)/2, data = r))
            
        rates = pd.concat(rates)
        total2 = rates.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2)
        
        idx = total2[total2<np.percentile(total2,20)].index.values   
        tmp2 = [[idx[0]]]
        for i in range(1,len(idx)):
            if (idx[i] - idx[i-1]) > bin_size:
                 tmp2.append([idx[i]])
            elif (idx[i] - idx[i-1]) == bin_size:
                 tmp2[-1].append(idx[i])
        
        dn_ep = np.array([[e[0],e[-1]] for e in tmp2 if len(e) > 1])
        dn_ep = nts.IntervalSet(start = dn_ep[:,0], end = dn_ep[:,1])
        dn_ep = dn_ep.drop_short_intervals(bin_size)
        dn_ep = dn_ep.reset_index(drop=True)
        dn_ep = dn_ep.merge_close_intervals(bin_size*2)
        dn_ep = dn_ep.drop_short_intervals(bin_size*3)
        dn_ep = dn_ep.drop_long_intervals(bin_size*50)
        us_ep = nts.IntervalSet(dn_ep['end'][0:-1], dn_ep['start'][1:])
        us_ep = new_sws_ep.intersect(us_ep)
        dn_ep = dn_ep.iloc[1:len(dn_ep)]
    
        for i in range(len(us_ep)):
            new = nts.IntervalSet(start = us_ep.iloc[i]['start'], end = dn_ep.iloc[i]['end'])
            epoch.append(new)
            #reference is the center of the DOWN state. Use it to calculate latency of last spike
            down.append(dn_ep.iloc[i,:].mean())
            
    epoch = pd.concat(epoch,0)
    epoch = nts.IntervalSet(epoch)
    epoch = epoch.reset_index(drop=True)
    
    epoch.to_pickle(rawpath + '/' + name + '_UDepoch.pkl')     
    np.save(rawpath + '/' + name + '_UDref', down)
    
    
    
    