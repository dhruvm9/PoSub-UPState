#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 13:33:24 2021

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
from Wavelets import MyMorlet as Morlet
from scipy.stats import pearsonr 
from scipy.stats import wilcoxon, mannwhitneyu

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
# datasets = np.loadtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
datasets = np.loadtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'

dur_D = []
dur_V = []
pmeans = []
diff = []

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
    
    for i in range(len(spikes)):
        if celltype['ex'][i] == 1 and celltype['gd'][i] == 1:
            data.loc[i,'celltype'] = 'ex' #0 for excitatory
        elif celltype['fs'][i] == 1 and celltype['gd'][i] == 1:
            data.loc[i,'celltype'] = 'fs' #1 for inhibitory
    
    # data = data[data['celltype'] == 'ex'] #Doing it for each cell type separately 
    # data = data[data['celltype'] == 'fs']
    
    bin_size = 10000 #us    
    
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
   
############################################################################################### 
    # COMPUTE EVENT CROSS CORRS
###############################################################################################  
                 
    binsize = 5
    nbins = 1000        
    neurons = list(mua.keys())
    times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2        
    cc = pd.DataFrame(index = times, columns = neurons)
    tsd_dn = down_ep.as_units('ms').start.values
    
    # ep_D = nts.IntervalSet(start = down_ep.start[0], end = down_ep.end.values[-1])
    rates = []
    
    for i in neurons:
        spk2 = mua[i].restrict(up_ep).as_units('ms').index.values
        tmp = crossCorr(tsd_dn, spk2, binsize, nbins)
        tmp = pd.DataFrame(tmp)
        tmp = tmp.rolling(window=4, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
        
        fr = len(spk2)/up_ep.tot_length('s')
        rates.append(fr)
        cc[i] = tmp.values
        cc[i] = tmp.values/fr
       
        dd = cc[-250:250]
        
    if len(dd.columns) > 0:
        tmp = dd.loc[5:] >  np.percentile(dd.values,20) #0.2
        
        
        tokeep = tmp.columns[tmp.sum(0) > 0]
        ends = np.array([tmp.index[np.where(tmp[j])[0][0]] for j in tokeep])
        es = pd.Series(index = tokeep, data = ends)
        
        tmp2 = dd.loc[-100:-5] > np.percentile(dd.values,20)  #0.2

        tokeep2 = tmp2.columns[tmp2.sum(0) > 0]
        start = np.array([tmp2.index[np.where(tmp2[k])[0][-1]] for k in tokeep2])
        st = pd.Series(index = tokeep2, data = start)
            
        ix = np.intersect1d(tokeep,tokeep2)
        ix = [int(m) for m in ix]
        
        
        depths_keeping = depth[ix]
        stk = st[ix]

        dur = np.zeros(len(ix))
        for i,n in enumerate(ix):
                dur[i] = es[ix][n] - st[ix][n]
        
        if dur[0] > 10 and dur[1] > 10:
            dur_D.append(dur[0])
            dur_V.append(dur[1])
                
        
diff = np.zeros(len(dur_D))

for i in range(len(diff)):
    diff[i] = dur_V[i] - dur_D[i]
    
# rge = np.linspace(min(diff),max(diff),10)
# plt.figure()
# plt.title('Mean (Ventral - Dorsal) duration (ms)')
# plt.xlabel('Difference (ms)')
# plt.ylabel('Number of sessions')
# plt.axvline(np.mean(diff), color = 'k')
# plt.hist(diff,rge, label = 'Mean = ' +  str(round(np.mean(diff),4)))
# plt.legend()

t,p = wilcoxon(dur_D,dur_V)

label = ['Dorsal', 'Ventral']
x1 = np.random.normal(0, 0.01, size=len(dur_D))
x2 = np.random.normal(0.3, 0.01, size=len(dur_D))
x = np.vstack([x1, x2])# the label locations
width = 0.3  # the width of the bars

plt.figure()
plt.boxplot(dur_D, positions=[0], showfliers=False, patch_artist=True, boxprops=dict(facecolor='mediumorchid', color='mediumorchid'),
            capprops=dict(color='mediumorchid'),
            whiskerprops=dict(color='mediumorchid'),
            medianprops=dict(color='white', linewidth = 2))
plt.boxplot(dur_V, positions=[0.3], showfliers=False, patch_artist=True, boxprops=dict(facecolor='violet', color='violet'),
            capprops=dict(color='violet'),
            whiskerprops=dict(color='violet'),
            medianprops=dict(color='white', linewidth = 2))

plt.xticks([0, 0.3],['Dorsal', 'Ventral'])
plt.title('Mean session DOWN-state duration')
plt.ylabel('DOWN duration (ms)')
pval = np.vstack([(dur_D), (dur_V)])
plt.plot(x, np.vstack(pval), 'o-', color = 'k', zorder = 3, markersize = 3, linewidth = 1 )


means_D = np.nanmean(dur_D)
means_V = np.nanmean(dur_V)

label = ['dorsal', 'ventral']
x = [0, 0.35]# the label locations
width = 0.35  # the width of the bars


fig, ax = plt.subplots()
rects1 = ax.bar(x[0], means_D, width, color = 'royalblue')
rects1 = ax.bar(x[1], means_V, width, color = 'lightsteelblue')
plt.plot(x, np.vstack(pval), 'o-', color = 'k', zorder = 3, markersize = 3, linewidth = 1 )