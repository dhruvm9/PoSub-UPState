#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 14:26:41 2021
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
from scipy.stats import wilcoxon, mannwhitneyu

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
# datasets = np.loadtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
datasets = np.loadtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'

dorsaldurs = []
ventraldurs = []
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
    
    # for i in range(len(spikes)):
    #     if celltype['ex'][i] == 1 and celltype['gd'][i] == 1:
    #         data.loc[i,'celltype'] = 'ex' #0 for excitatory
    #     elif celltype['fs'][i] == 1 and celltype['gd'][i] == 1:
    #         data.loc[i,'celltype'] = 'fs' #1 for inhibitory
    
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
    
    
    down_D = pd.DataFrame()
    down_V = pd.DataFrame()
    
    for j in new_sws_ep.index:                
        ep = new_sws_ep.loc[[j]]
        bins = np.arange(ep.loc[j,'start'], ep.loc[j,'end'], bin_size)        
        r = np.array([np.histogram(mua[i].restrict(ep).index.values, bins)[0] for i in range(2)])
        r = pd.DataFrame(index = bins[0:-1] + np.diff(bins)/2, data = r.T)            
        r2 = r.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2)
        dd = r2[0][r2[0]<np.percentile(r2[0],20,0)].index.values
        dv = r2[1][r2[1]<np.percentile(r2[1],20,0)].index.values
        
        tmp_d = [[dd[0]]]
        tmp_v = [[dv[0]]]
        
        for i in range(1,len(dd)):
            if (dd[i] - dd[i-1]) > bin_size:
                tmp_d.append([dd[i]])
            elif (dd[i] - dd[i-1]) == bin_size:
                tmp_d[-1].append(dd[i])
                
        for i in range(1,len(dv)):
            if (dv[i] - dv[i-1]) > bin_size:
                tmp_v.append([dv[i]])
            elif (dv[i] - dv[i-1]) == bin_size:
                tmp_v[-1].append(dv[i])
            
        down_dorsal = np.array([[e[0],e[-1]] for e in tmp_d if len(e) > 1])
        down_dorsal = nts.IntervalSet(start = down_dorsal[:,0], end = down_dorsal[:,1])
        down_dorsal = down_dorsal.drop_short_intervals(bin_size)
        down_dorsal = down_dorsal.reset_index(drop=True)
        down_dorsal = down_dorsal.merge_close_intervals(bin_size*2)
        down_dorsal = down_dorsal.drop_short_intervals(bin_size*3)
        down_dorsal = down_dorsal.drop_long_intervals(bin_size*50)
        down_D = down_D.append(down_dorsal)
        
        down_ventral = np.array([[e[0],e[-1]] for e in tmp_v if len(e) > 1])
        down_ventral = nts.IntervalSet(start = down_ventral[:,0], end = down_ventral[:,1])
        down_ventral = down_ventral.drop_short_intervals(bin_size)
        down_ventral = down_ventral.reset_index(drop=True)
        down_ventral = down_ventral.merge_close_intervals(bin_size*2)
        down_ventral = down_ventral.drop_short_intervals(bin_size*3)
        down_ventral = down_ventral.drop_long_intervals(bin_size*50)
        down_V = down_V.append(down_ventral)
    
    
    dur_dorsal = np.zeros(len(down_D))
    dur_ventral = np.zeros(len(down_V))
               
        
    for i in range(len(down_D)):
        dur_dorsal[i] = (down_D['end'].iloc[i] - down_D['start'].iloc[i]) / 1000
        
    dorsaldurs.append(np.mean(dur_dorsal))
        
    for i in range(len(down_V)):
        dur_ventral[i] = (down_V['end'].iloc[i] - down_V['start'].iloc[i]) / 1000
        
    ventraldurs.append(np.mean(dur_ventral))
        
    diff.append(np.mean(dur_ventral) - np.mean(dur_dorsal))
    
    # bins = np.linspace(min(min(dur_dorsal),min(dur_ventral)),max(max(dur_dorsal),max(dur_ventral)), 30)
    
    # plt.figure()
    # plt.subplot(131)
    # plt.suptitle(s)
    # x = np.arange(0, np.maximum(max(dur_dorsal), max(dur_ventral)),10)
    # d = np.histogram(dur_dorsal, x)[0]
    # v = np.histogram(dur_ventral, x)[0]
    # plt.bar(x[0:-1], d, width = 5, label = 'dorsal')
    # plt.bar(x[0:-1] + np.diff(x)/2, v, width = 5, label = 'ventral')
    
    # plt.ylabel('Number of DOWN states')
    # plt.xlabel('Duration (ms)')
    # plt.subplot(132)
    
    # plt.xlabel('Duration(ms)')
    # plt.ylabel('Cumulative sum')
    # x2 = x[0:-1]
    # plt.plot(x2, np.cumsum(d), label = 'dorsal')
    # plt.plot(x2, np.cumsum(v), label = 'ventral')
    # plt.legend()    
    # plt.subplot(133)
    # plt.boxplot(dur_dorsal, positions = [0], showfliers = False)
    # plt.boxplot(dur_ventral, positions = [0.3], showfliers = False)
    # plt.xticks([0, 0.3],['Dorsal', 'Ventral'])
    # plt.ylabel('Duration(ms)')

rge = np.linspace(min(diff),max(diff),10)
plt.figure()
plt.title('Mean (Ventral - Dorsal) duration (ms)')
plt.xlabel('Difference (ms)')
plt.ylabel('Number of sessions')
plt.axvline(np.mean(diff), color = 'k')
plt.hist(diff,rge, label = 'Mean = ' +  str(round(np.mean(diff),4)))
plt.legend()

t,p = mannwhitneyu(dorsaldurs,ventraldurs)
means_D = np.nanmean(dorsaldurs)
means_V = np.nanmean(ventraldurs)

label = ['Dorsal', 'Ventral']
x = [0, 0.35]# the label locations
width = 0.35  # the width of the bars


fig, ax = plt.subplots()
rects1 = ax.bar(x[0], means_D, width, color = 'royalblue', label = 'p = ' +  str(round(np.mean(p),4)) )
rects2 = ax.bar(x[1], means_V, width, color = 'lightsteelblue')

pval = np.vstack([(dorsaldurs), (ventraldurs)])

# x2 = [x-width/2, x+width/2]
plt.plot(x, np.vstack(pval), 'o-', color = 'k', markersize =5, linewidth = 1)

# # Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Duration (ms)')
ax.set_title('Mean session DOWN-state duration')
ax.set_xticks(x)
plt.legend(loc = 'lower right')
ax.set_xticklabels(label)
fig.tight_layout()

