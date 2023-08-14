#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 12:33:26 2023

@author: dhruv
"""

import numpy as np 
import pandas as pd 
import pynapple as nap 
import scipy.io
import os, sys
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import kendalltau, pearsonr, wilcoxon, mannwhitneyu
from sklearn.linear_model import LinearRegression

#%% 

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
# datasets = np.loadtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'

FRcoef = []
Depthcoef = []

for s in datasets:
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    rawpath = os.path.join(rwpath,s)

    data = nap.load_session(rawpath, 'neurosuite')
    data.load_neurosuite_xml(rawpath)
    spikes = data.spikes  
    epochs = data.epochs
    
#%% LOAD MAT FILES

    filepath = os.path.join(path, 'Analysis')
    listdir    = os.listdir(filepath)
    
    file = [f for f in listdir if 'CellDepth' in f]
    celldepth = scipy.io.loadmat(os.path.join(filepath,file[0]))
    depth = celldepth['cellDep']
    
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
            
#%% LOAD UP AND DOWN STATES 

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
    
#%% COMPUTE FIRING RATE IN NREM

    NREM_fr = spikes.restrict(new_sws_ep)._metadata['rate']
    
    
#%% COMPUTE EVENT CROSS CORRS 

    cc2 = nap.compute_eventcorrelogram(spikes, nap.Tsd(up_ep['start'].values), binsize = 0.005, windowsize = 0.255, ep = up_ep, norm = True)
    tmp = pd.DataFrame(cc2)
    tmp = tmp.rolling(window=8, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
    dd2 = tmp[0:0.155]  
    ee = dd2[pyr] 
   
    if len(ee.columns) > 0:
                    
        tokeep = []
        depths_keeping_ex = []
        sess_uponset = []
        NREM_fr_ex = []    
        
        for i in range(len(ee.columns)):
            a = np.where(ee.iloc[:,i] > 0.5)
            if len(a[0]) > 0:
                tokeep.append(ee.columns[i])  
                depths_keeping_ex.append(depth.flatten()[ee.columns[i]])
                
                res = ee.iloc[:,i].index[a]
                sess_uponset.append(res[0])
                
                NREM_fr_ex.append(NREM_fr[ee.columns[i]])
                
#%% 

    metrics = pd.DataFrame(data = [scipy.stats.zscore(sess_uponset), scipy.stats.zscore(NREM_fr_ex), scipy.stats.zscore(depths_keeping_ex)], index = ['UPonset', 'FR', 'Depth']).T    

    mlr = LinearRegression()
    mlr.fit(metrics[['FR', 'Depth']], metrics['UPonset'])
    
    FRcoef.append(mlr.coef_[0])
    Depthcoef.append(mlr.coef_[1])

#%% 

FRtype = pd.DataFrame(['FR' for x in range(len(FRcoef))])
Depthtype = pd.DataFrame(['Depth' for x in range(len(Depthcoef))])

coeff_df = pd.DataFrame()
coeff_df['corr'] = pd.concat([pd.Series(Depthcoef), pd.Series(FRcoef)])
coeff_df['type'] = pd.concat([Depthtype, FRtype])

sns.set_style('white')
palette = ['darkorange', 'burlywood']
ax = sns.violinplot( x = coeff_df['type'], y=coeff_df['corr'] , data = coeff_df, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = coeff_df['type'], y=coeff_df['corr'] , data = coeff_df, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.swarmplot(x = coeff_df['type'], y=coeff_df['corr'], data=coeff_df, color = 'k', dodge=False, ax=ax)

for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.axhline(0, color = 'silver')
plt.ylabel('Regression Coefficient')
ax.set_box_aspect(1) 