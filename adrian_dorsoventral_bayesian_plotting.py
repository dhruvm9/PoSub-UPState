#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 08:31:32 2024

@author: adrien
"""

import matplotlib.pyplot as plt 
import numpy as np
import os 
import pandas as pd
import seaborn as sns

#%% 

data_directory = '/media/adrien/LaCie/PoSub-UPState/Data/###AllPoSub'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

rwpath = '/media/adrien/LaCie/PoSub-UPState/Project/Data'

mean_decoding_error = np.load(rwpath + '/' + '_bayes_err.npy')

zscored = []

for k, s in enumerate(datasets):
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    rawpath = os.path.join(rwpath,s)
    
    shuffle = np.load(rawpath + '/' + s + '_shuffle.npy')
    
    bins = np.linspace(min(shuffle), max(shuffle), 20)    
    relcounts_all,_ = np.histogram(shuffle, bins)     
    p_rel = relcounts_all/sum(relcounts_all)
    
    plt.figure()
    plt.title(s)
    plt.axvline(mean_decoding_error[k], color = 'k', label = 'data', linewidth = 2)
    plt.stairs(p_rel, bins,  label = 'shuffle', color = 'orange', linewidth =  2)
    plt.xlabel('DV decoded ang diff (rad)')
    plt.ylabel('% events')
    plt.legend(loc = 'upper right')
    
    z = (mean_decoding_error[k] - np.mean(shuffle)) / (np.std(shuffle)) 
    zscored.append(z)
    
#%% 



sns.set_style('white')
palette = ['peru']
ax = sns.violinplot( data = zscored, dodge=False,
                    palette = palette,cut = 2,
                      scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(data = zscored, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.swarmplot(data = zscored, color = 'k', dodge=False, ax=ax)
# # sns.stripplot(x = b['type'], y=b['corr'], data=b, color = 'k', dodge=False, ax=ax)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Z-scored difference')
plt.axhline(0, color = 'silver', linestyle = '--')
ax.set_box_aspect(1)
