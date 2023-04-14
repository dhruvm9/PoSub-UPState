#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 17:32:28 2022

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
from itertools import combinations
import seaborn as sns

#%% 

corrs = []
pvals = []

ratecorrs = []
ratepvals = [] 

allrates = [] 

uponset = []
peak_above_mean = []

allPETH = pd.DataFrame()

data_directory = '/media/DataDhruv/Recordings/WatsonBO'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')

for s in datasets:
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)

    listdir    = os.listdir(path)
    file = [f for f in listdir if 'spikes' in f]
    spikedata = scipy.io.loadmat(os.path.join(path,file[0]))  
    
    listdir    = os.listdir(path)
    file = [f for f in listdir if 'events' in f]
    events = scipy.io.loadmat(os.path.join(path,file[0]))  
    
    listdir    = os.listdir(path)
    file = [f for f in listdir if 'states' in f]
    states = scipy.io.loadmat(os.path.join(path,file[0])) 
    
    listdir    = os.listdir(path)
    file = [f for f in listdir if 'CellClass' in f]
    cellinfo = scipy.io.loadmat(os.path.join(path,file[0])) 


#%% 
    #Load EX cells
    ex = cellinfo['CellClass'][0][0][1][0]
    pyr = []
    
    for i in range(len(ex)):
        if ex[i] == 1:
            pyr.append(i)

    #Load Spikes (only EX cells)

    spks = spikedata['spikes']
    spk = {}
    for i in range(len(spks[0][0][1][0])):
        spk[i] = nap.Ts(spks[0][0][1][0][i])
    spikes = nap.TsGroup(spk)
    spikes = spikes[pyr]
    
    #Load Sleep States 
    
    sleepstate = states['SleepState']
    wake_ep = nap.IntervalSet(start = sleepstate[0][0][0][0][0][0][:,0], end = sleepstate[0][0][0][0][0][0][:,1])
    nrem_ep = nap.IntervalSet(start = sleepstate[0][0][0][0][0][1][:,0], end = sleepstate[0][0][0][0][0][1][:,1])
    rem_ep = nap.IntervalSet(start = sleepstate[0][0][0][0][0][2][:,0], end = sleepstate[0][0][0][0][0][2][:,1])
    
    
    #Load UP and DOWN states 
    
    slowwaves = events['SlowWaves']
    up_ep = nap.IntervalSet( start = slowwaves[0][0][2][0][0][0][:,0], end = slowwaves[0][0][2][0][0][0][:,1])
    down_ep = nap.IntervalSet( start = slowwaves[0][0][2][0][0][1][:,0], end = slowwaves[0][0][2][0][0][1][:,1])
                     
    

#%% 

# COMPUTE EVENT CROSS CORRS
## Peak firing       

### UD 
    
    cc = nap.compute_eventcorrelogram(spikes, nap.Tsd(down_ep['start'].values), binsize = 0.005, windowsize = 0.255, ep = up_ep, norm = True)
    tmp = pd.DataFrame(cc)
    tmp = tmp.rolling(window=8, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
    
    # plt.imshow(cc.T,  aspect = 'auto', cmap = 'inferno', origin = 'lower')
    


#%% 
### DU

    cc = nap.compute_eventcorrelogram(spikes, nap.Tsd(up_ep['start'].values), binsize = 0.005, windowsize = 0.255, ep = up_ep, norm = True)
    tmp = pd.DataFrame(cc)
    tmp = tmp.rolling(window=8, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
    dd = tmp[0:0.155]
    
    rates = spikes.restrict(up_ep)._metadata['rate'].values
    
    if len(dd.columns) > 0:
                    
        indexplot = []
        tokeep = []
        peaks_keeping = []
        rates_keeping = []
                
            
        for i in range(len(dd.columns)):
            a = np.where(dd.iloc[:,i] > 0.5)
            if len(a[0]) > 0:
                tokeep.append(dd.columns[i])  
                peaks_keeping.append(dd.iloc[:,i].max())
                peak_above_mean.append(dd.iloc[:,i].max())
                res = dd.iloc[:,i].index[a]
                indexplot.append(res[0])
                uponset.append(res[0])
                rates_keeping.append(rates[i])
                allrates.append(rates[i])

    allPETH = pd.concat([allPETH, dd[tokeep]], axis = 1)
        
    corr, p = kendalltau(indexplot, peaks_keeping)
    corrs.append(corr)
    pvals.append(p)

#%% 
        
    # plt.figure()
    # plt.rc('font', size = 15)
    # plt.title('Peak-mean rate ratio v/s UP onset_' + s)
    # plt.scatter(indexplot,peaks_keeping, label = 'R = ' + str((round(corr,2))), color = 'cornflowerblue')
    # plt.xlabel('Time from UP onset (s)')
    # plt.ylabel('Peak-mean rate ratio')
    # plt.legend(loc = 'upper right')

#%% 

#Arrange DU by peak-to-mean

    # idx = np.argsort(peaks_keeping)
    # plt.imshow(dd[dd.columns[idx]].T, aspect = 'auto', cmap = 'inferno', 
    #            extent =[0,150,len(spikes),1],
    #            origin = 'lower')



#%% 

    ratecorr, ratep = kendalltau(rates_keeping, indexplot)
    ratecorrs.append(ratecorr)
    ratepvals.append(ratep)

    # plt.figure()
    # plt.title('NREM FR v/s UP onset delay_' + s)
    # plt.scatter(rates_keeping_ex,indexplot_ex, label = 'R = ' + str((round(ratecorr,2))))
    # plt.xlabel('Mean NREM FR')
    # plt.ylabel('UP onset delay (ms)')
    # plt.legend(loc = 'upper right')

#%% 

binsize = 0.005
pooledcorr, pooledp = kendalltau(uponset, peak_above_mean)
(counts,onsetbins,peakbins) = np.histogram2d(uponset,peak_above_mean,bins=[len(np.arange(0,0.155,binsize))+1,len(np.arange(0,0.155,binsize))+1],
                                                 range=[[-0.0025,0.1575],[0.5,3.6]])

masked_array = np.ma.masked_where(counts == 0, counts)
cmap = plt.cm.viridis  # Can be any colormap that you want after the cm
cmap.set_bad(color='white')

plt.figure()
plt.imshow(masked_array.T, origin='lower', extent = [onsetbins[0],onsetbins[-1],peakbins[0],peakbins[-1]],
                                               aspect='auto', cmap = cmap)
plt.colorbar(ticks = [min(counts.flatten()),max(counts.flatten())])
plt.xlabel('UP onset delay (s)')
plt.ylabel('Peak-mean rate ratio')
plt.gca().set_box_aspect(1)



y_est = np.zeros(len(uponset))
m, b = np.polyfit(uponset, peak_above_mean, 1)
for i in range(len(uponset)):
     y_est[i] = m*uponset[i]

plt.plot(uponset, y_est + b, color = 'r')

r1, p1 = kendalltau(allrates,uponset)

plt.figure()
plt.rc('font', size = 15)
plt.title('NREM FR v/s UP onset: PFC pooled data')
sns.kdeplot(x = allrates, y = uponset, color = 'cornflowerblue')
plt.scatter(allrates, uponset, label = 'R = ' + str((round(r1,2))), color = 'cornflowerblue', s = 4)
plt.xlabel('NREM FR (Hz)')
plt.ylabel('UP onset delay (s)')
plt.legend(loc = 'upper right')
plt.legend(loc = 'upper right')
plt.gca().set_box_aspect(1)


#%% 

#Arrange DU by peak-to-mean

idx = np.argsort(peak_above_mean)
allPETH.columns = range(allPETH.columns.size)

plt.figure()
plt.imshow(allPETH[allPETH.columns[idx]].T, aspect = 'auto', cmap = 'inferno', 
            extent =[0,150,allPETH.columns.size,1],
            origin = 'lower', vmin = 0, vmax = 2)
plt.colorbar()
plt.plot([i * 1000 for i in uponset],np.arange(1,951,1),'o', color = 'w', markersize = 1)

#%% 

