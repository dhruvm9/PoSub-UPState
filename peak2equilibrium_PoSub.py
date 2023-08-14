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

#Parameters
##Ex cells, norm = True, period = 0- 100 ms, UP onset ---> 9/16 significant 


#%% On Lab PC
data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
# datasets = np.loadtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')
# datasets = np.loadtxt(os.path.join(data_directory,'dataset_Ver_DM.list'), delimiter = '\n', dtype = str, comments = '#')

rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'

corrs = []
pvals = []

corrs_sws = []
pvals_sws = []

ratecorrs = []
ratepvals = []

peak_above_mean = []
peaktiming = []
uponset = []
allrates = []

depthcorrs = []
depthpvals = []

timingcorrs = []
timingpvals = [] 

for s in datasets:
    print(s)
    name = s.split('/')[-1]
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
    tmp = tmp.rolling(window=8, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
    # dd2 = tmp[0:0.255]
    dd2 = tmp[0:0.155]  
            
    rates = spikes[pyr].restrict(up_ep)._metadata['rate'].values
    
    # #Excitatory cells only 
    ee = dd2[pyr] 
    
    if len(ee.columns) > 0:
                    
        indexplot_ex = []
        depths_keeping_ex = []
        peaks_keeping_ex = []
        rates_keeping_ex = []
        peaktiming_ex = []                   
            
        for i in range(len(ee.columns)):
            a = np.where(ee.iloc[:,i] > 0.5)
            if len(a[0]) > 0:
              peaks_keeping_ex.append(ee.iloc[:,i].max())
              peak_above_mean.append(ee.iloc[:,i].max())
              peaktiming_ex.append(ee.iloc[:,i].idxmax())
              peaktiming.append(ee.iloc[:,i].idxmax())
              depths_keeping_ex.append(depth.flatten()[ee.columns[i]])
              rates_keeping_ex.append(rates[i])
              allrates.append(rates[i])
                
              res = ee.iloc[:,i].index[a]
              indexplot_ex.append(res[0])
              uponset.append(res[0])
      
#%% 

#     a = []
#     for i in ee.columns:
#         if ee[i].values[-1] < 0.9:
#             a.append(i)  
      
# #%%    
# #Best examples 
    # plt.figure()
    # plt.plot(tmp[pyr][23][-0.05:0.2])
    # plt.plot(tmp[pyr][50][-0.05:0.2])
    # plt.axhline(1, linestyle = '--', color = 'silver')
    # plt.axvline(0, color = 'k')
    # plt.yticks([])
    # plt.gca().set_box_aspect(1)
   

#%%             
    # corr, p = kendalltau(indexplot_ex, peaks_keeping_ex)
    
    corr, p = kendalltau(indexplot_ex, peaks_keeping_ex)
    corrs.append(corr)
    pvals.append(p)
        
         
    plt.figure()
    plt.rc('font', size = 15)
    plt.title('Peak-mean rate ratio v/s UP onset_' + s)
    plt.scatter(indexplot_ex, peaks_keeping_ex, label = 'R = ' + str((round(corr,2))), color = 'cornflowerblue')
    plt.xlabel('Time from UP onset (s)')
    plt.ylabel('Peak-mean rate ratio')
    plt.legend(loc = 'upper right')
    plt.gca().set_box_aspect(1)

#%% 

    depthcorr, depthp = kendalltau(peaks_keeping_ex, depths_keeping_ex)
    depthcorrs.append(depthcorr)
    depthpvals.append(depthp)

    # plt.figure()
    # plt.title('Peak/ mean FR v/s Depth_' + s)
    # plt.scatter(peaks_keeping_ex,depths_keeping_ex, label = 'R = ' + str((round(depthcorr,2))))
    # plt.xlabel('Peak/mean FR')
    # plt.ylabel('Depth from top of probe (um)')
    # plt.legend(loc = 'upper right')
    
#%% 

    ratecorr, ratep = kendalltau(rates_keeping_ex, indexplot_ex)
    ratecorrs.append(ratecorr)
    ratepvals.append(ratep)
    
    # plt.figure()
    # plt.title('NREM FR v/s UP onset delay_' + s)
    # plt.scatter(rates_keeping_ex,indexplot_ex, label = 'R = ' + str((round(ratecorr,2))))
    # plt.xlabel('Mean NREM FR')
    # plt.ylabel('UP onset delay (ms)')
    # plt.legend(loc = 'upper right')

#%% 

    timingcorr, timingp = kendalltau(peaktiming_ex, indexplot_ex)
    timingcorrs.append(timingcorr)
    timingpvals.append(timingp)

    # plt.figure()
    # plt.title('Timing of peak v/s UP onset delay_' + s)
    # plt.scatter(peaktiming_ex,indexplot_ex, label = 'R = ' + str((round(timingcorr,2))))
    # plt.xlabel('Timing of peak FR')
    # plt.ylabel('UP onset delay (ms)')
    # plt.legend(loc = 'upper right')
    # plt.gca().set_box_aspect(1)
            
#%% Pooled plot         

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
r2, p2 = kendalltau(peaktiming, uponset)

plt.figure()
plt.rc('font', size = 15)
plt.title('NREM FR v/s UP onset: HDC pooled data')
sns.kdeplot(x = allrates, y = uponset, color = 'cornflowerblue')
plt.scatter(allrates, uponset, label = 'R = ' + str((round(r1,2))), color = 'cornflowerblue', s = 4)
plt.xlabel('NREM FR (Hz)')
plt.ylabel('UP onset delay (s)')
plt.legend(loc = 'upper right')
plt.gca().set_box_aspect(1)

plt.figure()
plt.rc('font', size = 15)
plt.title('Timing of peak FR v/s UP onset: HDC pooled data')
sns.kdeplot(x = peaktiming, y = uponset, color = 'cornflowerblue')
plt.scatter(peaktiming, uponset, label = 'R = ' + str((round(r2,2))), color = 'cornflowerblue', s = 4)
plt.xlabel('Timing of peak FR (Hz)')
plt.ylabel('UP onset delay (s)')
plt.legend(loc = 'upper right')
plt.gca().set_box_aspect(1)


    
#%% 

##Summary plot 

summary = pd.DataFrame()
summary['corr'] = corrs
summary['p'] = pvals
summary['depthcorr'] = depthcorrs
summary['depthp'] = depthpvals 

plt.figure()
plt.boxplot(corrs, positions=[0], showfliers=False, patch_artist=True, boxprops=dict(facecolor='royalblue', color='royalblue'),
              capprops=dict(color='royalblue'),
              whiskerprops=dict(color='royalblue'),
              medianprops=dict(color='white', linewidth = 2))
# plt.boxplot(depthcorrs, positions=[0.3], showfliers=False, patch_artist=True, boxprops=dict(facecolor='lightsteelblue', color='lightsteelblue'),
#               capprops=dict(color='lightsteelblue'),
#               whiskerprops=dict(color='lightsteelblue'),
#               medianprops=dict(color='white', linewidth = 2))

x1 = np.random.normal(0, 0.01, size=len(summary['corr'][summary['p'] < 0.05]))
x2 = np.random.normal(0, 0.01, size=len(summary['corr'][summary['p'] >= 0.05]))
# x3 = np.random.normal(0.3, 0.01, size=len(summary['depthcorr'][summary['depthp'] < 0.05]))
# x4 = np.random.normal(0.3, 0.01, size=len(summary['depthcorr'][summary['depthp'] >= 0.05]))

plt.plot(x1, summary['corr'][summary['p'] < 0.05], 'x', color = 'k', fillstyle = 'none', markersize = 6, zorder =3, label = 'p < 0.05')
plt.plot(x2, summary['corr'][summary['p'] >= 0.05], '.', color = 'k', fillstyle = 'none', markersize = 6, zorder =3,  label = 'p >= 0.05')
# plt.plot(x3, summary['depthcorr'][summary['depthp'] < 0.05], 'x', color = 'k', fillstyle = 'none', markersize = 6, zorder =3)
# plt.plot(x4, summary['depthcorr'][summary['depthp'] >= 0.05], '.', color = 'k', fillstyle = 'none', markersize = 6, zorder =3)
# plt.axhline(0, color = 'silver')
# plt.xticks([0, 0.3],['vs delay', 'vs depth'])
plt.xticks([])
# plt.title('Peak/mean FR v/s UP onset - Summary')
plt.legend(loc = 'upper right')
plt.ylabel('Peak/mean v/s UP onset (R)')




    
        





