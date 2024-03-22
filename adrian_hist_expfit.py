#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 12:32:46 2024

@author: dhruv
"""

#loading the dataset
import numpy as np 
import pandas as pd 
import scipy.io
from functions import *
from wrappers import *
import os, sys
import time 
import matplotlib.pyplot as plt 
from Wavelets import MyMorlet as Morlet
from scipy.stats import kendalltau, pearsonr, wilcoxon, mannwhitneyu
from scipy.optimize import curve_fit
import seaborn as sns

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'

allcoefs_up = []
allcoefs_up_ex = []
allspeeds_up = []
allspeeds_up_ex = []
pvals = []
pvals_ex = []
N_units = []
N_ex = []
N_hd = [] 

range_DUonset = []
allDU = []

expfit = []
linfit = []

for s in datasets:
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    rawpath = os.path.join(rwpath,s)

    spikes, shank = loadSpikeData(path)
    N_units.append(len(spikes))
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

    N_ex.append(len(pyr))
    N_hd.append(len(hd))
# ############################################################################################### 
#     # LOAD UP AND DOWN STATE, NEW SWS AND NEW WAKE EPOCHS
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
    # COMPUTE EVENT CROSS CORRS
###############################################################################################  
    binsize = 5
    nbins = 1000        
    neurons = list(spikes.keys())
    times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2        
    cc = pd.DataFrame(index = times, columns = neurons)
    tsd_up = up_ep.as_units('ms').start.values
  
#UP State    
    ep_U = nts.IntervalSet(start = up_ep.start[0], end = up_ep.end.values[-1])
                  
    rates = []
    
    sess_DU = []
               
    for i in neurons:
        # spk2 = spikes[i].restrict(ep_U).as_units('ms').index.values
        spk2 = spikes[i].restrict(up_ep).as_units('ms').index.values
        tmp = crossCorr(tsd_up, spk2, binsize, nbins)
        tmp = pd.DataFrame(tmp)
        tmp = tmp.rolling(window=8, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
        
        # fr = len(spk2)/ep_U.tot_length('s')
        fr = len(spk2)/up_ep.tot_length('s')
        rates.append(fr)
        cc[i] = tmp.values
        cc[i] = tmp.values/fr
                     
        dd = cc[0:150]
        # dd = cc[0:250]
               
    #Cell types 
    ee = dd[pyr]
            
#######Ex cells 
    if len(ee.columns) > 0:
                    
        indexplot_ex = []
        depths_keeping_ex = []
                
            
        for i in range(len(ee.columns)):
            a = np.where(ee.iloc[:,i] > 0.5)
            if len(a[0]) > 0:
                depths_keeping_ex.append(depth.flatten()[ee.columns[i]])
                res = ee.iloc[:,i].index[a]
                indexplot_ex.append(res[0])
                allDU.append(res[0])
            
        sess_DU.append(indexplot_ex)
        
        
        range_DUonset.append(np.std(sess_DU))
        
        #Latency v/s depth 
        coef_ex, p_ex = kendalltau(indexplot_ex,depths_keeping_ex)
        
        pvals_ex.append(p_ex)
        allcoefs_up_ex.append(coef_ex)
        
         
        ###LINEAR FIT
        
        y_est_ex = np.zeros(len(depths_keeping_ex))
        m_ex, b_ex = np.polyfit(indexplot_ex, depths_keeping_ex, 1)
        allspeeds_up_ex.append(m_ex)
        
        for i in range(len(indexplot_ex)):
            y_est_ex[i] = m_ex*indexplot_ex[i]
            
        ###EXPONENTIAL FIT
        
        def exp_fn(x, m, t, b):
            return m*np.exp(-t*x) + b
        
        def lin_fn(x, m, b):
            return m*x + b
        
        popt, pcov = curve_fit(exp_fn, np.array(indexplot_ex), np.array(depths_keeping_ex), p0 = [800, 1/30, -800], maxfev = 5000)
        m_opt, t_opt, b_opt = popt
               
        popt, pcov = curve_fit(lin_fn, np.array(indexplot_ex), np.array(depths_keeping_ex), p0 = [-7, 0], maxfev = 5000)
        m_opt2, b_opt2 = popt
        
        
        squaredDiffs = np.square(depths_keeping_ex - exp_fn(np.array(indexplot_ex), m_opt, t_opt, b_opt))
        squaredDiffsFromMean = np.square(depths_keeping_ex - np.mean(depths_keeping_ex))
        rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
        expfit.append(rSquared)
        
        squaredDiffs = np.square(depths_keeping_ex - lin_fn(np.array(indexplot_ex), m_opt2, b_opt2))
        squaredDiffsFromMean = np.square(depths_keeping_ex - np.mean(depths_keeping_ex))
        rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
        linfit.append(rSquared)
        
               
        x_fitted = np.linspace(np.min(indexplot_ex), np.max(indexplot_ex), len(indexplot_ex))
            
        ###PLOTS
        plt.figure()
        plt.scatter(indexplot_ex, depths_keeping_ex, color = 'cornflowerblue')
        # plt.plot(indexplot_ex, y_est_ex + b_ex, color = 'cornflowerblue')
        plt.plot(x_fitted, exp_fn(x_fitted, m_opt, t_opt, b_opt), '--', color = 'k')
        plt.plot(x_fitted, lin_fn(x_fitted, m_opt2, b_opt2), '--', color = 'cornflowerblue')
        plt.title('Bin where FR > 50% baseline rate_' + s)
        plt.ylabel('Depth from top of probe (um)')
        plt.yticks([0, -400, -800])
        plt.xlabel('Lag (ms)')
  
        
        # sys.exit()
    
#%% Out of loop 

plt.figure()
plt.boxplot(allcoefs_up_ex, positions=[0], showfliers=False, patch_artist=True, boxprops=dict(facecolor='royalblue', color='royalblue'),
              capprops=dict(color='royalblue'),
              whiskerprops=dict(color='royalblue'),
              medianprops=dict(color='white', linewidth = 2))
plt.boxplot(expfit, positions=[0.3], showfliers=False, patch_artist=True, boxprops=dict(facecolor='lightsteelblue', color='lightsteelblue'),
              capprops=dict(color='lightsteelblue'),
              whiskerprops=dict(color='lightsteelblue'),
              medianprops=dict(color='white', linewidth = 2))

x1 = np.random.normal(0, 0.01, size=len(allcoefs_up_ex))
x2 = np.random.normal(0.3, 0.01, size=len(expfit))
plt.plot(x1, allcoefs_up_ex, '.', color = 'k', fillstyle = 'none', markersize = 6, zorder =3)
plt.plot(x2, expfit, '.', color = 'k', fillstyle = 'none', markersize = 6, zorder =3)
plt.axhline(0, color = 'silver')
plt.xticks([0, 0.3],['Linear fit R', 'Exp. R squared'])


#%% 

# plt.figure()
# plt.scatter(linfit, expfit)
# plt.xlabel('Linear R^2')
# plt.ylabel('Exponential R^2')
# plt.gca().axline((min(min(linfit),min(expfit)),min(min(linfit),min(expfit)) ), slope=1, color = 'silver', linestyle = '--')
# plt.gca().set_box_aspect(1)