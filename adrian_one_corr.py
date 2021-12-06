# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:21:34 2020

@author: Dhruv
"""

#loading the dataset
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
from scipy.stats import kendalltau, pearsonr, wilcoxon

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
#datasets = np.loadtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
datasets = np.loadtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'

allcoefs_up = []
allcoefs_dn = []

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
    tsd_dn = down_ep.as_units('ms').start.values
  
#UP State    
    ep_U = nts.IntervalSet(start = up_ep.start[0], end = up_ep.end.values[-1])
    ep_D = nts.IntervalSet(start = down_ep.start[0], end = down_ep.end.values[-1])
       
    # rates = []
    
    # for i in neurons:
    #     spk2 = spikes[i].restrict(ep_U).as_units('ms').index.values
    #     tmp = crossCorr(tsd_up, spk2, binsize, nbins)
    #     fr = len(spk2)/ep_U.tot_length('s')
    #     rates.append(fr)
    #     cc[i] = tmp
    #     cc[i] = tmp/fr

    # dd = cc[-250:250]
    
    # #Cell types 
    # #ee = dd[pyr]
    # ee = dd[interneuron]
    
    # n = len(depth)
    # #tmp = np.argsort(depth[pyr].flatten())
    # tmp = np.argsort(depth[interneuron].flatten())
    # #tmp = np.argsort(depth.flatten())
    # desc = tmp[::-1][:n]
    
    # order = []
    # # for i in range(len(pyr)): 
    # #     order.append(pyr[desc[i]])
    
    # for i in range(len(interneuron)): 
    #     order.append(interneuron[desc[i]])
    
        
    # finalRates = ee[order]
    # # finalRates = dd[desc]
    
    # if len(ee.columns) > 5:
    # # if len(dd.columns) > 5:
        
    #     plt.figure()
    #     plt.imshow(finalRates.T,extent=[-250 , 250, len(interneuron) , 1],aspect = 'auto', cmap = 'jet')        
    #     # plt.imshow(finalRates.T,extent=[-250 , 250, len(neurons) , 1],aspect = 'auto', cmap = 'jet')        
    #     # plt.imshow(finalRates.T,extent=[-250 , 250, len(pyr) , 1],aspect = 'auto', cmap = 'jet')        
    #     plt.clim(0,5)
    #     plt.colorbar()
    #     plt.title('Event-related Xcorr, aligned to UP state onset_' + s)
    #     plt.ylabel('Neuron number')
    #     plt.xlabel('Lag (ms)')

    #     thresholdrate = 1.5 * np.array(rates)

    #     bin_50peak = []
        
    #     # for i in neurons:
    #     # for i in pyr:
    #     for i in interneuron: 
    #         threshold = thresholdrate[i]
    #         neuron_rate = ee[i]
    #         # neuron_rate = dd[i]
    #         diff = abs(neuron_rate - threshold)
    #         ix = diff.idxmin()
    #         bin_50peak.append(ix)
    
    #     # posIxOnly = np.zeros((len(neurons),2))
    #     # posIxOnly[:,0] = bin_50peak
    #     # posIxOnly[:,1] = depth.flatten()
            
    #     posIxOnly = np.zeros((len(interneuron),2))
    #     posIxOnly[:,0] = bin_50peak
    #     posIxOnly[:,1] = depth[interneuron].flatten()
        
    #     # posIxOnly = np.zeros((len(pyr),2))
    #     # posIxOnly[:,0] = bin_50peak
    #     # posIxOnly[:,1] = depth[pyr].flatten()
        
        
    #     pos_ix = np.where(posIxOnly[:,0] > 0)
    #     n = posIxOnly[pos_ix]
    #     n1 = n[:,0]
    #     n2 = n[:,1]
            
    #     print(len(n))
            
    #     if len(n) > 5:
    #         coef, p = kendalltau(n1,n2)
    #         allcoefs_up.append(coef)
                
    #         plt.figure()
    #         plt.scatter(n1,n2, label = 'Kendall tau = ' + str(round(coef,4)))
    #         plt.title('Bin where FR > 50% baseline rate_' + s)
    #         plt.ylabel('Depth from top of probe (um)')
    #         plt.xlabel('Lag (ms)')
    #         plt.legend(loc = 'upper right')
            
#DOWN state
    rates = []
    
    for i in neurons:
        spk2 = spikes[i].restrict(ep_D).as_units('ms').index.values
        tmp = crossCorr(tsd_dn, spk2, binsize, nbins)
        fr = len(spk2)/ep_D.tot_length('s')
        rates.append(fr)
        cc[i] = tmp
        cc[i] = tmp/fr

    dd = cc[-250:250]
    
    #Cell types 
    ee = dd[pyr]
    #ee = dd[interneuron]
    
    n = len(depth)
    tmp = np.argsort(depth[pyr].flatten())
    # tmp = np.argsort(depth[interneuron].flatten())
    #tmp = np.argsort(depth.flatten())
    desc = tmp[::-1][:n]
    
    order = []
    for i in range(len(pyr)): 
        order.append(pyr[desc[i]])
    
    # for i in range(len(interneuron)): 
    #     order.append(interneuron[desc[i]])
    
        
    finalRates = ee[order]
    # finalRates = dd[desc]
    
    if len(ee.columns) > 5:
    # if len(dd.columns) > 5:
        
        plt.figure()
        # plt.imshow(finalRates.T,extent=[-250 , 250, len(interneuron) , 1],aspect = 'auto', cmap = 'jet')        
        # plt.imshow(finalRates.T,extent=[-250 , 250, len(neurons) , 1],aspect = 'auto', cmap = 'jet')        
        plt.imshow(finalRates.T,extent=[-250 , 250, len(pyr) , 1],aspect = 'auto', cmap = 'jet')        
        plt.clim(0,5)
        plt.colorbar()
        plt.title('Event-related Xcorr, aligned to DOWN state onset_' + s)
        plt.ylabel('Neuron number')
        plt.xlabel('Lag (ms)')

        #sys.exit()
        
        indexplot = [] 
        
        for i in range(len(finalRates.columns)):
            a = np.where(finalRates.iloc[:,i] < 0.5)
            res = finalRates.iloc[:,i].index[a]
            indexplot.append(res[0])
            
            
        coef, p = kendalltau(indexplot,depth[pyr].flatten())
            # allcoefs_up.append(coef)
        allcoefs_dn.append(coef)
            
        plt.figure()
        plt.scatter(indexplot,depth[pyr].flatten(), label = 'Kendall tau = ' + str(round(coef,4)))
        plt.title('Bin where FR < 50% baseline rate_' + s)
        plt.ylabel('Depth from top of probe (um)')
        plt.xlabel('Lag (ms)')
        plt.legend(loc = 'upper right')
            
            
        # thresholdrate = 0.5 * np.array(rates)

        # bin_50peak = []
        
        # # for i in neurons:
        # for i in pyr:
        # # for i in interneuron: 
        #     threshold = thresholdrate[i]
        #     neuron_rate = ee[i]
        #     # neuron_rate = dd[i]
        #     diff = abs(neuron_rate - threshold)
        #     ix = diff.idxmin()
        #     bin_50peak.append(ix)
    
        # # posIxOnly = np.zeros((len(neurons),2))
        # # posIxOnly[:,0] = bin_50peak
        # # posIxOnly[:,1] = depth.flatten()
            
        # # posIxOnly = np.zeros((len(interneuron),2))
        # # posIxOnly[:,0] = bin_50peak
        # # posIxOnly[:,1] = depth[interneuron].flatten()
        
        # posIxOnly = np.zeros((len(pyr),2))
        # posIxOnly[:,0] = bin_50peak
        # posIxOnly[:,1] = depth[pyr].flatten()
        
        
        # # pos_ix = np.where(posIxOnly[:,0] < 0)
        # # n = posIxOnly[pos_ix]
        # # n1 = n[:,0]
        # # n2 = n[:,1]
            
        # print(len(posIxOnly))
            
        # if len(posIxOnly) > 5:
        #     coef, p = kendalltau(posIxOnly[:,0],posIxOnly[:,1])
        #     # allcoefs_up.append(coef)
        #     allcoefs_dn.append(coef)
            
        #     plt.figure()
        #     plt.scatter(posIxOnly[:,0],posIxOnly[:,1], label = 'Kendall tau = ' + str(round(coef,4)))
        #     plt.title('Bin where FR < 50% baseline rate_' + s)
        #     plt.ylabel('Depth from top of probe (um)')
        #     plt.xlabel('Lag (ms)')
        #     plt.legend(loc = 'upper right')

    

#Out of loop 

# z_up, p_up = wilcoxon(np.array(allcoefs_up)-0)
# plt.figure()
# plt.hist(allcoefs_up,label = 'p-value =' + str(round(p_up,4)))
# plt.axvline(np.mean(allcoefs_up), color = 'k')
# plt.title('Distribution of Kendall Tau for UP-state onset')
# plt.xlabel('Kendall tau value') rates = []
  
z_dn, p_dn = wilcoxon(np.array(allcoefs_dn)-0)
plt.figure()
plt.hist(allcoefs_dn,label = 'p =' + str(round(p_dn,4)))
plt.axvline(np.mean(allcoefs_dn), color = 'k')
plt.title('Distribution of Kendall Tau for DOWN-state onset')
plt.xlabel('Kendall tau value')
plt.ylabel('Number of sessions')
plt.legend(loc = 'upper right')    
            
