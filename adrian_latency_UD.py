#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 14:17:35 2021

@author: dhruv
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
from scipy.stats import kendalltau, pearsonr,mannwhitneyu  

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
# datasets = np.loadtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
datasets = np.loadtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'

allcorrs_pyr = []
allcorrs_int = []
allcorrs_hd = []
nonparmean_int = []
nonparmean_pyr = []

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
    # COMPUTE MEAN FIRING RATES
###############################################################################################  
       
    rates = computeMeanFiringRate(spikes, [new_wake_ep, new_sws_ep, up_ep], ['wake', 'sws', 'up'])
    
    #Find the center of the down state
    down = np.zeros(len(down_ep))
    
    for i in range(len(down_ep)): 
        down[i] = down_ep.iloc[i,:].mean()
    

#find the spike in UP state closest to the center of DOWN state 
#compute correlation between spike time and cell depth

    pcorr_pyr = []
    pp_pyr = []
    
    pcorr_int = []
    pp_int = []

    pcorr_hd = []
    pp_hd = []
    
    frcorrW_pyr = []
    frpvalsW_pyr = []
    
    frcorrW_int = []
    frpvalsW_int = []
    
    frcorrW_hd = []
    frpvalsW_hd = []
        
    frcorrS_pyr = []
    frpvalsS_pyr = []
    
    frcorrS_int = []
    frpvalsS_int = []
    
    frcorrS_hd = []
    frpvalsS_hd = []
    
    nonparcorr_pyr = []
    nonparpvals_pyr = []
 
    nonparcorr_int = []
    nonparpvals_int = []
 
    nonparcorr_hd = []
    nonparpvals_hd = []
 
    
    
###################
     
    spkIndex = np.zeros(len(spikes.keys()))
    corrDw = np.zeros((len(down)))
    
    
    for i in range(len(down)-1): 
        
        difftimes1 = np.zeros(len(spikes.keys()), dtype = np.int32)
        tokeep = np.zeros(len(spikes.keys()))
        dt = np.zeros(len(spikes.keys()))
                
        # Is the next Down state >500ms
                       
        if down[i+1] - down[i] > 500000:
            ref = down[i]
              
                
            for j in spikes.keys(): 
                while spkIndex[j] < len(spikes[j]) and spikes[j].index.values[int(spkIndex[j])] < ref:
                    spkIndex[j] += 1
                    
                if spikes[j].index.values[int(spkIndex[j]-1)] < ref:
                    difftimes1[j] = spikes[j].index.values[int(spkIndex[j]-1)] - ref
                    dt[j] = abs(difftimes1[j])/1000
                
    # Is latency <500ms If so, label the neuron as "tokeep"
    # compute correlation only with these neurons
                           
                if dt[j] != np.nan and (dt[j] > 0 and dt[j] < 500):
                    tokeep[j] = 1         
          
                
            keep_idx = np.where(tokeep==1)
            pyr_keep = list(set.intersection(set(list(keep_idx[0])),set((pyr))))
            int_keep = list(set.intersection(set(list(keep_idx[0])),set((interneuron))))
            hd_keep = list(set.intersection(set(list(keep_idx[0])),set((hd))))
            
            if len(pyr_keep)>5:
            
                # fcW_pyr,fpW_pyr = pearsonr(dt[pyr_keep],rates['wake'][pyr_keep])    
                # frcorrW_pyr.append(fcW_pyr)
                # frpvalsW_pyr.append(fpW_pyr)
            
                # fcS_pyr,fpS_pyr = pearsonr(dt[pyr_keep],rates['sws'][pyr_keep])    
                # frcorrS_pyr.append(fcS_pyr)
                # frpvalsS_pyr.append(fpS_pyr)
            
                corr_pyr,p_pyr = kendalltau(dt[pyr_keep],depth[pyr_keep])    
                nonparcorr_pyr.append(corr_pyr)
                nonparpvals_pyr.append(p_pyr)
                allcorrs_pyr.append(nonparcorr_pyr)
                    
                
            if len(int_keep)>5: 
                # fcW_int,fpW_int = pearsonr(dt[int_keep],rates['wake'][int_keep])    
                # frcorrW_int.append(fcW_int)
                # frpvalsW_int.append(fpW_int)
                
                # fcS_int,fpS_int = pearsonr(dt[int_keep],rates['sws'][int_keep])    
                # frcorrS_int.append(fcS_int)
                # frpvalsS_int.append(fpS_int)
            
                corr_int,p_int = kendalltau(dt[int_keep],depth[int_keep])    
                nonparcorr_int.append(corr_int)
                nonparpvals_int.append(p_int)
                allcorrs_int.append(nonparcorr_int)
                    
                
            # if len(hd_keep)>5: 
                # fcW_hd,fpW_hd = pearsonr(dt[hd_keep],rates['wake'][hd_keep])    
                # frcorrW_hd.append(fcW_hd)
                # frpvalsW_hd.append(fpW_hd)
                
                # fcS_hd,fpS_hd = pearsonr(dt[hd_keep],rates['sws'][hd_keep])    
                # frcorrS_hd.append(fcS_hd)
                # frpvalsS_hd.append(fpS_hd)
            
                # corr_hd,p_hd = kendalltau(dt[hd_keep],depth[hd_keep])    
                # nonparcorr_hd.append(corr_hd)
                # nonparpvals_hd.append(p_hd)
                # allcorrs_hd.append(nonparcorr_hd)
                    
           
            
#################################################################################
###PLOTS
#################################################################################

############################################################################################### 
    # FIRING RATE AND LATENCY CORR
###############################################################################################  
               
            
            
    # plt.figure()
    # plt.title('Correlation between wake FR and latency to last spike (PYR)_' + name)
    # plt.hist(frcorrW_pyr,np.linspace(-1,1), label = 'Mean = ' + str(round(np.mean(frcorrW_pyr),4)))
    # plt.xlabel('Pearson r value')
    # plt.ylabel('Frequency')
    # plt.legend(loc = 'upper right')
   
    # plt.figure()
    # plt.title('Correlation between wake FR and latency to last spike (INT)_' + name)
    # plt.hist(frcorrW_int,np.linspace(-1,1), label = 'Mean = ' + str(round(np.mean(frcorrW_int),4)))
    # plt.xlabel('Pearson r value')
    # plt.ylabel('Frequency')
    # plt.legend(loc = 'upper right')
    
    # plt.figure()
    # plt.title('Correlation between wake FR and latency to last spike (HD)_' + name)
    # plt.hist(frcorrW_hd,np.linspace(-1,1), label = 'Mean = ' + str(round(np.mean(frcorrW_hd),4)))
    # plt.xlabel('Pearson r value')
    # plt.ylabel('Frequency')
    # plt.legend(loc = 'upper right')
    
    # plt.figure()
    # plt.title('Correlation between NREM FR and latency to last spike (PYR)_' + name)
    # plt.hist(frcorrS_pyr,np.linspace(-1,1), label = 'Mean = ' + str(round(np.mean(frcorrS_pyr),4)))
    # plt.xlabel('Pearson r value')
    # plt.ylabel('Frequency')
    # plt.legend(loc = 'upper right')
    
    # plt.figure()
    # plt.title('Correlation between NREM FR and latency to last spike (INT)_' + name)
    # plt.hist(frcorrS_int,np.linspace(-1,1), label = 'Mean = ' + str(round(np.mean(frcorrS_int),4)))
    # plt.xlabel('Pearson r value')
    # plt.ylabel('Frequency')
    # plt.legend(loc = 'upper right')
    
    # plt.figure()
    # plt.title('Correlation between NREM FR and latency to last spike (HD)_' + name)
    # plt.hist(frcorrS_hd,np.linspace(-1,1), label = 'Mean = ' + str(round(np.mean(frcorrS_hd),4)))
    # plt.xlabel('Pearson r value')
    # plt.ylabel('Frequency')
    # plt.legend(loc = 'upper right')               


############################################################################################### 
    # LATENCY AND DEPTH CORR
###############################################################################################  
    
    plt.figure()
    plt.title('Correlation between latency to last spike and cell depth_' + name)
    plt.hist(nonparcorr_pyr,np.linspace(-1,1), alpha = 0.5, label = 'Mean (ex) = ' + str(round(np.nanmean(nonparcorr_pyr),4)))
    nonparmean_pyr.append(round(np.nanmean(nonparcorr_pyr),4))
    plt.hist(nonparcorr_int,np.linspace(-1,1), alpha = 0.5, label = 'Mean (FS) = ' + str(round(np.nanmean(nonparcorr_int),4)))
    nonparmean_int.append(round(np.nanmean(nonparcorr_int),4))
    plt.xlabel('Kendall tau value')
    plt.ylabel('Frequency')
    plt.legend(loc = 'upper right')          
    plt.show()
    
    # plt.figure()
    # plt.title('Correlation between latency to last spike and cell depth (HD)_' + name)
    # plt.hist(nonparcorr_hd,np.linspace(-1,1), label = 'Mean = ' + str(round(np.mean(nonparcorr_hd),4)))
    # plt.xlabel('Kendall tau value')
    # plt.ylabel('Frequency')
    # plt.legend(loc = 'upper right')          
    # plt.show()

t,pvalue = mannwhitneyu(nonparmean_pyr,nonparmean_int)
means_ex = np.nanmean(nonparmean_pyr)
means_inh = np.nanmean(nonparmean_int)

label = ['All sessions']
x = np.arange(len(label))  # the label locations
width = 0.35  # the width of the bars


fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, means_ex, width, label='Ex')
rects2 = ax.bar(x + width/2, means_inh, width, label='FS')

pval = np.vstack([(nonparmean_pyr), (nonparmean_int)])

x2 = [x-width/2, x+width/2]
plt.plot(x2, np.vstack(pval), 'o-', color = 'k')

# # Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Pearson r value')
ax.set_title('Latency to last spike v/s depth correlation')
ax.set_xticks(x)
ax.set_xticklabels(label)
ax.legend(loc = 'upper right')

fig.tight_layout()


    
    
