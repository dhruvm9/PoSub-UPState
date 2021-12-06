#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 17:16:17 2021

@author: dhruv
"""
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 10:52:10 2020

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
from scipy.stats import kendalltau, pearsonr 

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/AllPoSubHor'
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

#find the channel corresponding to each neuron
    d1 = scipy.io.loadmat(rawpath +'/Analysis/Waveforms.mat')
    d2 = scipy.io.loadmat(rawpath +'/Analysis/WaveformFeatures.mat')
    
    for i in d1.keys(): 
        if 'meanWaveforms' in i :
            meanWaveF = d1['meanWaveforms'][0]        
           
        elif 'meanW' in i:        
            meanWaveF = d1['meanW'][0]    
    
    maxIx = d2['maxIx']
    index_neurons = [data_directory.split("/")[-1]+"_"+str(n) for n in spikes.keys()]
    to_return = {} 

    maxIx = maxIx.reshape(len(spikes.keys()))

#find the spike closest to the center 
#compute correlation between spike time and channel index
   
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
 
    
    # sys.exit()
    ###################
     
    #####################
    # down = down.astype(np.int64)
    # down_tsd = nts.Ts(t = down)
    # t = np.linspace(down[0], down[-1] + 10e6, 10)
    # bins = nts.IntervalSet(start = t[0:-1], end = t[1:])    
    # #down_latency = {i:np.zeros((len(spikes)), dtype = np.int64) for i in range(len(down))}
    # down_latency = np.zeros((len(down),len(spikes)), dtype = np.int64)
    # count = 0    
    # for j in bins.index:
    #     down_grp = down_tsd.restrict(bins.loc[[j]])      
    #     if len(down_grp)!=0:
    #         down_idx = np.arange(len(down_grp))+count
    #         for i, n in enumerate(spikes.keys()):            
    #             print(j,n)
    #             spktimes = spikes[n].restrict(bins.loc[[j]]).index.values        
    #             if len(spktimes)!=0:
    #                 difft = np.vstack(spktimes) - down_grp.index.values
    #                 maxt = np.max(difft)*2
    #                 difft[difft<=0] = maxtspikes[j].index.values[int(spkIndex[j])]
    #                 tokeep = ~np.all(difft==maxt, 0)
    #                 difft = difft[:,tokeep]
    #                 tpos = np.argmin(difft,0)
    #                 gr = pd.Series(index = down_idx[tokeep]).groupby(tpos).groups
    #                 for k in gr.keys():
    #                     #down_latency[gr[k][-1]][i] = difft[k,gr[k][-1]-count]
    #                     down_latency[gr[k][-1],i] = difft[k,gr[k][-1]-count]
                        
    #         count = count + len(down_grp)
    
    # maxt = np.max(down_latency)*2
    # down_latency[down_latency==0] = maxt
    # tokeep = ~np.all(down_latency==maxt,1)
    # down_latency = down_latency[tokeep]
    # neuron_index = np.argmin(down_latency, 1)    
    # neuron_latency = np.min(down_latency, 1)
    
    # down_latency_pyr = down_latency[:,pyr]
    # pyr_rates = rates.loc[pyr,'sws'].values.astype(np.float64)
    
    # corrr = []
    # for i in range(len(down_latency_pyr)):
    #     tmp = down_latency_pyr[i]
    #     idx = np.where(tmp != maxt)[0]
    #     if len(idx)>5:            
    #         r, p = pearsonr(tmp[idx], pyr_rates[idx])
    #         corrr.append(r)
    
    #sys.exit()

    
    
    
    
    
    
    
    # for  i in range(len(down_latency[:,pyr])):
    #     idx_d = np.where(~(down_latency[:,pyr][i] == maxt))[0]
            
    #     if len(idx_d)>=2:
    #         fcW_pyr,fpW_pyr = pearsonr(down_latency[:,pyr][i][idx_d],rates['sws'][pyr].iloc[idx_d])
    #         frcorrS_pyr.append(fcW_pyr)
    #         frpvalsS_pyr.append(fpW_pyr)
                
        
    # plot(neuron_latency/1000, maxIx[neuron_index], 'o')
    
    ###################
    spkIndex = np.zeros(len(spikes.keys()))
    corrDw = np.zeros((len(down)))
    
    
    

    for i in range(len(down)-1): 
        
        tokeep = np.zeros(len(spikes.keys()))
        difftimes1 = np.zeros(len(spikes.keys()))
        dt = np.zeros(len(spikes.keys()))
    
        
    # Is the next Down state >500ms
                       
        if down[i+1] - down[i] > 500000:
            ref = down[i]
        
            for j in spikes.keys(): 
                
            #print(j)
                               
                while spkIndex[j] < len(spikes[j]) and spikes[j].index.values[int(spkIndex[j])] < ref:
                    spkIndex[j] += 1
                    
                if spkIndex[j] == len(spikes[j]):
                        difftimes1[j] = np.nan
                    
                else: difftimes1[j] = spikes[j].index.values[int(spkIndex[j])] - ref
            
            
    # Is latency <500ms If so, label the neuron as "tokeep"
    # compute correlation only with these neurons
                dt[j] = difftimes1[j]/1000
                                
                if dt[j] != np.nan and (dt[j] > 0 and dt[j] < 500):
                    tokeep[j] = 1
                
            keep_idx = np.where(tokeep==1)
            pyr_keep = list(set.intersection(set(list(keep_idx[0])),set((pyr))))
            int_keep = list(set.intersection(set(list(keep_idx[0])),set((interneuron))))
            hd_keep = list(set.intersection(set(list(keep_idx[0])),set((hd))))
            
            if len(pyr_keep)>5:
            #     fcW_pyr,fpW_pyr = pearsonr(dt[pyr_keep],rates['wake'][pyr_keep])    
            #     frcorrW_pyr.append(fcW_pyr)
            #     frpvalsW_pyr.append(fpW_pyr)
            
                # fcS_pyr,fpS_pyr = pearsonr(dt[pyr_keep],rates['sws'][pyr_keep])    
                # frcorrS_pyr.append(fcS_pyr)
                # frpvalsS_pyr.append(fpS_pyr)
            
                # corr_pyr,p_pyr = kendalltau(dt[pyr_keep],depth[pyr_keep])    
                # nonparcorr_pyr.append(corr_pyr)
                # nonparpvals_pyr.append(p_pyr)
                
                corr_pyr,p_pyr = PartialCorr(dt[pyr_keep],depth[pyr_keep],rates['sws'][pyr_keep])    
                pcorr_pyr.append(corr_pyr)
                ppvals_pyr.append(p_pyr)
                
            
                
            # if len(int_keep)>5: 
            # #     fcW_int,fpW_int = pearsonr(dt[int_keep],rates['wake'][int_keep])    
            # #     frcorrW_int.append(fcW_int)
            # #     frpvalsW_int.append(fpW_int)
                
            #     fcS_int,fpS_int = pearsonr(dt[int_keep],rates['sws'][int_keep])    
            #     frcorrS_int.append(fcS_int)
            #     frpvalsS_int.append(fpS_int)
            
                # corr_int,p_int = kendalltau(dt[int_keep],depth[int_keep])    
                # nonparcorr_int.append(corr_int)
                # nonparpvals_int.append(p_int)
                
            # if len(hd_keep)>5: 
            # # #     fcW_hd,fpW_hd = pearsonr(dt[hd_keep],rates['wake'][hd_keep])    
            # # #     frcorrW_hd.append(fcW_hd)
            # # #     frpvalsW_hd.append(fpW_hd)
                
            # #     fcS_hd,fpS_hd = pearsonr(dt[hd_keep],rates['sws'][hd_keep])    
            # #     frcorrS_hd.append(fcS_hd)
            # #     frpvalsS_hd.append(fpS_hd)
            
            #     corr_hd,p_hd = kendalltau(dt[hd_keep],depth[hd_keep])    
            #     nonparcorr_hd.append(corr_hd)
            #     nonparpvals_hd.append(p_hd)
           
                        
 ############################################################################################### 
    # FIRING RATE AND LATENCY CORR
###############################################################################################  
               
            
            
    # plt.figure()
    # plt.title('Correlation between wake FR and latency to first spike (PYR)_' + name)
    # plt.hist(frcorrW_pyr,np.linspace(-1,1), label = 'Mean = ' + str(np.mean(frcorrW_pyr)))
    # plt.xlabel('Pearson r value')
    # plt.ylabel('Frequency')
    # plt.legend(loc = 'upper right')
    
    # plt.figure()
    # plt.title('Correlation between wake FR and latency to first spike (INT)_' + name)
    # plt.hist(frcorrW_int,np.linspace(-1,1), label = 'Mean = ' + str(np.mean(frcorrW_int)))
    # plt.xlabel('Pearson r value')
    # plt.ylabel('Frequency')
    # plt.legend(loc = 'upper right')
    
    # plt.figure()
    # plt.title('Correlation between wake FR and latency to first spike (HD)_' + name)
    # plt.hist(frcorrW_hd,np.linspace(-1,1), label = 'Mean = ' + str(np.mean(frcorrW_hd)))
    # plt.xlabel('Pearson r value')
    # plt.ylabel('Frequency')
    # plt.legend(loc = 'upper right')
    
    # plt.figure()
    # plt.title('Correlation between NREM FR and latency to first spike (PYR)_' + name)
    # plt.hist(frcorrS_pyr,np.linspace(-1,1), label = 'Mean = ' + str(np.mean(frcorrS_pyr)))
    # plt.xlabel('Pearson r value')
    # plt.ylabel('Frequency')
    # plt.legend(loc = 'upper right')
    
    # plt.figure()
    # plt.title('Correlation between NREM FR and latency to first spike (INT)_' + name)
    # plt.hist(frcorrS_int,np.linspace(-1,1), label = 'Mean = ' + str(np.mean(frcorrS_int)))
    # plt.xlabel('Pearson r value')
    # plt.ylabel('Frequency')
    # plt.legend(loc = 'upper right')
    
    # plt.figure()
    # plt.title('Correlation between NREM FR and latency to first spike (HD)_' + name)
    # plt.hist(frcorrS_hd,np.linspace(-1,1), label = 'Mean = ' + str(np.mean(frcorrS_hd)))
    # plt.xlabel('Pearson r value')
    # plt.ylabel('Frequency')
    # plt.legend(loc = 'upper right')               
    
      
############################################################################################### 
    # FIRING RATE AS A FUNCTION OF DEPTH
###############################################################################################  
  
    # corr_pyr, p_pyr = pearsonr(df_chan_lat['frS'][pyr], df_chan_lat['depth'][pyr])  
    
    # if len(interneuron) > 1:
    #   corr_int, p_int = pearsonr(df_chan_lat['frS'][interneuron],df_chan_lat['depth'][interneuron]) 
  
    # plt.figure()
    # plt.title('NREM firing rate v/s cell depth_' + name) 
    # plt.scatter(df_chan_lat['frS'][pyr],df_chan_lat['depth'][pyr],label = 'Pearson r (PYR) = ' + str(corr_pyr))
    # plt.scatter(df_chan_lat['frS'][interneuron],df_chan_lat['depth'][interneuron], label = 'Pearson r (INT) = ' + str(corr_int))
    # plt.xscale('log')
    # plt.xlim(0.001, max(df_chan_lat['frS'][interneuron]) + 5)
    # plt.ylabel('Cell depth from top channel (microns)')
    # plt.legend(loc = 'lower left')
    # plt.xlabel('NREM Firing rate (Hz)')
    
    # corr_pyr, p_pyr = pearsonr(df_chan_lat['frW'][pyr], df_chan_lat['depth'][pyr])  
    
    # if len(interneuron) > 1:
    #   corr_int, p_int = pearsonr(df_chan_lat['frW'][interneuron],df_chan_lat['depth'][interneuron])
    
    # plt.figure()
    # plt.title('Wake firing rate v/s cell depth_' + name) 
    # plt.scatter(df_chan_lat['frW'][pyr],df_chan_lat['depth'][pyr],label = 'Pearson r (PYR) = ' + str(corr_pyr))
    # plt.scatter(df_chan_lat['frW'][interneuron],df_chan_lat['depth'][interneuron], label = 'Pearson r (INT) = ' + str(corr_int))
    # plt.xscale('log')
    # plt.xlim(0.001, max(df_chan_lat['frW'][interneuron]) + 5)
    # plt.ylabel('Cell depth from top channel (microns)')
    # plt.legend(loc = 'lower left')
    # plt.xlabel('Wake Firing rate (Hz)')
    
    # corr, p = pearsonr(df_chan_lat['frS'][hd], df_chan_lat['depth'][hd])
    # plt.figure()
    # plt.title('NREM firing rate v/s cell depth_' + name) 
    # plt.scatter(df_chan_lat['frS'][hd],df_chan_lat['depth'][hd],label = 'Pearson r = ' + str(corr))
    # plt.xscale('log')
    # plt.xlim(0.001, max(df_chan_lat['frS'][hd]) + 5)
    # plt.ylabel('Cell depth from top channel (microns)')
    # plt.legend(loc = 'lower left')
    # plt.xlabel('NREM Firing rate (Hz)')
    # plt.show()
    
    # corr, p = pearsonr(df_chan_lat['frW'][hd], df_chan_lat['depth'][hd])
    # plt.figure()
    # plt.title('Wake firing rate v/s cell depth_' + name) 
    # plt.scatter(df_chan_lat['frW'][hd],df_chan_lat['depth'][hd],label = 'Pearson r = ' + str(corr))
    # plt.xscale('log')
    # plt.xlim(0.001, max(df_chan_lat['frW'][hd]) + 5)
    # plt.ylabel('Cell depth from top channel (microns)')
    # plt.legend(loc = 'lower left')
    # plt.xlabel('Wake Firing rate (Hz)')
    # plt.show()

############################################################################################### 
    # LATENCY AND DEPTH CORR
###############################################################################################  
    
    # plt.figure()
    # plt.title('Correlation between latency to first spike and cell depth (PYR)_' + name)
    # plt.hist(nonparcorr_pyr,np.linspace(-1,1), label = 'Mean = ' + str(np.mean(nonparcorr_pyr)))
    # plt.xlabel('Kendall tau value')
    # plt.ylabel('Frequency')
    # plt.legend(loc = 'upper right')          
    # plt.show()
    
    # plt.figure()
    # plt.title('Correlation between latency to first spike and cell depth (INT)_' + name)
    # plt.hist(nonparcorr_int,np.linspace(-1,1), label = 'Mean = ' + str(np.mean(nonparcorr_int)))
    # plt.xlabel('Kendall tau value')
    # plt.ylabel('Frequency')
    # plt.legend(loc = 'upper right')          
    # plt.show()
    
    # plt.figure()
    # plt.title('Correlation between latency to first spike and cell depth (HD)_' + name)
    # plt.hist(nonparcorr_hd,np.linspace(-1,1), label = 'Mean = ' + str(np.mean(nonparcorr_hd)))
    # plt.xlabel('Kendall tau value')
    # plt.ylabel('Frequency')
    # plt.legend(loc = 'upper right')          
    # plt.show()

############################################################################################### 
    # PARTIAL CORR
###############################################################################################  


    plt.figure()
    plt.title('Partial correlation between latency to first spike and cell depth (PYR)_' + name)
    plt.hist(pcorr_pyr,np.linspace(-1,1), label = 'Mean = ' + str(np.mean(pcorr_pyr)))
    plt.xlabel('Kendall tau value')
    plt.ylabel('Frequency')
    plt.legend(loc = 'upper right')          
    plt.show()
    
    # plt.figure()
    # plt.title('Correlation between latency to first spike and cell depth (INT)_' + name)
    # plt.hist(nonparcorr_int,np.linspace(-1,1), label = 'Mean = ' + str(np.mean(nonparcorr_int)))
    # plt.xlabel('Kendall tau value')
    # plt.ylabel('Frequency')
    # plt.legend(loc = 'upper right')          
    # plt.show()
    
    # plt.figure()
    # plt.title('Correlation between latency to first spike and cell depth (HD)_' + name)
    # plt.hist(nonparcorr_hd,np.linspace(-1,1), label = 'Mean = ' + str(np.mean(nonparcorr_hd)))
    # plt.xlabel('Kendall tau value')
    # plt.ylabel('Frequency')
    # plt.legend(loc = 'upper right')          
    # plt.show()
