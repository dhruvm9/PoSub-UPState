#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 15:25:50 2021

@author: dhruv
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 11:17:37 2021

@author: dhruv
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 10:12:29 2021

@author: dhruv
"""
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
# datasets = np.loadtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
datasets = np.loadtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'

allmeans = []

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
    cc2 = pd.DataFrame(index = times, columns = neurons)
    tsd_dn = down_ep.as_units('ms').start.values
    
    sub_tsd_dn = np.array_split(tsd_dn,2)
   
#UP State    
       
    rates = []
    rates2 = []
    
    k = []
      
    cc = compute_EventCrossCorr(spikes, nts.Ts(sub_tsd_dn[0], time_units = 'ms'), new_sws_ep, norm=True)
    cc2 = compute_EventCrossCorr(spikes, nts.Ts(sub_tsd_dn[1], time_units = 'ms'), new_sws_ep, norm=True)
    
    dd = cc[-250:250]
    dd2 = cc2[-250:250]
    
      
    # if s == 'A3703-191215':
    #     sys.exit()
    
    #Cell types 
    # ee = dd[pyr]
    # ee2 = dd2[pyr]
    
    # ee = dd[interneuron]
    # ee2 = dd2[interneuron]
    
    n = len(depth)
    t2 = np.argsort(depth.flatten())
    
    # res = np.in1d(pyr,k)
    # b = np.where(res == True)
    # pyr = np.delete(pyr,b[0])
    # t2 = np.argsort(depth[pyr].flatten())
    
    # res = np.in1d(interneuron,k)
    # b = np.where(res == True)
    # pyr = np.delete(interneuron,b[0])
    # t2 = np.argsort(depth[interneuron].flatten())
       
     
      
    desc = t2[::-1][:n]
    
    # order = []
    # for i in range(len(pyr)): 
    #     order.append(pyr[desc[i]])
    
    # for i in range(len(interneuron)): 
    #     order.append(interneuron[desc[i]])
    
        
    # finalRates = ee[order]
    # finalRates2 = ee2[order]
    
    finalRates = dd[desc]
    finalRates2 = dd2[desc]

    R = np.corrcoef(finalRates.T,finalRates2.T)    

    plt.figure()
    plt.title('Half session correlation_'+ s)
    plt.xlabel('Pearson R')
    plt.ylabel('Number of cells')
    
    # plt.hist(np.diagonal(R[0:len(pyr),len(pyr):]), label = 'Mean R = ' + str(round(np.nanmean(np.diagonal(R[0:len(pyr),len(pyr):])),4)))
    # plt.hist(np.diagonal(R[0:len(interneuron),len(interneuron):]), label = 'Mean R = ' + str(round(np.nanmean(np.diagonal(R[0:len(interneuron),len(interneuron):])),4)))
    plt.hist(np.diagonal(R[0:n,n:]),label = 'Mean R = ' + str(round(np.nanmean(np.diagonal(R[0:n,n:])),4)))
    
    plt.legend(loc = 'upper right')
    
    # allmeans.append(np.nanmean(np.diagonal(R[0:len(pyr),len(pyr):])))
    # allmeans.append(np.nanmean(np.diagonal(R[0:len(interneuron),len(interneuron):])))
    allmeans.append(np.nanmean(np.diagonal(R[0:n,n:])))
    
    

    # if len(ee.columns) > 5:
    # if len(dd.columns) > 5:
        
    #     fig, (ax1,ax2) = plt.subplots(nrows = 1, ncols = 2)
    #     fig.suptitle('DOWN PETH_' + s)

    #     cax = ax1.imshow(finalRates.T,extent=[-250 , 250, len(interneuron) , 1],vmin = 0, vmax = 3, aspect = 'auto', cmap = 'hot')
    #     cax = ax2.imshow(finalRates2.T,extent=[-250 , 250, len(interneuron) , 1],vmin = 0, vmax = 3, aspect = 'auto', cmap = 'hot')
        
        # cax = ax1.imshow(finalRates.T,extent=[-250 , 250, len(neurons) , 1],vmin = 0, vmax = 3, aspect = 'auto', cmap = 'hot')
        # cax = ax2.imshow(finalRates2.T,extent=[-250 , 250, len(neurons) , 1],vmin = 0, vmax = 3, aspect = 'auto', cmap = 'hot')
        
        # cax = ax1.imshow(finalRates.T,extent=[-250 , 250, len(pyr) , 1],vmin = 0, vmax = 3, aspect = 'auto', cmap = 'hot')
        # cax = ax2.imshow(finalRates2.T,extent=[-250 , 250, len(pyr) , 1],vmin = 0, vmax = 3, aspect = 'auto', cmap = 'hot')
        
        
        
        # cbar = fig.colorbar(cax, ticks=[0, 1, 2, 3], label = 'Normalized FR')
        # cbar.ax.set_yticklabels(['0', '1', '2', '3'])
        # cax.set_clim([0, 3])
        # ax1.set_title('First half')
        # ax2.set_title('Second half')
        # ax1.set_ylabel('Neuron number')
        # ax1.set_xlabel('Lag (ms)')
        # ax2.set_xlabel('Lag (ms)')

z_stat, p_val = wilcoxon(np.array(allmeans)-0)
plt.figure()
plt.hist(allmeans,label = 'p-value =' + str(round(p_val,4)))
plt.axvline(np.mean(allmeans), color = 'k')
plt.title('Distribution of Pearson R within session')
plt.xlabel('Pearson R value') 
plt.ylabel('Number of sessions')
plt.legend(loc = 'upper right')    
     