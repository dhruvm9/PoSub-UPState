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
from scipy.stats import kendalltau, pearsonr, wilcoxon, mannwhitneyu

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
# datasets = np.loadtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
datasets = np.loadtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'

allmeans_ex = []
allmeans_fs = []

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
    tsd_up = up_ep.as_units('ms').start.values
   
#UP State    
    sub_ep_U = np.array_split(up_ep,2)             
    
    rates = []
    rates2 = []
    
    k = []
      
    for i in neurons:
        spk2 = spikes[i].restrict(sub_ep_U[0]).as_units('ms').index.values
        spk3 = spikes[i].restrict(sub_ep_U[1]).as_units('ms').index.values
        
        tmp = crossCorr(tsd_up, spk2, binsize, nbins)
        tmp2 = crossCorr(tsd_up, spk3, binsize, nbins)
        
        fr = len(spk2)/sub_ep_U[0].tot_length('s')
        fr2 = len(spk3)/sub_ep_U[1].tot_length('s')
        
        if fr == 0 or fr2 == 0:
            k.append(i)
                                    
        rates.append(fr)
        rates2.append(fr2)
    
        cc[i] = tmp
        cc[i] = tmp/fr
        
       
        cc2[i] = tmp2
        cc2[i] = tmp2/fr2        
        

        dd = cc[-50:250]
        dd2 = cc2[-50:250]
    
      
    # if s == 'A3703-191215':
    #     sys.exit()
    
    #Cell types 
    ee = dd[pyr]
    ee2 = dd2[pyr]
    
    ff = dd[interneuron]
    ff2 = dd2[interneuron]
    
    n = len(depth)
    # t2 = np.argsort(depth.flatten())
    # res = np.in1d(t2,k)
    # b = np.where(res == True)
    # t2 = np.delete(t2,b[0])
    
    res_ex = np.in1d(pyr,k)
    b_ex = np.where(res_ex == True)
    pyr = np.delete(pyr,b_ex[0])
    t2_ex = np.argsort(depth[pyr].flatten())
    
    res_fs = np.in1d(interneuron,k)
    b_fs = np.where(res_fs == True)
    interneuron = np.delete(interneuron,b_fs[0])
    t2_fs = np.argsort(depth[interneuron].flatten())
       
     
      
    # desc = t2[::-1][:n]
    desc_ex = t2_ex[::-1][:n]
    desc_fs = t2_fs[::-1][:n]
    
    order_ex = []
    order_fs = []
    
    for i in range(len(pyr)): 
        order_ex.append(pyr[desc_ex[i]])
    
    for i in range(len(interneuron)): 
        order_fs.append(interneuron[desc_fs[i]])
    
        
    finalRates_ex = ee[order_ex]
    finalRates2_ex = ee2[order_ex]
    
    finalRates_fs = ff[order_fs]
    finalRates2_fs = ff2[order_fs]
    
    
    # finalRates = dd[desc]
    # finalRates2 = dd2[desc]

    R_ex = np.corrcoef(finalRates_ex.T,finalRates2_ex.T)
    R_fs = np.corrcoef(finalRates_fs.T,finalRates2_fs.T)
        
    plt.figure()
    plt.title('Half session correlation_'+ s)
    plt.xlabel('Pearson R')
    plt.ylabel('Number of cells')
    
    bins = np.linspace(0,1,20)
    
    plt.hist(np.diagonal(R_ex[0:len(pyr),len(pyr):]),bins, alpha = 0.5, label = 'Mean R(ex) = ' + str(round(np.mean(np.diagonal(R_ex[0:len(pyr),len(pyr):])),4)))
    plt.hist(np.diagonal(R_fs[0:len(interneuron),len(interneuron):]), bins, alpha = 0.5, label = 'Mean R(FS) = ' + str(round(np.mean(np.diagonal(R_fs[0:len(interneuron),len(interneuron):])),4)))
    # plt.hist(np.diagonal(R[0:n,n:]),label = 'Mean R = ' + str(round(np.nanmean(np.diagonal(R[0:n,n:])),4)))
    
    plt.legend(loc = 'upper right')
    
    allmeans_ex.append(np.mean(np.diagonal(R_ex[0:len(pyr),len(pyr):])))
    allmeans_fs.append(np.mean(np.diagonal(R_fs[0:len(interneuron),len(interneuron):])))
    # allmeans.append(np.mean(np.diagonal(R[0:n,n:])))
    
    

    # if len(ee.columns) > 5:
    # if len(dd.columns) > 5:
        
        # fig, (ax1,ax2) = plt.subplots(nrows = 1, ncols = 2)
        # fig.suptitle('UP PETH_' + s)

        # cax = ax1.imshow(finalRates.T,extent=[-50 , 250, len(interneuron) , 1],vmin = 0, vmax = 3, aspect = 'auto', cmap = 'hot')
        # cax = ax2.imshow(finalRates2.T,extent=[-50 , 250, len(interneuron) , 1],vmin = 0, vmax = 3, aspect = 'auto', cmap = 'hot')
        
        # cax = ax1.imshow(finalRates.T,extent=[-50 , 250, len(neurons) , 1],vmin = 0, vmax = 3, aspect = 'auto', cmap = 'hot')
        # cax = ax2.imshow(finalRates2.T,extent=[-50 , 250, len(neurons) , 1],vmin = 0, vmax = 3, aspect = 'auto', cmap = 'hot')
        
        # # cax = ax1.imshow(finalRates.T,extent=[-50 , 250, len(pyr) , 1],vmin = 0, vmax = 3, aspect = 'auto', cmap = 'hot')
        # # cax = ax2.imshow(finalRates2.T,extent=[-50 , 250, len(pyr) , 1],vmin = 0, vmax = 3, aspect = 'auto', cmap = 'hot')
        
        
        
        # cbar = fig.colorbar(cax, ticks=[0, 1, 2, 3], label = 'Normalized FR')
        # cbar.ax.set_yticklabels(['0', '1', '2', '3'])
        # cax.set_clim([0, 3])
        # ax1.set_title('First half')
        # ax2.set_title('Second half')
        # ax1.set_ylabel('Neuron number')
        # ax1.set_xlabel('Lag (ms)')
        # ax2.set_xlabel('Lag (ms)')

z_stat_ex, p_val_ex = wilcoxon(np.array(allmeans_ex)-0)
z_stat_fs, p_val_fs = wilcoxon(np.array(allmeans_fs)-0)
t, pvalue = mannwhitneyu(allmeans_ex, allmeans_fs)
bins = np.linspace(0,1,20)
plt.figure()
plt.hist(allmeans_ex,bins, alpha = 0.5, label = 'p-value (ex) =' + str(round(p_val_ex,4)))
plt.hist(allmeans_fs,bins, alpha = 0.5, label = 'p-value (FS) =' + str(round(p_val_fs,4)))
plt.title('Distribution of Pearson R within session')
plt.xlabel('Pearson R value') 
plt.ylabel('Number of sessions')
plt.legend(loc = 'upper left')    
     