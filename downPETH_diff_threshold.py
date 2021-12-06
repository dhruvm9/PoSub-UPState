#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 11:13:59 2021

@author: dhruv
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 13:54:03 2021

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
from scipy.stats import kendalltau, pearsonr, wilcoxon

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
# datasets = np.loadtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
datasets = np.loadtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'

allcoefs_dn_ex = []
allcoefs_dn_fs = []
durcoeffs_ex = []
durcoeffs_fs = []
allspeeds_ex = []
allspeeds_fs = []

n_pyr = []
n_int = []

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

    n_pyr.append(len(pyr))
    n_int.append(len(interneuron))
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
    tsd_dn = down_ep.as_units('ms').start.values
  
#DOWN State    
    ep_D = nts.IntervalSet(start = down_ep.start[0], end = down_ep.end.values[-1])
               
    rates = []
    reslist = []
            
    for i in neurons:
        spk2 = spikes[i].restrict(ep_D).as_units('ms').index.values
        tmp = crossCorr(tsd_dn, spk2, binsize, nbins)
        tmp = pd.DataFrame(tmp)
        tmp = tmp.rolling(window=4, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
        
        fr = len(spk2)/ep_D.tot_length('s')
        rates.append(fr)
        cc[i] = tmp.values
        cc[i] = tmp.values/fr
       
        dd = cc[-250:250]
        
    #Cell types 
    ee = dd[pyr]
    ff = dd[interneuron]
   
#####EX cells    
    if len(ee.columns) > 0: 
         
        mn = np.mean(ee[-250:-100])
        threshold = np.zeros(len(ee.columns))  
         
        for i in ee.columns:
            threshold[i] = mn[i] / 2
            tmp2_ex = ee[i].loc[-250:-5] > threshold[i]    
            
            if tmp2_ex.sum(0) > 0:
                start_ex = np.array([tmp2_ex.index[np.where(tmp2_ex[i])[0][-1]] for i in tokeep2_ex])        
     
        
        
        
        
        
    
        
        
        st_ex = pd.Series(index = tokeep2_ex, data = start_ex)
            
        ix_ex = np.intersect1d(tokeep_ex,tokeep2_ex)
        ix_ex = [int(i) for i in ix_ex]
        
        
        depths_keeping_ex = depth[ix_ex]
        dur_ex = np.zeros(len(ix_ex))
         
        
        for i,n in enumerate(ix_ex):
                dur_ex[i] = es_ex[ix_ex][n] - st_ex[ix_ex][n]
     
    #Plot for threshold crossing           
    coef_ex, p_ex = kendalltau(st_ex[ix_ex],depths_keeping_ex)
    allcoefs_dn_ex.append(coef_ex)
    
    stk_ex = st_ex[ix_ex]
    
        
#####FS cells    
    if len(ff.columns) > 0: 
    
        mn2 = np.mean(ff[-250:-100])
        # tmp = dd.loc[5:] > 0.2
        tmp_fs = ff.loc[5:] > np.mean(mn2/2)
        
        tokeep_fs = tmp_fs.columns[tmp_fs.sum(0) > 0]
        ends_fs = np.array([tmp_fs.index[np.where(tmp_fs[i])[0][0]] for i in tokeep_fs])
        es_fs = pd.Series(index = tokeep_fs, data = ends_fs)
        
        # tmp2 = dd.loc[-250:-5] > 0.2
        tmp2_fs = ff.loc[-250:-5] > np.mean(mn2/2)
        
        tokeep2_fs = tmp2_fs.columns[tmp2_fs.sum(0) > 0]
        start_fs = np.array([tmp2_fs.index[np.where(tmp2_fs[i])[0][-1]] for i in tokeep2_fs])
        st_fs = pd.Series(index = tokeep2_fs, data = start_fs)
            
        ix_fs = np.intersect1d(tokeep_fs,tokeep2_fs)
        ix_fs = [int(i) for i in ix_fs]
        
        
        depths_keeping_fs = depth[ix_fs]
        dur_fs = np.zeros(len(ix_fs))
         
        
        for i,n in enumerate(ix_fs):
                dur_fs[i] = es_fs[ix_fs][n] - st_fs[ix_fs][n]
     
    #Plot for threshold crossing           
    coef_fs, p_fs = kendalltau(st_fs[ix_fs],depths_keeping_fs)
    allcoefs_dn_fs.append(coef_fs)
    
    stk_fs = st_fs[ix_fs]
 
####SPEED COMPUTATION 

    y_est_ex = np.zeros(len(stk_ex))
    m_ex, b_ex = np.polyfit(stk_ex, depths_keeping_ex, 1)
    allspeeds_ex.append((m_ex[0])/10)
        
    for i in range(len(stk_ex)):
        y_est_ex[i] = m_ex*stk_ex.values[i]
    

    y_est_fs = np.zeros(len(stk_fs))
    m_fs, b_fs = np.polyfit(stk_fs, depths_keeping_fs, 1)
    allspeeds_fs.append((m_fs[0])/10)
        
    for i in range(len(stk_fs)):
        y_est_fs[i] = m_fs*stk_fs.values[i]


####PLOT        
    plt.figure()
    plt.scatter(stk_ex, depths_keeping_ex, color = 'blue', label = 'R(ex) = ' + str(round(coef_ex,4)))
    plt.scatter(stk_fs, depths_keeping_fs, color = 'red', label = 'R(FS) = ' + str(round(coef_fs,4)))
    # plt.plot(stk_ex, y_est_ex + b_ex, color = 'blue')
    # plt.plot(stk_fs, y_est_fs + b_fs, color = 'orange')
    plt.title('Last bin before DOWN where FR > 50% baseline rate_' + s)
    plt.ylabel('Depth from top of probe (um)')
    plt.yticks([0, -400, -800])
    plt.xlabel('Lag (ms)')
    plt.legend(loc = 'upper right')
            
#####Plot for duration
    # coef2_ex, p2_ex = kendalltau(dur_ex,depths_keeping_ex)
    # coef2_fs, p2_fs = kendalltau(dur_fs,depths_keeping_fs)
            
    # durcoeffs_ex.append(coef2_ex)
    # durcoeffs_fs.append(coef2_fs)
            
    # plt.figure()
    # plt.scatter(dur_ex,depths_keeping_ex, label = 'R(ex) = ' + str(round(coef2_ex,4)))
    # plt.scatter(dur_fs,depths_keeping_fs, label = 'R(FS) = ' + str(round(coef2_fs,4)))
    # plt.title('Duration of < 50% baseline firing v/s depth_' + s)
    # plt.ylabel('Depth from top of probe (um)')
    # plt.yticks([0, -400, -800])
    # plt.xlabel('Duration of cell firing < 50% (ms)')
    # plt.legend(loc = 'upper right')
          
    # sys.exit()
    
#Out of loop 

z_dn_ex, p_dn_ex = wilcoxon(np.array(allcoefs_dn_ex)-0)
z_dn_fs, p_dn_fs = wilcoxon(np.array(allcoefs_dn_fs)-0)
# bins = np.linspace(min(min(allcoefs_dn_ex), min(allcoefs_dn_fs)),max(max(allcoefs_dn_ex), max(allcoefs_dn_fs)),20)
plt.figure()
plt.hist(allcoefs_dn_ex, color = 'blue', label = 'p-value =' + str(round(p_dn_ex,4)))
plt.axvline(np.mean(allcoefs_dn_ex),color = 'k')
# plt.hist(allcoefs_dn_fs,bins, alpha = 0.5, label = 'p-value (FS) = ' + str(round(p_dn_fs,4)))
plt.title('Distribution of Kendall Tau for DOWN-state onset (ex cells)')
plt.xlabel('Kendall tau value') 
plt.ylabel('Number of sessions')

plt.legend(loc = 'upper right')  


# z_dn_ex, p_dn_ex = wilcoxon(np.array(durcoeffs_ex)-0)
# z_dn_fs, p_dn_fs = wilcoxon(np.array(durcoeffs_fs)-0)
# # bins = np.linspace(min(min(durcoeffs_ex), min(durcoeffs_fs)),max(max(durcoeffs_ex), max(durcoeffs_fs)),20)

# plt.figure()
# plt.hist(durcoeffs_ex, label = 'p-value =' + str(round(p_dn_ex,4)))
# plt.axvline(np.mean(durcoeffs_ex),color = 'k')
# # plt.hist(durcoeffs_fs, bins, alpha = 0.5, label = 'p-value (FS) =' + str(round(p_dn_fs,4)))
# plt.title('Distribution of Kendall Tau for DOWN-duration (ex cells)')
# plt.xlabel('Kendall tau value') 
# plt.ylabel('Number of sessions')
# plt.legend(loc = 'upper right')  


# z_spd_ex, p_spd_ex = wilcoxon(np.array(allspeeds_ex)-0)
# z_spd_fs, p_spd_fs = wilcoxon(np.array(allspeeds_fs)-0)
# bins = np.linspace(min(min(allspeeds_ex), min(allspeeds_fs)),max(max(allspeeds_ex), max(allspeeds_fs)),20)
# plt.figure()
# plt.hist(allspeeds_ex,bins, alpha = 0.5, label = 'Mean = ' + str(round(np.mean(allspeeds_ex),4)))
# plt.hist(allspeeds_fs,bins, alpha = 0.5, label = 'Mean = ' + str(round(np.mean(allspeeds_fs),4)))
# plt.title('Speed of DOWN propagation')
# plt.xlabel('Speed (cm/s)') 
# plt.ylabel('Number of sessions')
# plt.legend(loc = 'upper right')          


#Raster and LFP code 

# fig, ax = plt.subplots()
# [plot(spikes[n].restrict(per).as_units('s').fillna(n), '|', color = 'k') for n in spikes.keys()]
# # plot(lfp.restrict(per).as_units('s'), color = 'k')
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.axes.get_yaxis().set_visible(False)

