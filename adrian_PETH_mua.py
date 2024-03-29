#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 13:33:24 2021

@author: dhruv
"""

import numpy as np 
import pandas as pd 
import scipy.io
from functions import *
from wrappers import *
import os, sys
import neuroseries as nts 
import time 
import matplotlib.pyplot as plt 
from Wavelets import MyMorlet as Morlet
from scipy.stats import pearsonr 
from scipy.stats import wilcoxon, mannwhitneyu
import seaborn as sns

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
# datasets = np.loadtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'

dur_D = []
dur_V = []
pmeans = []
diff = []
sess = []

allends = []
allstarts = []
dends = []
dstart = []
vends = []
vstart = []

meanupdur = [] 
meandowndur = []
CVup = []
CVdown = []

allupdur = [] 
alldowndur = []

updist = pd.DataFrame()
downdist = pd.DataFrame()

uplogdist = pd.DataFrame()
downlogdist = pd.DataFrame()

for s in datasets:
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    rawpath = os.path.join(rwpath,s)

    spikes, shank = loadSpikeData(path)
    n_channels, fs, shank_to_channel = loadXML(rawpath)
  
    filepath = os.path.join(path, 'Analysis')
    listdir    = os.listdir(filepath)  
    file = [f for f in listdir if 'CellDepth' in f]
    celldepth = scipy.io.loadmat(os.path.join(filepath,file[0]))
    depth = celldepth['cellDep']
    
    file = [f for f in listdir if 'CellTypes' in f]
    celltype = scipy.io.loadmat(os.path.join(filepath,file[0]))
    
     
############################################################################################### 
############################################################################################### 
    data = pd.DataFrame()   
        
       
    data['depth'] = np.reshape(depth,(len(spikes.keys())),)
    data['level'] = pd.cut(data['depth'],2, precision=0, labels=[1,0]) #0 is dorsal, 1 is ventral
    data['celltype'] = np.nan
    data['gd'] = 0
    
    for i in range(len(spikes)):
        if celltype['gd'][i] == 1:
            data.loc[i,'gd'] = 1
            
    data = data[data['gd'] == 1]
    
    #CONTROL: Use every other cell
    # data = data.iloc[::2,:]
    
    for i in range(len(spikes)):
        if celltype['ex'][i] == 1 and celltype['gd'][i] == 1:
            data.loc[i,'celltype'] = 'ex' #0 for excitatory
        elif celltype['fs'][i] == 1 and celltype['gd'][i] == 1:
            data.loc[i,'celltype'] = 'fs' #1 for inhibitory
    
    # data = data[data['celltype'] == 'ex'] #Doing it for each cell type separately 
    # data = data[data['celltype'] == 'fs']
    
    bin_size = 10000 #us    
    
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
        
        
    #%% Durations of UP and DOWN states 
              
    dep = down_ep/1e6
    uep = up_ep/1e6

    updur = (uep['end'] - uep['start']) 
    meanupdur.append(np.mean(updur))
    CVup.append(np.std(updur)/np.mean(updur))
    allupdur.append([i for i in updur.values])
    
    downdur = (dep['end'] - dep['start'])
    alldowndur.append([i for i in downdur.values])
    meandowndur.append(np.mean(downdur))
    CVdown.append(np.std(downdur)/np.mean(downdur))
       

    upbins = np.linspace(0,8,60)
    downbins = np.linspace(0,2,60)
    logbins = np.linspace(np.log10(0.02), np.log10(50), 30)

    upd, _ = np.histogram(updur, upbins)
    upd = upd/sum(upd)
    
    downd, _  = np.histogram(downdur, downbins)
    downd = downd/sum(downd)
    
    uplogd,_ = np.histogram(np.log10(updur), logbins)
    uplogd = uplogd/sum(uplogd)
    
    downlogd,_ = np.histogram(np.log10(downdur), logbins)
    downlogd = downlogd/sum(downlogd)
    
    
    updist = pd.concat([updist, pd.Series(upd)], axis = 1)
    downdist = pd.concat([downdist, pd.Series(downd)], axis = 1)
            
    uplogdist = pd.concat([uplogdist, pd.Series(uplogd)], axis = 1)
    downlogdist = pd.concat([downlogdist, pd.Series(downlogd)], axis = 1)
    
#%% 


########################################################################################################   
#FIND MUA THRESHOLD CROSSING FOR DORSAL AND VENTRAL        
########################################################################################################  
    
    mua = {}
    
    latency_dorsal = []
    latency_ventral = []
    
    # define mua for dorsal and ventral
    for i in range(2):
        mua[i] = []        
        for n in data[data['level'] == i].index:            
            mua[i].append(spikes[n].index.values)
        mua[i] = nts.Ts(t = np.sort(np.hstack(mua[i])))
   
############################################################################################### 
    # COMPUTE EVENT CROSS CORRS
###############################################################################################  
                 
    binsize = 5
    nbins = 1000        
    neurons = list(mua.keys())
    times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2        
    cc = pd.DataFrame(index = times, columns = neurons)
    tsd_dn = down_ep.as_units('ms').start.values
    
    # ep_D = nts.IntervalSet(start = down_ep.start[0], end = down_ep.end.values[-1])
    rates = []
    
    ddur = []
    vdur = []
    
    for i in neurons:
        spk2 = mua[i].restrict(up_ep).as_units('ms').index.values
        tmp = crossCorr(tsd_dn, spk2, binsize, nbins)
        tmp = pd.DataFrame(tmp)
        tmp = tmp.rolling(window=8, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
        
        fr = len(spk2)/up_ep.tot_length('s')
        rates.append(fr)
        cc[i] = tmp.values
        cc[i] = tmp.values/fr
       
        dd = cc[-250:250]
           
        tmp = dd[i].loc[5:] > 0.2 #np.percentile(dd[i].values,20)
        ends = tmp.where(tmp == True).first_valid_index()
        allends.append(ends)
        
        tmp2 = dd[i].loc[-150:-5] > 0.2 #np.percentile(dd[i].values,20)  
        start = tmp2.where(tmp2 == True).last_valid_index()
        allstarts.append(start)
        
        if i == 0: 
            ddur.append(ends - start)
            dends.append(ends)
            dstart.append(start)
        else: 
            vdur.append(ends - start)
            vends.append(ends)
            vstart.append(start)
               
    dur_D.append(ddur[0])
    dur_V.append(vdur[0])
          

        
        
#%%         
        
diff = np.zeros(len(dur_D))

for i in range(len(diff)):
    diff[i] = dur_V[i] - dur_D[i]
    
# rge = np.linspace(min(diff),max(diff),10)
# plt.figure()
# plt.title('Mean (Ventral - Dorsal) duration (ms)')
# plt.xlabel('Difference (ms)')
# plt.ylabel('Number of sessions')
# plt.axvline(np.mean(diff), color = 'k')
# plt.hist(diff,rge, label = 'Mean = ' +  str(round(np.mean(diff),4)))
# plt.legend()

t,p = wilcoxon(dur_D,dur_V)

#%%


plt.scatter(dur_D, dur_V, color = 'k', zorder = 3) 
plt.gca().axline((min(min(dur_D),min(dur_V)),min(min(dur_D),min(dur_V)) ), slope=1, color = 'silver', linestyle = '--')
plt.xlabel('Dorsal DOWN duration (ms)')
plt.ylabel('Ventral DOWN duration (ms)')
plt.axis('square')



#%%

# label = ['Dorsal', 'Ventral']
# # x1 = np.random.normal(0, 0.01, size=len(dur_D))
# # x2 = np.random.normal(0.3, 0.01, size=len(dur_D))
# # x = np.vstack([x1, x2])# the label locations
# x = [0, 0.35]# the label locations
# width = 0.3  # the width of the bars

# plt.figure()
# plt.boxplot(dur_D, positions=[0], showfliers=False, patch_artist=True, boxprops=dict(facecolor='mediumorchid', color='mediumorchid'),
#             capprops=dict(color='mediumorchid'),
#             whiskerprops=dict(color='mediumorchid'),
#             medianprops=dict(color='white', linewidth = 2))
# plt.boxplot(dur_V, positions=[0.3], showfliers=False, patch_artist=True, boxprops=dict(facecolor='violet', color='violet'),
#             capprops=dict(color='violet'),
#             whiskerprops=dict(color='violet'),
#             medianprops=dict(color='white', linewidth = 2))

# plt.xticks([0, 0.3],['Dorsal', 'Ventral'])
# plt.title('Mean session DOWN-state duration')
# plt.ylabel('DOWN duration (ms)')
# pval = np.vstack([(dur_D), (dur_V)])
# plt.plot(x, np.vstack(pval), 'o-', color = 'k', zorder = 3, markersize = 3, linewidth = 1 )


# means_D = np.nanmean(dur_D)
# means_V = np.nanmean(dur_V)

# label = ['dorsal', 'ventral']
# x = [0, 0.35]# the label locations
# width = 0.35  # the width of the bars


# fig, ax = plt.subplots()
# rects1 = ax.bar(x[0], means_D, width, color = 'royalblue')
# rects1 = ax.bar(x[1], means_V, width, color = 'lightsteelblue')
# plt.plot(x, np.vstack(pval), 'o-', color = 'k', zorder = 3, markersize = 3, linewidth = 1 )

#%% Durations of UP and DOWN states 

upbincenter = 0.5 * (upbins[1:] + upbins[:-1])
downbincenter = 0.5 * (downbins[1:] + downbins[:-1])
logbincenter = 0.5 * (logbins[1:] + logbins[:-1])

uperr = updist.std(axis=1)
downerr = downdist.std(axis=1)
uplogerr = uplogdist.std(axis=1)
downlogerr = downlogdist.std(axis=1)

plt.figure()
plt.xlabel('Duration (s)')
plt.ylabel('P (duration)')
plt.plot(upbincenter, updist.mean(axis = 1), color = 'r', label = 'UP')
plt.fill_between(upbincenter, updist.mean(axis = 1) - uperr, updist.mean(axis = 1) + uperr, color = 'r', alpha = 0.2)
plt.plot(downbincenter, downdist.mean(axis = 1), color = 'b', label = 'DOWN')
plt.fill_between(downbincenter, downdist.mean(axis = 1) - downerr, downdist.mean(axis = 1) + downerr, color = 'b', alpha = 0.2)
plt.legend(loc = 'upper right')

plt.figure()
plt.ylabel('P (duration)')
plt.plot(logbincenter, uplogdist.mean(axis = 1), color = 'r', label = 'UP')
plt.fill_between(logbincenter, uplogdist.mean(axis = 1) - uplogerr, uplogdist.mean(axis = 1) + uplogerr, color = 'r', alpha = 0.2)
plt.plot(logbincenter, downlogdist.mean(axis = 1), color = 'b', label = 'DOWN')
plt.fill_between(logbincenter, downlogdist.mean(axis = 1) - downlogerr, downlogdist.mean(axis = 1) + downlogerr, color = 'b', alpha = 0.2)
plt.legend(loc = 'upper right')

#%% 

updurs = [item for sublist in allupdur for item in sublist]
downdurs = [item for sublist in alldowndur for item in sublist]
