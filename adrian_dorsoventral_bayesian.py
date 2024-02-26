#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 16:28:04 2024

@author: dhruv
"""

import numpy as np 
import pandas as pd 
import scipy.io
import pynapple as nap 
import nwbmatic as ntm
import os, sys
import time 
import matplotlib.pyplot as plt 
from scipy.stats import kendalltau, pearsonr, wilcoxon, mannwhitneyu
import matplotlib.cm as cm
import seaborn as sns
import pycircstat

#%% 

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'

allerr = []

angprop = []

allh = []
allmu = []
allci = []

for s in datasets:
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    rawpath = os.path.join(rwpath,s)

    data = ntm.load_session(rawpath, 'neurosuite')
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
        
#%% Load angles 
    
    filepath = os.path.join(path, 'Analysis')
    data = pd.read_csv(filepath + '/Tracking_data.csv', header = None)
    position = pd.DataFrame(index = data[0].values, data = data[[1,2,3]].values, columns=['x', 'y', 'ang'])
    position = position.loc[~position.index.duplicated(keep='first')]
    position['ang'] = position['ang'] *(np.pi/180) #convert degrees to radian
    position['ang'] = (position['ang'] + 2*np.pi) % (2*np.pi) #convert [-pi, pi] to [0, 2pi]
    position = nap.TsdFrame(position) 
    position = position.restrict(nap.IntervalSet(epochs['wake']))    

#%% Split into dorsal and ventral population

    spkdata = pd.DataFrame()   
    spkdata['depth'] = np.reshape(depth,(len(spikes.keys())),)
    spkdata['level'] = pd.cut(spkdata['depth'],2, precision=0, labels=[1,0]) #0 is dorsal, 1 is ventral
    spkdata['gd'] = 0
    
    for i in range(len(spikes)):
        if celltype['gd'][i] == 1:
            spkdata.loc[i,'gd'] = 1
            
    spkdata = spkdata[spkdata['gd'] == 1]

    dorsal_spikes = spikes[spkdata.index[spkdata['level'] == 0]]
    ventral_spikes = spikes[spkdata.index[spkdata['level'] == 1]]      

    dorsal_hd = np.intersect1d(hd, dorsal_spikes.index)
    ventral_hd = np.intersect1d(hd, ventral_spikes.index)    
    
#%% Compute HD tuning curves 

    tcurves_D = nap.compute_1d_tuning_curves(dorsal_spikes[dorsal_hd], feature = position['ang'], nb_bins = 61)
    tcurves_V = nap.compute_1d_tuning_curves(ventral_spikes[ventral_hd], feature = position['ang'], nb_bins = 61)
    
#%% UP state epochs where at least 5 HD cells from each population are active
    
    tmp1 = dorsal_spikes[dorsal_hd].count(0.025, up_ep).as_dataframe().sum(axis=1)
    tmp2 = ventral_spikes[ventral_hd].count(0.025, up_ep).as_dataframe().sum(axis=1)
    
    tmp1 = tmp1[tmp1 >= 5]
    tmp2 = tmp1[tmp2 >= 5]
    
    active_up = np.intersect1d(tmp1.index.values, tmp2.index.values)
       
    
#%% Decode during sleep

    d_D, p_feature_D = nap.decode_1d(tuning_curves = tcurves_D,  group = dorsal_spikes[dorsal_hd], ep = up_ep, bin_size = 0.025, 
                                           feature = position['ang'])
    
    d_V, p_feature_V = nap.decode_1d(tuning_curves = tcurves_V,  group = ventral_spikes[ventral_hd], ep = up_ep, bin_size = 0.025, 
                                           feature = position['ang'])
    
    decoded_D = nap.Ts(active_up).value_from(d_D)
    decoded_V = nap.Ts(active_up).value_from(d_V)

#%% Compute histogram of angular differences 

    # lin_error = np.abs(decoded_D.values - decoded_V.values)
    lin_error = decoded_D.values - decoded_V.values
    allerr.extend(lin_error)
    
    decode_error =  (lin_error + np.pi) % (2 * np.pi) - np.pi
    
    bins = np.linspace(-np.pi, np.pi, 61)    
    # bins = np.linspace(0, 2*np.pi, 61)    
    
    relcounts_all,_ = np.histogram(decode_error, bins)     
    # relcounts_all,_ = np.histogram(lin_error, bins)     
    p_rel = relcounts_all/sum(relcounts_all)
    angprop.append(p_rel)
    
    h, mu, ci = pycircstat.mtest(decode_error,0)
    # h, mu, ci = pycircstat.mtest(lin_error,0)
    allh.append(h)
    allmu.append(mu)
    allci.append(ci)
    # print(h, mu, ci)
    
#%% Plotting for each session

    # plt.figure()
    # plt.title(s)
    # plt.stairs(p_rel, bins, linewidth = 2, color = 'k')
    # plt.xlabel('DV decoded ang diff (rad)')
    # plt.ylabel('% events')
    # plt.gca().set_box_aspect(1)

#%% Pooled plot 
    
# rosybrown = [colors.to_rgba('darkred'), colors.to_rgba('rosybrown')] 
# rb_map = colors.LinearSegmentedColormap.(name = 'rb', colors = rosybrown, N = len(angprop))
    
plt.figure()
plt.xlabel('Ang diff between dorsal and ventral populations (rad)')
plt.ylabel('% events')
plt.gca().set_box_aspect(1)

stack = np.array([0]*(len(bins)-1))
histnorm =  np.array([angprop[i] for i in range(len(angprop))]).sum(axis = 0).sum()

for i in range(len(angprop)):
   
    plt.stairs((angprop[i] + stack) / histnorm, bins, fill = True, baseline = stack/histnorm, linewidth =  2, color = cm.copper(i/16))
    stack = stack + angprop[i]
    
#%% Plot histogram of mean values 

tmp = [(i + np.pi) % (2 * np.pi) - np.pi for i in allmu]

hpool, mupool, cipool = pycircstat.mtest(np.array(tmp),0)

plt.figure()
plt.gca().set_box_aspect(1)
plt.title('Distribution of mean DV ang diff')

plt.boxplot(tmp, positions=[0], showfliers=False, patch_artist=True, boxprops=dict(facecolor = 'orange', color = 'orange'),
              capprops=dict(color='orange'),
              whiskerprops=dict(color='orange'),
              medianprops=dict(color='white', linewidth = 2))

x1 = np.random.normal(0, 0.01, size=len(tmp))

plt.plot(x1, tmp, '.', color = 'k', fillstyle = 'none', markersize = 6, zorder =3)
plt.xticks([])
plt.ylabel('Ang diff. (rad)')
    
      
 #%% Example
 
# ex_ep = nap.IntervalSet(start = 2107.7 , end = 2108.5)

# plt.subplot(311)
# plt.plot(decoded_D.restrict(ex_ep), 'o-')
# plt.plot(decoded_V.restrict(ex_ep), 'o-') 
# plt.subplot(312)
# for i in dorsal_spikes[dorsal_hd]:
#       plt.plot(dorsal_spikes[dorsal_hd][i].restrict(ex_ep).fillna(i), '|', color = 'royalblue')
# plt.subplot(313)
# for i in ventral_spikes[ventral_hd]:
#      plt.plot(ventral_spikes[ventral_hd][i].restrict(ex_ep).fillna(i), '|', color = 'royalblue')
 