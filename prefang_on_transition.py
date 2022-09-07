#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 10:06:25 2022

@author: dhruv
"""

from LinearDecoder import linearDecoder
import numpy as np
import pynapple as nap
import os,sys
import pandas as pd
import scipy.io
import scipy.stats as stats
import pingouin as pg 
import matplotlib.pyplot as plt

#Load the data (from NWB)

#%% On lab PC

# data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
# rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'
# s = 'A3707-200317'
# path = os.path.join(data_directory, s)
# rawpath = os.path.join(rwpath,s)
# data = nap.load_session(rawpath, 'neurosuite')
# file = os.path.join(rawpath, s +'.DM.new_sws.evt')

#%% On Nibelungen 

data_directory = '/mnt/DataNibelungen/Dhruv/A3707-200317'
rwpath = '/mnt/DataNibelungen/Dhruv/'
data = nap.load_session(data_directory, 'neurosuite')
s = 'A3707-200317'

spikes = data.spikes
epochs = data.epochs
file = os.path.join(data_directory, s +'.DM.new_sws.evt')

new_sws_ep  = data.read_neuroscope_intervals(name = 'new_sws', path2file = file)

file = os.path.join(data_directory, s +'.evt.py.dow')
if os.path.exists(file):
    tmp = np.genfromtxt(file)[:,0]
    tmp = tmp.reshape(len(tmp)//2,2)/1000
    down_ep = nap.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')

file = os.path.join(data_directory, s +'.evt.py.upp')
if os.path.exists(file):
    tmp = np.genfromtxt(file)[:,0]
    tmp = tmp.reshape(len(tmp)//2,2)/1000
    up_ep = nap.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')

filepath = os.path.join(data_directory, 'Analysis')
data = pd.read_csv(filepath + '/Tracking_data.csv', header = None)
position = pd.DataFrame(index = data[0].values, data = data[[1,2,3]].values, columns=['x', 'y', 'ang'])
position = position.loc[~position.index.duplicated(keep='first')]
position['ang'] = position['ang'] *(np.pi/180) #convert degrees to radian
position['ang'] = (position['ang'] + 2*np.pi) % (2*np.pi) #convert [-pi, pi] to [0, 2pi]
position = nap.TsdFrame(position) 
position = position.restrict(epochs['wake'])

#%%  
#Convert spikes to rates

sleep_dt = 0.005 #5ms overlapping bins 
sleep_binwidth = np.arange(0.025,0.1,0.01) #25ms binwidth
wake_dt = 0.23
numHDbins = 12
N_units = len(spikes)

HDbinedges = np.linspace(0,2*np.pi,numHDbins+1)
centre_bins = 0.5 * (HDbinedges[0:-1] + HDbinedges[1:])

mean_ud_mrl = []
mean_du_mrl = []

num_overlapping_bins_list = []

for j in sleep_binwidth: 
    
    num_overlapping_bins = int(j/sleep_dt)    
    num_overlapping_bins_list.append(num_overlapping_bins)
    
    sleep_activity = spikes.count(sleep_dt, new_sws_ep)
    sleep_activity = sleep_activity.as_dataframe().rolling(num_overlapping_bins, min_periods = 1, center = True, axis = 0).sum() #25 ms bins for sleep
    
    sleep_rates = sleep_activity/j
    

#%%
#Decode HD from test set
    decoder = linearDecoder(N_units,numHDbins)
    
    decoder = decoder.load('HDbins_' + str(numHDbins) + '_dt_' + str(wake_dt),rwpath + 'param_search/' )
    decoded, p = decoder.decode(sleep_rates.values, withSoftmax=True)
    
            # decoder.save('HDbins_' + str(numHDbins) + '_dt_' + str(bin_dt), rwpath + 'decoder_test/')
    decoder.save(s + '_sleep_HDbins_' + str(numHDbins) + '_dt_' + str(num_overlapping_bins*sleep_dt), rwpath + 'sleep_decoding/')
           
    #Calculate decoding error
    
    wtavg = np.zeros(len(p))
    MRL = np.zeros(len(p))
    
    for i in range(len(p)):
        wtavg[i] = pg.circ_mean(centre_bins, w = p[i,:])
        MRL[i] = pg.circ_r(centre_bins, w = p[i,:])
    
    wtavg = np.mod(wtavg, 2*np.pi)
    
    poprate = sleep_rates.sum(axis=1)
    poprate = poprate/poprate.median()
    tmp = np.log10(poprate.values)
    poprate = nap.Tsd(t = poprate.index.values, d = tmp)
            
#%%

    p_x = pd.DataFrame(index = sleep_rates.index.values, data = p)
    MRL = nap.Tsd(t = sleep_rates.index.values, d = MRL)
    wtavg = nap.Tsd(t = sleep_rates.index.values, d = wtavg)
    poprate = nap.Tsd(t = sleep_rates.index.values, d = poprate)

#%% 

    winlength = sleep_dt * num_overlapping_bins
    
        
    ud = down_ep['start'].values - (winlength/2)
    # ud = down_ep['start'].values - (i/2)
    ud = nap.Tsd(ud)
    
    du = down_ep['end'].values + (winlength/2)
    # du = down_ep['end'].values + (i/2)
    du = nap.Tsd(du)
    
    angle_du = du.value_from(wtavg)
    mrl_du = du.value_from(MRL)
    
    angle_ud = ud.value_from(wtavg)
    mrl_ud = ud.value_from(MRL)
    
    angdiff = abs(angle_du.values - angle_ud.values)
    angdiff = np.minimum((2*np.pi - abs(angdiff)), abs(angdiff))
    
    mean_ud_mrl.append(np.mean(mrl_ud))
    mean_du_mrl.append(np.mean(mrl_du))
    
plt.figure()
plt.plot(sleep_binwidth,mean_ud_mrl,'o-', label = 'UD')
plt.plot(sleep_binwidth,mean_du_mrl, 'o-', label = 'DU')
plt.xlabel('Window length (s)')
plt.ylabel('Mean MRL')
plt.legend(loc = 'lower right')
    
    #%%
    # f, ax = plt.subplots()
    # ax.set_title('MRL relationship :' + str(i))
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # im = ax.scatter(mrl_ud,mrl_du, c = angdiff, cmap = 'hot',s = 1)
    # f.colorbar(im, ax=ax)
    # ax.set_xlabel('UD MRL')
    # ax.set_ylabel('DU MRL')
    # ax.set_xlim(1e-3,1)
    # ax.set_ylim(1e-3,1)
    
    # plt.figure()
    # plt.title('Angular diff :' + str(i))
    # plt.hist(angdiff)
    
    

threshold = 0.5
tokeep = []

for i in range(len(mrl_ud)):
    if (mrl_ud.iloc[i] > threshold) & (mrl_du.iloc[i] > threshold):
        tokeep.append(i)        

ang_keep = angdiff[tokeep]

f, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')
im = ax.scatter(mrl_ud.values[tokeep],mrl_du.values[tokeep], c = ang_keep, cmap = 'hot',s = 1)
f.colorbar(im, ax=ax)
ax.set_xlabel('UD MRL')
ax.set_ylabel('DU MRL')
# ax.set_xlim(1e-1,1)
# ax.set_ylim(1e-1,1)   

plt.figure()
plt.hist(ang_keep)

#%% 

a = stats.binned_statistic_2d(mrl_ud.values[tokeep], mrl_du.values[tokeep], ang_keep, statistic = 'mean', bins = 20)

plt.figure()
plt.imshow(a[0].T,origin='lower', extent = [a[1][0],a[1][-1],a[2][0],a[2][-1]],
                                               aspect='auto')
plt.xlabel('UD MRL')
plt.ylabel('DU MRL')
plt.colorbar()
    
