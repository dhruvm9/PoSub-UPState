#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 12:16:50 2022

@author: dhruv
"""

from LinearDecoder import linearDecoder
import numpy as np
import pynapple as nap
import os,sys
import pandas as pd
import scipy.io
import pingouin as pg 
import matplotlib.pyplot as plt

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
wake_dt = 0.23
numHDbins = 12
N_units = len(spikes)

HDbinedges = np.linspace(0,2*np.pi,numHDbins+1)
centre_bins = 0.5 * (HDbinedges[0:-1] + HDbinedges[1:])

sleep_activity = spikes.count(sleep_dt, new_sws_ep)
sleep_activity = sleep_activity.as_dataframe().rolling(5, min_periods = 1, center = True, axis = 0).sum() #25 ms bins for sleep

#%%
#Decode HD from test set
decoder = linearDecoder(N_units,numHDbins)

decoder = decoder.load('HDbins_' + str(numHDbins) + '_dt_' + str(wake_dt),rwpath + 'param_search/' )
decoded, p = decoder.decode(sleep_activity.values, withSoftmax=True)

        # decoder.save('HDbins_' + str(numHDbins) + '_dt_' + str(bin_dt), rwpath + 'decoder_test/')
decoder.save(s + '_sleep_HDbins_' + str(numHDbins) + '_dt_' + str(5*sleep_dt), rwpath + 'sleep_decoding/')
       
#Calculate decoding error

wtavg = np.zeros(len(p))
MRL = np.zeros(len(p))

for i in range(len(p)):
    wtavg[i] = pg.circ_mean(centre_bins, w = p[i,:])
    MRL[i] = pg.circ_r(centre_bins, w = p[i,:])

wtavg = np.mod(wtavg, 2*np.pi)

poprate = sleep_activity.sum(axis=1)
poprate = poprate/max(poprate)

#%%

p_x = pd.DataFrame(index = sleep_activity.index.values, data = p)
MRL = nap.Tsd(t = sleep_activity.index.values, d = MRL)
wtavg = nap.Tsd(t = sleep_activity.index.values, d = wtavg)
poprate = nap.Tsd(t = sleep_activity.index.values, d = poprate)

#%%
peri_du = {}

for i in range(len(new_sws_ep)):
    mrlvec = MRL.restrict(new_sws_ep.loc[[i]])
    du = nap.Ts(up_ep['start'].values).restrict(new_sws_ep.loc[[i]])
       
    MRL_PETH = nap.compute_perievent(mrlvec, du , minmax = (-0.3, 0.3), time_unit = 's')
    tmp = []

    for j in range(len(MRL_PETH)):
        if len(MRL_PETH[j]) >= 120:
            tmp.append(MRL_PETH[j].as_series())
        
    tmp = pd.concat(tmp, axis = 1, join = 'inner')
    tmp = tmp.mean(axis = 1)  
    peri_du[i] = pd.Series(data = tmp, name = i)    

du_peth_all = pd.DataFrame(index = peri_du[0].index.values)

for i in range(len(new_sws_ep)):
    du_peth_all = pd.concat([du_peth_all, peri_du[i]], axis = 1)

plt.figure()
plt.imshow(du_peth_all.T, aspect='auto', extent = [du_peth_all.index[0],du_peth_all.index[-1],du_peth_all.columns[0],du_peth_all.columns[-1]], origin='lower')
plt.colorbar()

#%%
peri_ud = {}

for i in range(len(new_sws_ep)):
    mrlvec = MRL.restrict(new_sws_ep.loc[[i]])
    ud = nap.Ts(down_ep['start'].values).restrict(new_sws_ep.loc[[i]])
       
    MRL_PETH = nap.compute_perievent(mrlvec, ud , minmax = (-0.3, 0.3), time_unit = 's')
    tmp = []

    for j in range(len(MRL_PETH)):
        if len(MRL_PETH[j]) >= 120:
            tmp.append(MRL_PETH[j].as_series())
        
    tmp = pd.concat(tmp, axis = 1, join = 'inner')
    tmp = tmp.mean(axis = 1)  
    peri_ud[i] = pd.Series(data = tmp, name = i)    

ud_peth_all = pd.DataFrame(index = peri_du[0].index.values)

for i in range(len(new_sws_ep)):
    ud_peth_all = pd.concat([ud_peth_all, peri_ud[i]], axis = 1)

plt.figure()
plt.imshow(ud_peth_all.T, aspect='auto', extent = [ud_peth_all.index[0],ud_peth_all.index[-1],ud_peth_all.columns[0],du_peth_all.columns[-1]], origin='lower')
plt.colorbar()


