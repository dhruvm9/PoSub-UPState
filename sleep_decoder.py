#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 17:11:01 2022

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
sleep_binwidth = 0.025 #25ms binwidth
wake_dt = 0.23

numHDbins = 12
N_units = len(spikes)

HDbinedges = np.linspace(0,2*np.pi,numHDbins+1)
centre_bins = 0.5 * (HDbinedges[0:-1] + HDbinedges[1:])

num_overlapping_bins = int(sleep_binwidth/sleep_dt)

sleep_activity = spikes.count(sleep_dt, new_sws_ep)
sleep_activity = sleep_activity.as_dataframe().rolling(num_overlapping_bins, min_periods = 1, center = True, axis = 0).sum() #25 ms bins for sleep
sleep_rates = sleep_activity/sleep_binwidth

wake_activity = spikes.count(wake_dt, position.time_support)
#sleep_activity = sleep_activity.as_dataframe().rolling(num_overlapping_bins, min_periods = 1, center = True, axis = 0).sum() #25 ms bins for sleep
wake_rates = wake_activity/wake_dt
    
#%%
        
#Plot data
# plt.figure()

# plt.subplot(2,1,1)
# #Pynapple Question: do we want this to pull timestamps, like plot?
# plt.imshow(rates.T, aspect='auto')  
# plt.ylabel('Cell')

# plt.subplot(2,1,2)
# plt.plot(train_HD)
# plt.plot(test_HD)

# plt.xlabel('t (s)')
# plt.ylabel('HD (bin)')

# plt.show()

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

start = new_sws_ep['start'].values[0]
ends = new_sws_ep['end'].values[0]

p_x = pd.DataFrame(index = sleep_rates.index.values, data = p)
MRL = nap.Tsd(t = sleep_rates.index.values, d = MRL)
wtavg = nap.Tsd(t = sleep_rates.index.values, d = wtavg)
poprate = nap.Tsd(t = sleep_rates.index.values, d = poprate)

MRL_thresholded = MRL.threshold(0.5)
wtavg_toShow = nap.Tsd(t = sleep_rates.index.values, d = wtavg[MRL_thresholded.index.values])

p_x = p_x[start:ends]

upons = up_ep['start'].values[(up_ep['start'].values >= start) & (up_ep['start'].values <= ends)]
upoffs = up_ep['end'].values[(up_ep['end'].values >= start) & (up_ep['end'].values <= ends)]

plt.figure()
plt.imshow(p_x.T, aspect='auto',interpolation='none', extent = [start,ends,HDbinedges[0],HDbinedges[-1]], origin='lower')
plt.plot(wtavg_toShow[start:ends],'r')
plt.plot(MRL[start:ends],'w')
plt.plot(poprate[start:ends], 'g')

for pos in range(len(upons)):
    plt.axvline(upons[pos], color='m')
    plt.axvline(upoffs[pos], color='c')
    
plt.colorbar()



#%%

(ratecounts,mrlbins,ratebins) = np.histogram2d(MRL,poprate,bins=[30,30],
                                                 range=[[0,1],[-1,0.5]])

#conditional Distribution 

# P_rate_MRL = ratecounts/np.sum(ratecounts,axis=0)

P_MRL_rate = ratecounts/np.sum(ratecounts,axis=0)
plt.figure()
plt.imshow(P_MRL_rate, origin='lower', extent = [ratebins[0],ratebins[-1],mrlbins[0],mrlbins[-1]],
                                               aspect='auto',vmax=0.2)
plt.ylabel('MRL')
plt.xlabel('log( norm Population rate)')


#Joint Distribution

plt.figure()
plt.imshow(ratecounts, origin='lower', extent = [ratebins[0],ratebins[-1],
                                                  mrlbins[0],mrlbins[-1]],
            aspect='auto')
plt.ylabel('MRL')
plt.xlabel('log (norm Population rate)')


#%%
