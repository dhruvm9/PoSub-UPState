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
import pingouin as pg 
import matplotlib.pyplot as plt

#Load the data (from NWB)

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'

s = 'A3707-200317'
path = os.path.join(data_directory, s)
rawpath = os.path.join(rwpath,s)

data = nap.load_session(rawpath, 'neurosuite')

spikes = data.spikes
epochs = data.epochs

file = os.path.join(rawpath, s +'.DM.new_sws.evt')
new_sws_ep  = data.read_neuroscope_intervals(name = 'new_sws', path2file = file)

#%%

#Convert spikes to rates

dt = [0.025,0.03]
HDbins = [10,12]

errs = pd.DataFrame(index = dt,columns = HDbins)

for i in range(len(dt)):

    bin_dt = dt[i] #200ms bins
    rates = spikes.count(bin_dt)
        `
    #New Tsd with HD signal at same timepoints as rates
    HD = rates.value_from(position.ang)
    
    #Restrict Rates to only times we have HD info
    rates = rates.restrict(position.time_support) 

#%%
#Bin HD and put it into a Tsd
        
    for j in range(len(HDbins)):
        
        numHDbins = HDbins[j]
        HDbinedges = np.linspace(0,2*np.pi,numHDbins+1)
        centre_bins = 0.5 * (HDbinedges[0:-1] + HDbinedges[1:])
        
        HD_binned = np.digitize(HD.values,HDbinedges)-1 #(-1 for 0-indexed category)
        HD_binned = nap.Tsd(d=HD_binned, t=HD.index.values)
    
    #Pynapple Question: why doesn't this work? Do we want it to?
    #HD['binned'] = np.digitize(HD.values,HDbinedges)
    
    #%%
    #Separate Train and Test data
        holdout = 0.2 #percentage of data to hold out for test set
        
        # train_rates = rates.head(np.int32((1-holdout)*len(rates)))
        # train_HD = HD_binned.head(np.int32((1-holdout)*len(rates)))
        
        # test_rates = rates.tail(np.int32((holdout)*len(rates)))
        # test_HD = HD_binned.tail(np.int32((holdout)*len(rates)))
        
        train_rates = rates.tail(np.int32((1-holdout)*len(rates)))
        train_HD = HD_binned.tail(np.int32((1-holdout)*len(rates)))
        
        test_rates = rates.head(np.int32((holdout)*len(rates)))
        test_HD = HD_binned.head(np.int32((holdout)*len(rates)))


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
        N_units = len(spikes)
        decoder = linearDecoder(N_units,numHDbins)
        
        #Train the decoder 
        batchSize = 0.75
        numBatches = 5000
        decoder.train(train_rates.values, train_HD.values, 
                      batchSize=batchSize, numBatches=numBatches,
                      Znorm=False)
#%%
#Decode HD from test set
        decoded, p = decoder.decode(test_rates.values, withSoftmax=True)
        # decoder.save('HDbins_' + str(numHDbins) + '_dt_' + str(bin_dt), rwpath + 'decoder_test/')
        decoder.save('HDbins_' + str(numHDbins) + '_dt_' + str(bin_dt), rwpath + 'param_search/')
      
        
#Calculate decoding error

        wtavg = np.zeros(len(p))
        MRL = np.zeros(len(p))
        
        for i in range(len(p)):
            wtavg[i] = pg.circ_mean(centre_bins, w = p[i,:])
            MRL[i] = pg.circ_r(centre_bins, w = p[i,:])
    
        wtavg = np.mod(wtavg, 2*np.pi)
                
        # lin_error = np.abs(centre_bins[decoded] - HD[test_HD.index].values)
        lin_error = np.abs(wtavg - HD[test_HD.index].values)
        
        decode_error = np.minimum((2*np.pi - abs(lin_error)), abs(lin_error))

        errs.loc[bin_dt][numHDbins] = np.mean(decode_error)
#%%

for i in errs:
    errs[i] = np.array(errs[i]).astype(float)
    
#%%
fig, ax = plt.subplots(1,1)
img = ax.imshow(errs.T,extent=[min(errs.index),max(errs.index),max(errs.columns),min(errs.columns)],aspect = 'auto', origin = 'lower')

ax.set_yticks(errs.columns.tolist(), labels=errs.columns.tolist())
ax.set_xticks(errs.index.tolist(), labels=errs.index.tolist())
fig.colorbar(img,label = 'Decode Error (rad)')
ax.set_ylabel('Number of HD bins')
ax.set_xlabel('Time bins (s)')
#%%
a, b = errs.stack().idxmin()
print('Best decoder has time bin of ' +  str(a) + 's, and ' + str(b) + ' HD bins')

# decoder = decoder.load('HDbins_' + str(b) + '_dt_' + str(a),rwpath + 'decoder_test/' )
decoder = decoder.load('HDbins_' + str(b) + '_dt_' + str(a),rwpath + 'param_search/' )

bin_dt = a 
rates = spikes.count(bin_dt)
HD = rates.value_from(position.ang)
rates = rates.restrict(position.time_support) 

numHDbins = b
HDbinedges = np.linspace(0,2*np.pi,numHDbins+1)
centre_bins = 0.5 * (HDbinedges[0:-1] + HDbinedges[1:])
HD_binned = np.digitize(HD.values,HDbinedges)-1 #(-1 for 0-indexed category)
HD_binned = nap.Tsd(d=HD_binned, t=HD.index.values)
test_rates = rates.head(np.int32((holdout)*len(rates)))
test_HD = HD_binned.head(np.int32((holdout)*len(rates)))

decoded, p = decoder.decode(test_rates.values, withSoftmax=True)

wtavg = np.zeros(len(p))
MRL = np.zeros(len(p))

for i in range(len(p)):
    wtavg[i] = pg.circ_mean(centre_bins, w = p[i,:])
    MRL[i] = pg.circ_r(centre_bins, w = p[i,:])

wtavg = np.mod(wtavg, 2*np.pi)
        
lin_error = np.abs(wtavg - HD[test_HD.index].values)
decode_error = np.minimum((2*np.pi - abs(lin_error)), abs(lin_error))

#%%
import matplotlib.pyplot as plt
plt.figure()
plt.subplot(2,2,4)
plt.hist(decode_error)
plt.xlabel('Error (rad)')
plt.subplot(2,1,1)
plt.imshow(p.T, aspect='auto',interpolation='none', extent = [test_HD.time_support['start'].values[0],
                                                              test_HD.time_support['end'].values[0],
                                                              HDbinedges[0],HDbinedges[-1]],
           origin='lower')
plt.plot(test_HD.index.values, HD[test_HD.index].values,'r')
plt.plot(test_HD.index.values, wtavg)
plt.ylabel('Angle (rad)')
#plt.xlim(0,100)
#plt.xlim(1000,1500)
plt.show()


#%%

(errorcounts,errorbins,mrlbins) = np.histogram2d(decode_error,MRL,bins=[20,10],
                                                 range=[[0,np.pi/2],[0.5,1]])
P_error_MRL = errorcounts/np.sum(errorcounts,axis=0)

plt.figure()
plt.imshow(P_error_MRL, origin='lower', extent = [mrlbins[0],mrlbins[-1],
                                                  errorbins[0],errorbins[-1]],
           aspect='auto')
plt.xlabel('MRL')
plt.ylabel('Error (rad)')

#%%
a = pd.Series(data = [0.01,0.025,0.03,0.046,0.05,0.076])

overlap = 0.005
bins_lo = np.arange(0,a.values[-1],overlap)
bins_hi = np.arange(overlap, a.values[-1]+overlap, overlap)

for i in range(len(bins_lo)):
    b[i] = [bins_lo[i], bins_hi[i]]

binwidth = 0.01
bins = np.arange(0, a.values[-1],binwidth)
overlap = 0.005

for i in range(len(bins)):
    b[i] = [bins[i],bins[i]+overlap]
