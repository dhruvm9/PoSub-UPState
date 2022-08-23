#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 16:38:04 2022

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

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
# datasets = np.loadtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
datasets = np.loadtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'

for s in datasets:
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    rawpath = os.path.join(rwpath,s)
   
    data = nap.load_session(rawpath, 'neurosuite')
    data.load_neurosuite_xml(rawpath)
    
    data = nap.load_session(rawpath, 'neurosuite')
    data.load_neurosuite_xml(rawpath)
    
    spikes = data.spikes  
    epochs = data.epochs
    
    filepath = os.path.join(path, 'Analysis')
    data = pd.read_csv(filepath + '/Tracking_data.csv', header = None)
    position = pd.DataFrame(index = data[0].values, data = data[[1,2,3]].values, columns=['x', 'y', 'ang'])
    position = position.loc[~position.index.duplicated(keep='first')]
    position['ang'] = position['ang'] *(np.pi/180) #convert degrees to radian
    position['ang'] = (position['ang'] + 2*np.pi) % (2*np.pi) #convert [-pi, pi] to [0, 2pi]
    position = nap.TsdFrame(position) 
    position = position.restrict(epochs['wake'])
    
#%% 
    numHDbins = 12
    bin_dt = 0.23  
    holdout = 0.2 
    
    decoder = linearDecoder(len(spikes),numHDbins)
    
    decoder = decoder.load(s + '_HDbins_' + str(numHDbins) + '_dt_' + str(bin_dt),rwpath + '/trained_decoders/' ) 

#%%

    rates = spikes.count(bin_dt)
    HD = rates.value_from(position.ang)
    rates = rates.restrict(position.time_support)    
    HDbinedges = np.linspace(0,2*np.pi,numHDbins+1)
    centre_bins = 0.5 * (HDbinedges[0:-1] + HDbinedges[1:])
    HD_binned = np.digitize(HD.values,HDbinedges)-1 #(-1 for 0-indexed category)
    HD_binned = nap.Tsd(d=HD_binned, t=HD.index.values)
    test_rates = rates.head(np.int32((holdout)*len(rates)))
    test_HD = HD_binned.head(np.int32((holdout)*len(rates)))

#%%
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
    (errorcounts,errorbins,mrlbins) = np.histogram2d(decode_error,MRL,bins=[20,10],
                                                     range=[[0,np.pi/2],[0,1]])
    P_error_MRL = errorcounts/np.sum(errorcounts,axis=0)
    
    plt.figure()
    plt.title(s)
    plt.imshow(P_error_MRL, origin='lower', extent = [mrlbins[0],mrlbins[-1],
                                                      errorbins[0],errorbins[-1]],
               aspect='auto')
    plt.xlabel('MRL')
    plt.ylabel('Error (rad)')