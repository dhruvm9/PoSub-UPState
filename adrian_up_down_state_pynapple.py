#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 10:24:53 2022

@author: dhruv
"""
#loading the dataset
import numpy as np 
import pandas as pd 
import scipy.io
from functions import *
from wrappers import *
import os, sys
import pynapple as nap
import time 
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import wilcoxon


data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_Ver_DM.list'), delimiter = '\n', dtype = str, comments = '#')
rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'


all_down_dur = []


for s in datasets:
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    rawpath = os.path.join(rwpath,s)
    
############################################################################################### 
    # LOADING DATA
###############################################################################################
    data = nap.load_session(rawpath, 'neurosuite')
    data.load_neurosuite_xml(rawpath)
    channelorder = data.group_to_channel[0]
    spikes = data.spikes
    epochs = data.epochs
 

#fishing out wake and sleep epochs
    sleep_ep = epochs['sleep']
    wake_ep = epochs['wake']
    acceleration = loadAuxiliary(rawpath, 1, fs = 20000) 
        
    file = os.path.join(path, 'Sleep', name + '.sws.evt')
    sws_ep = data.read_neuroscope_intervals(name = 'SWS', path2file = file)
   
        
############################################################################################### 
    # REFINE WAKE 
###############################################################################################      

    vl = acceleration[0].restrict(wake_ep)
    vl = vl.as_series().diff().abs().dropna() 
    
    a, _ = scipy.signal.find_peaks(vl, 0.025)
    peaks = nap.Tsd(vl.iloc[a])
    duration = np.diff(peaks.as_units('s').index.values)
    interval = nap.IntervalSet(start = peaks.index.values[0:-1], end = peaks.index.values[1:])
    
    rest_ep = interval.iloc[duration>15.0]
    rest_ep = rest_ep.reset_index(drop=True)
    rest_ep = rest_ep.merge_close_intervals(100000, time_units ='us')

    new_wake_ep = wake_ep.set_diff(rest_ep)
    data.write_neuroscope_intervals('.new_wake.evt' , new_wake_ep, 'Wake')
   

############################################################################################### 
    # DETECTION OF UP AND DOWN STATES
############################################################################################### 

    bin_size = 0.01 #s
    smoothing_window = 0.02

    rates = spikes.count(bin_size, sws_ep)
       
    total2 = rates.as_dataframe().rolling(window = 100 ,win_type='gaussian',
                                          center=True,min_periods=1, 
                                          axis = 0).mean(std= int(smoothing_window/bin_size))
    
    total2 = total2.sum(axis =1)
    total2 = nap.Tsd(total2)
    idx = total2.threshold(np.percentile(total2.values,20),'below')
            
    down_ep = idx.time_support
    
    down_ep = nap.IntervalSet(start = down_ep['start'], end = down_ep['end'])
    down_ep = down_ep.drop_short_intervals(bin_size)
    down_ep = down_ep.merge_close_intervals(bin_size*2)
    down_ep = down_ep.drop_short_intervals(bin_size*3)
    down_ep = down_ep.drop_long_intervals(bin_size*50)
   
    # sys.exit() 
   
    up_ep = nap.IntervalSet(down_ep['end'][0:-1], down_ep['start'][1:])
    down_ep = sws_ep.intersect(down_ep)
    
    up_ep = sws_ep.intersect(up_ep)
    
    
    down_dur = down_ep.tot_length('s') / sws_ep.tot_length('s') 
    all_down_dur.append(down_dur)
    
    
############################################################################################### 
    # WRITING FOR NEUROSCOPE
############################################################################################### 
    
    data.write_neuroscope_intervals(extension = '.evt.py.dow', isets = down_ep, name = 'DOWN') 
    data.write_neuroscope_intervals(extension = '.evt.py.upp', isets = up_ep, name = 'UP') 
    

