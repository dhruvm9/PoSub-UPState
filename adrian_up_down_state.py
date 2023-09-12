# -*- coding: utf-8 -*-
"""
Created on Thu May 21 12:27:26 2020

@author: Dhruv
"""
#loading the dataset
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
import seaborn as sns
from scipy.stats import wilcoxon

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')
#datasets = np.loadtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'

all_down_dur = []
all_down_wake_dur = []

for s in datasets:
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    rawpath = os.path.join(rwpath,s)
    
############################################################################################### 
    # LOADING DATA
###############################################################################################

    spikes, shank = loadSpikeData(path)
    n_channels, fs, shank_to_channel = loadXML(rawpath)

#fishing out wake and sleep epochs
    filepath = os.path.join(path, 'Analysis')
    listdir    = os.listdir(filepath)
    file = [f for f in listdir if 'BehavEpochs' in f]
    behepochs = scipy.io.loadmat(os.path.join(filepath,file[0]))

    if s == 'A3701-191119':
        sleep1_ep = np.hstack([behepochs['sleepPreEp'][0][0][1],behepochs['sleepPreEp'][0][0][2]])
        sleep1_ep = nts.IntervalSet(sleep1_ep[:,0], sleep1_ep[:,1], time_units = 's').drop_short_intervals(0.0)

        sleep2_ep = np.hstack([behepochs['sleepPostEp'][0][0][1],behepochs['sleepPostEp'][0][0][2]])
        sleep2_ep = nts.IntervalSet(sleep2_ep[:,0], sleep2_ep[:,1], time_units = 's').drop_short_intervals(0.0)
        
        wake_ep = np.hstack([behepochs['wakeEp'][0][0][1],behepochs['wakeEp'][0][0][2]])
        wake_ep = nts.IntervalSet(wake_ep[:,0], wake_ep[:,1], time_units = 's').drop_short_intervals(0.0)
        
            
    else: 
        wake_ep = np.hstack([behepochs['wake1Ep'][0][0][1],behepochs['wake1Ep'][0][0][2]])
        wake_ep = nts.IntervalSet(wake_ep[:,0], wake_ep[:,1], time_units = 's').drop_short_intervals(0.0)
        
        sleep1_ep = np.hstack([behepochs['sleep1Ep'][0][0][1],behepochs['sleep1Ep'][0][0][2]])
        
        #check if it is not empty, then go to next step
        if sleep1_ep.size != 0:
            print('sleep1 exists')
            sleep1_ep = nts.IntervalSet(sleep1_ep[:,0], sleep1_ep[:,1], time_units = 's').drop_short_intervals(0.0)
                

        sleep2_ep = np.hstack([behepochs['sleep2Ep'][0][0][1],behepochs['sleep2Ep'][0][0][2]])
        
        #check if it is not empty, then go to next step
        if sleep2_ep.size != 0: 
            print('sleep2 exists')
            sleep2_ep = nts.IntervalSet(sleep2_ep[:,0], sleep2_ep[:,1], time_units = 's').drop_short_intervals(0.0)
        
        
        
        #if both sleep1 and sleep2 are not empty, merge them. Else make the non-empty epoch sleep_ep
    if (sleep1_ep.size !=0 and sleep2_ep.size !=0): 
        
        if (sleep1_ep.start.values[0] > sleep2_ep.start.values[0]):
            sleep1_ep, sleep2_ep = sleep2_ep, sleep1_ep
    
        sleep_ep = pd.concat((sleep1_ep, sleep2_ep)).reset_index(drop=True)
            
          
    elif sleep1_ep.size != 0:
        sleep_ep = sleep1_ep
            
    else: 
        sleep_ep = sleep2_ep       
        
    file = os.path.join(rawpath, name + '.lfp')
    if os.path.exists(file):    
        lfp = loadLFP(os.path.join(rawpath, name + '.lfp'), n_channels, 1, 1250, 'int16')
    else: 
        lfp = loadLFP(os.path.join(rawpath, name + '.eeg'), n_channels, 1, 1250, 'int16')
    
    downsample = 5 
    lfp = lfp[::downsample]
    # lfp = downsample(lfp, 1, 5)
    
    acceleration = loadAuxiliary(rawpath, 1, fs = 20000) 
    newsleep_ep = refineSleepFromAccel(acceleration, sleep_ep)
    
    
    file = os.path.join(rawpath, name +'.sws.evt')
    if os.path.exists(file):
        tmp = np.genfromtxt(file)[:,0]
        tmp = tmp.reshape(len(tmp)//2,2)/1000
        sws_ep1 = nts.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
        
    new_sws_ep = newsleep_ep.intersect(sws_ep1)

    
############################################################################################### 
    # THETA/DELTA RATIO FOR SWS
###############################################################################################   
    
    # new_sws_start = []
    # new_sws_end = []
    # ratio_sws = []
    # ratio_wake = [] 

    # for i in sws_ep.index:  
    #     sws_lfp = lfp.restrict(sws_ep.loc[[i]])
    
    #     lfp_filt_theta = nts.Tsd(sws_lfp.index.values, butter_bandpass_filter(sws_lfp, 4, 12, 1250/5, 2))
    #     power_theta = nts.Tsd(lfp_filt_theta.index.values, np.abs(lfp_filt_theta.values))
    #     power_theta = power_theta.rolling(window=1000,win_type='gaussian',center=True,min_periods=1).mean(std=80)

    #     lfp_filt_delta = nts.Tsd(sws_lfp.index.values, butter_bandpass_filter(sws_lfp, 0.5, 4, 1250/5, 2))
    #     power_delta = nts.Tsd(lfp_filt_delta.index.values, np.abs(lfp_filt_delta.values))
    #     power_delta = power_delta.rolling(window=1000,win_type='gaussian',center=True,min_periods=1).mean(std=80)
    
    
    #     ratio = nts.Tsd(t = power_theta.index.values, d = np.log(power_theta.values/power_delta.values))
    #     ratio2 = ratio.rolling(window=10000,win_type='gaussian',center=True,min_periods=1).mean(std=200)
    #     ratio2 = ratio2.mean()

    #     if ratio2 < 0:
    #         new_sws_start.append(sws_ep.iloc[i].start)
    #         new_sws_end.append(sws_ep.iloc[i].end)
    #         ratio_sws.append(ratio2)
    
    # new_sws_ep = nts.IntervalSet(start = new_sws_start, end = new_sws_end)
    # mean_ratio_sws = np.mean(ratio_sws)
    
############################################################################################### 
    # REFINE WAKE 
###############################################################################################      

    # vl = acceleration[0].restrict(wake_ep)
    # vl = vl.as_series().diff().abs().dropna() 
    
    # a, _ = scipy.signal.find_peaks(vl, 0.025)
    # peaks = nts.Tsd(vl.iloc[a])
    # duration = np.diff(peaks.as_units('s').index.values)
    # interval = nts.IntervalSet(start = peaks.index.values[0:-1], end = peaks.index.values[1:])

    # rest_ep = interval.iloc[duration>15.0]
    # rest_ep = rest_ep.reset_index(drop=True)
    # rest_ep = rest_ep.merge_close_intervals(100000, time_units ='us')

    # new_wake_ep = wake_ep.set_diff(rest_ep)
    
############################################################################################### 
    # THETA/DELTA RATIO FOR WAKE
###############################################################################################  
    # for i in new_wake_ep.index: 

    #     wake_lfp = lfp.restrict(new_wake_ep.loc[[i]])

    #     lfp_filt_theta = nts.Tsd(wake_lfp.index.values, butter_bandpass_filter(wake_lfp, 4, 12, 1250/5, 2))
    #     power_theta = nts.Tsd(lfp_filt_theta.index.values, np.abs(lfp_filt_theta.values))
    #     power_theta = power_theta.rolling(window=1000,win_type='gaussian',center=True,min_periods=1).mean(std=80)
    
    #     lfp_filt_delta = nts.Tsd(wake_lfp.index.values, butter_bandpass_filter(wake_lfp, 0.5, 4, 1250/5, 2))
    #     power_delta = nts.Tsd(lfp_filt_delta.index.values, np.abs(lfp_filt_delta.values))
    #     power_delta = power_delta.rolling(window=1000,win_type='gaussian',center=True,min_periods=1).mean(std=80)
    
    #     ratio = nts.Tsd(t = power_theta.index.values, d = np.log(power_theta.values/power_delta.values))
    #     ratio2 = ratio.rolling(window=10000,win_type='gaussian',center=True,min_periods=1).mean(std=200)
    #     ratio2 = ratio2.mean()

    #     ratio_wake.append(ratio2) 
        
    # mean_ratio_wake = np.mean(ratio_wake)
    
    
############################################################################################### 
    # PLOTS
###############################################################################################   

    # plt.figure()
    # plt.title('Histogram of theta/delta ratio for all NREM epochs_' + name) 
    # plt.hist(ratio_sws, label = 'NREM')
    # plt.xlabel('log(theta/delta)')
    # plt.ylabel('Counts')
    # plt.hist(ratio_wake, label = 'wake')
    # plt.axvline(x=mean_ratio_sws,color='blue', label = 'mean NREM ratio')
    # plt.axvline(x=mean_ratio_wake,color='orange', label = 'mean wake ratio')
    # plt.legend(loc='upper right')
    # plt.show()

############################################################################################### 
    # WRITING FOR NEUROSCOPE
############################################################################################### 

    # start = new_sws_ep.as_units('ms')['start'].values
    # ends = new_sws_ep.as_units('ms')['end'].values

    # datatowrite = np.vstack((start,ends)).T.flatten()

    # n = len(new_sws_ep)

    # texttowrite = np.vstack(((np.repeat(np.array(['PyNewSWS start 1']), n)), 
    #                           (np.repeat(np.array(['PyNewSWS stop 1']), n))
    #                           )).T.flatten()

    # evt_file = rawpath + '/' + name + '.DM.new_sws.evt'
    # f = open(evt_file, 'w')
    # for t, n in zip(datatowrite, texttowrite):
    #     f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
    # f.close()        

    # start = new_wake_ep.as_units('ms')['start'].values
    # ends = new_wake_ep.as_units('ms')['end'].values

    # datatowrite = np.vstack((start,ends)).T.flatten()

    # n = len(new_wake_ep)

    # texttowrite = np.vstack(((np.repeat(np.array(['PyNewWake start 1']), n)), 
    #                           (np.repeat(np.array(['PyNewWake stop 1']), n))
    #                           )).T.flatten()

    # evt_file = rawpath + '/' + name + '.DM.new_wake.evt'
    # f = open(evt_file, 'w')
    # for t, n in zip(datatowrite, texttowrite):
    #     f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
    # f.close()
    

############################################################################################### 
    # DETECTION OF UP AND DOWN STATES
############################################################################################### 

    bin_size = 10000 #us
    rates = []

    for e in new_sws_ep.index:
        ep = new_sws_ep.loc[[e]]
        bins = np.arange(ep.iloc[0,0], ep.iloc[0,1], bin_size)       
        r = np.zeros((len(bins)-1))
        
        for n in spikes.keys(): 
            tmp = np.histogram(spikes[n].restrict(ep).index.values, bins)[0]
            r = r + tmp
        rates.append(pd.Series(index = bins[0:-1] + np.diff(bins)/2, data = r))
    rates = pd.concat(rates)
    total2 = rates.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2)
    
    
    # idx = total2[total2<np.percentile(total2,20)].index.values   
    # idx = total2[total2<np.percentile(total2,30)].index.values   
    idx = total2[total2<np.percentile(total2,10)].index.values   
    
    tmp2 = [[idx[0]]]
    
    for i in range(1,len(idx)):
        if (idx[i] - idx[i-1]) > bin_size:
            tmp2.append([idx[i]])
        elif (idx[i] - idx[i-1]) == bin_size:
            tmp2[-1].append(idx[i])

    # idx3 = total2[total2>np.percentile(total2,20)].index.values   
    # tmp3 = [[idx3[0]]]
    
    # for i in range(1,len(idx3)):
    #     if (idx3[i] - idx3[i-1]) > bin_size:
    #         tmp3.append([idx3[i]])
    #     elif (idx3[i] - idx3[i-1]) == bin_size:
    #         tmp3[-1].append(idx3[i])

    
    
    
    down_ep = np.array([[e[0],e[-1]] for e in tmp2 if len(e) > 1])
    down_ep = nts.IntervalSet(start = down_ep[:,0], end = down_ep[:,1])
    down_ep = down_ep.drop_short_intervals(bin_size)
    down_ep = down_ep.reset_index(drop=True)
    down_ep = down_ep.merge_close_intervals(bin_size*2)
    down_ep = down_ep.drop_short_intervals(bin_size*3)
    down_ep = down_ep.drop_long_intervals(bin_size*50)
   
    # sys.exit() 
   
    up_ep = nts.IntervalSet(down_ep['end'][0:-1], down_ep['start'][1:])
    down_ep = new_sws_ep.intersect(down_ep)
    
    up_ep = new_sws_ep.intersect(up_ep)
    
    
    # down_dur = down_ep.tot_length('s') / new_sws_ep.tot_length('s') 
    # all_down_dur.append(down_dur)
    
    #wake using same threshold 
    # threshold = np.percentile(total2,20)
    
    # bin_size = 10000 #us
    # rates = []

    # for e in new_wake_ep.index:
    #     ep = new_wake_ep.loc[[e]]
    #     bins = np.arange(ep.iloc[0,0], ep.iloc[0,1], bin_size)       
    #     r = np.zeros((len(bins)-1))
        
    #     for n in spikes.keys(): 
    #         tmp = np.histogram(spikes[n].restrict(ep).index.values, bins)[0]
    #         r = r + tmp
    #     rates.append(pd.Series(index = bins[0:-1] + np.diff(bins)/2, data = r))
    # rates = pd.concat(rates)
    # total2 = rates.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2)
    # idx = total2[total2<threshold].index.values
    
    # if idx.size > 0:
    #     tmp2 = [[idx[0]]]
    
    #     for i in range(1,len(idx)):
    #         if (idx[i] - idx[i-1]) > bin_size:
    #             tmp2.append([idx[i]])
    #         elif (idx[i] - idx[i-1]) == bin_size:
    #                 tmp2[-1].append(idx[i])
 
    #     down_wake_ep = np.array([[e[0],e[-1]] for e in tmp2 if len(e) > 1])
    #     down_wake_ep = nts.IntervalSet(start = down_wake_ep[:,0], end = down_wake_ep[:,1])
    #     down_wake_ep = down_wake_ep.drop_short_intervals(bin_size)
    #     down_wake_ep = down_wake_ep.reset_index(drop=True)
    #     down_wake_ep = down_wake_ep.merge_close_intervals(bin_size*2)
    #     down_wake_ep = down_wake_ep.drop_short_intervals(bin_size*3)
    #     down_wake_ep = down_wake_ep.drop_long_intervals(bin_size*50)
   
        
    #     down_wake_dur = down_wake_ep.tot_length('s') / new_wake_ep.tot_length('s')
    #     all_down_wake_dur.append(down_wake_dur)
        
    # else: 
    #     print('No down states detected during wake!')
    #     down_wake_dur = 0
    #     all_down_wake_dur.append(down_wake_dur)
############################################################################################### 
    # WRITING FOR NEUROSCOPE
############################################################################################### 

    start = down_ep.as_units('ms')['start'].values
    ends = down_ep.as_units('ms')['end'].values

    datatowrite = np.vstack((start,ends)).T.flatten()

    n = len(down_ep)

    texttowrite = np.vstack(((np.repeat(np.array(['PyDown start 1']), n)), 
                              (np.repeat(np.array(['PyDown stop 1']), n))
                              )).T.flatten()

    # evt_file = rawpath + '/' + name + '.evt.py.dow'
    # evt_file = rawpath + '/' + name + '.evt.py.d3w'
    evt_file = rawpath + '/' + name + '.evt.py.d1w'
    
    f = open(evt_file, 'w')
    for t, n in zip(datatowrite, texttowrite):
        f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
    f.close()        

    start = up_ep.as_units('ms')['start'].values
    ends = up_ep.as_units('ms')['end'].values

    datatowrite = np.vstack((start,ends)).T.flatten()

    n = len(up_ep)

    texttowrite = np.vstack(((np.repeat(np.array(['PyUp start 1']), n)), 
                              (np.repeat(np.array(['PyUp stop 1']), n))
                              )).T.flatten()

    # evt_file = rawpath + '/' + name + '.evt.py.upp'
    # evt_file = rawpath + '/' + name + '.evt.py.u3p'
    evt_file = rawpath + '/' + name + '.evt.py.u1p'
    
    
    f = open(evt_file, 'w')
    for t, n in zip(datatowrite, texttowrite):
        f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
    f.close()

############################################################################################### 
    # DOWN STATE DURATIONS FOR SLEEP AND WAKE
############################################################################################### 

# durs = np.zeros((len(all_down_dur), 2))
# durs[:, 0] = all_down_dur
# durs[:, 1] = all_down_wake_dur
# durations = pd.DataFrame(data=durs, columns=['Sleep', 'Wake'])

# w, p = wilcoxon(durations['Sleep'],durations['Wake'])

# x1 = ['Sleep'] * len(all_down_dur)
# x2 = ['Wake'] * len(all_down_dur)

# plt.figure()
# for i in range(len(x1)):     plt.plot([x1[i],x2[i]], [durations['Sleep'][i],durations['Wake'][i]],color='k', marker='o')
# plt.title('Down state detection quality')
# plt.ylabel('Fraction of epoch')
# plt.show()


#%%

#
# fig, ax = plt.subplots()
# [plt.plot(spikes[n].restrict(new_sws_ep).as_units('us').fillna(n), '|', color = 'k') for n in spikes.keys()]
# plt.plot((total2/max(total2))*100) 
# plt.axhline(np.percentile((total2/max(total2)*100),20), color = 'r')