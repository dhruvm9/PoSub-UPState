#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 17:53:35 2021

@author: dhruv
"""

############################################################################################### 
    # LOADING DATA
###############################################################################################

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
from scipy.stats import pearsonr, mannwhitneyu 

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
# datasets = np.loadtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
datasets = np.loadtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'

diff_ex = []
diff_FS = []
depths_ex = []
depths_FS = []

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
            

# ############################################################################################### 
#     # LOAD EPOCHS
# ############################################################################################### 
    
#     if s == 'A3701-191119':
#         sleep1_ep = np.hstack([behepochs['sleepPreEp'][0][0][1],behepochs['sleepPreEp'][0][0][2]])
#         sleep1_ep = nts.IntervalSet(sleep1_ep[:,0], sleep1_ep[:,1], time_units = 's').drop_short_intervals(0.0)

#         sleep2_ep = np.hstack([behepochs['sleepPostEp'][0][0][1],behepochs['sleepPostEp'][0][0][2]])
#         sleep2_ep = nts.IntervalSet(sleep2_ep[:,0], sleep2_ep[:,1], time_units = 's').drop_short_intervals(0.0)
        
#         wake_ep = np.hstack([behepochs['wakeEp'][0][0][1],behepochs['wakeEp'][0][0][2]])
#         wake_ep = nts.IntervalSet(wake_ep[:,0], wake_ep[:,1], time_units = 's').drop_short_intervals(0.0)
        
            
#     else: 
#         wake_ep = np.hstack([behepochs['wake1Ep'][0][0][1],behepochs['wake1Ep'][0][0][2]])
#         wake_ep = nts.IntervalSet(wake_ep[:,0], wake_ep[:,1], time_units = 's').drop_short_intervals(0.0)
        
#         sleep1_ep = np.hstack([behepochs['sleep1Ep'][0][0][1],behepochs['sleep1Ep'][0][0][2]])
        
#          #check if it is not empty, then go to next step
#         if sleep1_ep.size != 0:
#             print('sleep1 exists')
#             sleep1_ep = nts.IntervalSet(sleep1_ep[:,0], sleep1_ep[:,1], time_units = 's').drop_short_intervals(0.0)
                

#         sleep2_ep = np.hstack([behepochs['sleep2Ep'][0][0][1],behepochs['sleep2Ep'][0][0][2]])
        
# #         #check if it is not empty, then go to next step
#         if sleep2_ep.size != 0: 
#             print('sleep2 exists')
#             sleep2_ep = nts.IntervalSet(sleep2_ep[:,0], sleep2_ep[:,1], time_units = 's').drop_short_intervals(0.0)
        
        
        
# #         #if both sleep1 and sleep2 are not empty, merge them. Else make the non-empty epoch sleep_ep
#     if (sleep1_ep.size !=0 and sleep2_ep.size !=0): 
        
#         if (sleep1_ep.start.values[0] > sleep2_ep.start.values[0]):
#             sleep1_ep, sleep2_ep = sleep2_ep, sleep1_ep
    
#         sleep_ep = pd.concat((sleep1_ep, sleep2_ep)).reset_index(drop=True)
            
          
#     elif sleep1_ep.size != 0:
#         sleep_ep = sleep1_ep
            
#     else: 
#         sleep_ep = sleep2_ep   
        
    # file = os.path.join(rawpath, name + '.lfp')
     
    # if os.path.exists(file):    
    #       lfp = loadLFP(os.path.join(rawpath, name + '.lfp'), n_channels, 1, 1250, 'int16')
    # else: 
    #       lfp = loadLFP(os.path.join(rawpath, name + '.eeg'), n_channels, 1, 1250, 'int16')
   
    # lfp = downsample(lfp, 1, 5)
    
    # acceleration = loadAuxiliary(rawpath, 1, fs = 20000) 
    # newsleep_ep = refineSleepFromAccel(acceleration, sleep_ep)    
        
    # file = os.path.join(rawpath, name +'.sws.evt')
    # if os.path.exists(file):
    #     tmp = np.genfromtxt(file)[:,0]
    #     tmp = tmp.reshape(len(tmp)//2,2)/1000
    #     sws_ep1 = nts.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
        
    # sws_ep = newsleep_ep.intersect(sws_ep1)
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
        
# ############################################################################################### 
#     # THETA/DELTA RATIO FOR SWS
# ###############################################################################################   
    
    # new_sws_start = []
    # new_sws_end = []
    # ratio_sws = []
    # ratio_wake = [] 

    # for i in sws_ep.index:  
    #     sws_lfp = lfp.restrict(sws_ep.loc[[i]])
    
    #     lfp_filt_theta = nts.TsFigure_d(sws_lfp.index.values, butter_bandpass_filter(sws_lfp, 4, 12, 1250/5, 2))
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
    
# ############################################################################################### 
#     # REFINE WAKE 
# ###############################################################################################      

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
    

# ############################################################################################### 
#     # THETA/DELTA RATIO FOR WAKE
# ###############################################################################################  
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
    # COMPUTE MEAN FIRING RATES
###############################################################################################  
        
    #WAKE v/s NREM
    rates = computeMeanFiringRate(spikes, [new_wake_ep, new_sws_ep, up_ep], ['wake', 'sws', 'up'])
    
    # corr_pyr, p_pyr = pearsonr(rates['sws'][pyr], rates['wake'][pyr])
    
    # if len(interneuron) > 1:
    #     corr_int, p_int = pearsonr(rates['sws'][interneuron], rates['wake'][interneuron]) 
        
    # plt.figure()
    # plt.title('Wake FR v/s NREM FR for PYR/INT_' + name) 
    # plt.scatter(x = np.array(rates['sws'][pyr]), y = np.array(rates['wake'][pyr]),label = 'Pearson r (PYR) = ' + str(corr_pyr))
    # plt.scatter(x = np.array(rates['sws'][interneuron]), y = np.array(rates['wake'][interneuron]),label = 'Pearson r (INT) = ' + str(corr_int))
    # plt.xlabel('NREM firing rate (Hz)')
    # plt.ylabel('Wake firing rate (Hz)')
    # plt.legend(loc='upper right')
    # plt.show()
    
       
    # corr, p = pearsonr(rates['sws'][hd], rates['wake'][hd])
    # plt.figure()
    # plt.title('Wake FR v/s NREM FR for HD cells_' + name) 
    # plt.scatter(x = np.array(rates['sws'][hd]), y = np.array(rates['wake'][hd]),label = 'Pearson r = ' + str(corr))
    # plt.xlabel('NREM firing rate (Hz)')
    # plt.ylabel('Wake firing rate (Hz)')
    # plt.legend(loc='upper right')
    # plt.show()

    #WAKE v/s UP STATE
    
    # corr_pyr, p_pyr = pearsonr(rates['up'][pyr], rates['wake'][pyr])
    # if len(interneuron) > 1:
    #   corr_int, p_int = pearsonr(rates['up'][interneuron], rates['wake'][interneuron])   
    
    # plt.figure()
    # plt.title('Wake FR v/s UP State FR for PYR/INT_' + name) 
    # plt.scatter(x = np.array(rates['up'][pyr]), y = np.array(rates['wake'][pyr]),label = 'Pearson r (PYR) = ' + str(corr_pyr))
    # plt.scatter(x = np.array(rates['up'][interneuron]), y = np.array(rates['wake'][interneuron]),label = 'Pearson r (INT) = ' + str(corr_int))
    # plt.xlabel('UP State firing rate (Hz)')
    # plt.ylabel('Wake firing rate (Hz)')
    # plt.legend(loc='upper right')
    # plt.show()
   
    # corr, p = pearsonr(rates['up'][hd], rates['wake'][hd])
    # plt.figure()
    # plt.title('Wake FR v/s UP State FR for HD cells_' + name) 
    # plt.scatter(x = np.array(rates['up'][hd]), y = np.array(rates['wake'][hd]),label = 'Pearson r = ' + str(corr))
    # plt.xlabel('UP State firing rate')
    # plt.ylabel('Wake firing rate')
    # plt.legend(loc='upper right')
    # plt.show()
    

    # #NREM v/s UP STATE
    
    # corr_pyr, p_pyr = pearsonr(rates['up'][pyr], rates['sws'][pyr])
    # if len(interneuron) > 1:
    #   corr_int, p_int = pearsonr(rates['up'][interneuron], rates['sws'][interneuron]) 
     
    # plt.figure()
    # plt.title('NREM FR v/s UP State FR for PYR/INT_' + name) 
    # plt.scatter(x = np.array(rates['up'][pyr]), y = np.array(rates['sws'][pyr]),label = 'Pearson r (PYR) = ' + str(corr_pyr))
    # plt.scatter(x = np.array(rates['up'][interneuron]), y = np.array(rates['sws'][interneuron]),label = 'Pearson r (INT) = ' + str(corr_int))
    # plt.xlabel('UP State firing rate')
    # plt.ylabel('NREM firing rate')
    # plt.legend(loc='upper right')
    # plt.show()
   
    # corr, p = pearsonr(rates['up'][hd], rates['sws'][hd])
    # plt.figure()
    # plt.title('NREM FR v/s UP State FR for HD cells_' + name) 
    # plt.scatter(x = np.array(rates['up'][hd]), y = np.array(rates['sws'][hd]),label = 'Pearson r = ' + str(corr))
    # plt.xlabel('UP State firing rate')
    # plt.ylabel('NREM firing rate')
    # plt.legend(loc='upper right')
    # plt.show()
    
###############################################################################
###Stats for Wake v/s UP state firing rates
###############################################################################
    d_ex = np.log10(rates['wake'][pyr].astype(np.float64)) - np.log10(rates['sws'][pyr].astype(np.float64))
    diff_ex.append(d_ex)
    depths_ex.append(depth[pyr])
    d_FS = np.log10(rates['wake'][interneuron].astype(np.float64)) - np.log10(rates['sws'][interneuron].astype(np.float64))
    
    diff_FS.append(d_FS)
    depths_FS.append(depth[interneuron])

diff_ex = np.concatenate(diff_ex)
diff_FS = np.concatenate(diff_FS)

depths_ex = np.concatenate(depths_ex)
depths_ex = depths_ex.flatten()
depths_FS = np.concatenate(depths_FS)
depths_FS = depths_FS.flatten()

# bins = np.linspace(min(min(diff_FS),min(diff_FS)),max(max(diff_FS),max(diff_FS)))

# plt.figure()
# plt.title('Mean FR difference')
# plt.xlabel('log(Wake FR) - log (NREM FR)')
# plt.ylabel('Number of cells')
# plt.hist(diff_ex,bins, alpha = 0.5, label = 'Excitatory cells')
# plt.hist(diff_FS,bins, alpha = 0.5, label = 'FS cells')
# plt.legend(loc = 'upper right')

# t,pvalue = mannwhitneyu(diff_ex,diff_FS)

corr_ex, p_ex = pearsonr(diff_ex,depths_ex)
corr_FS, p_FS = pearsonr(diff_FS,depths_FS)

plt.figure()
plt.title('FR difference as a function of depth')
plt.xlabel('log(Wake FR) - log (NREM FR)')
plt.ylabel('Depth from top of probe (um)')
plt.scatter(diff_ex,depths_ex, label = 'R (ex) = ' + str(np.round(corr_ex,4)))
plt.scatter(diff_FS,depths_FS, label = 'R (FS) = ' + str(np.round(corr_FS,4)))
plt.legend(loc = 'upper right')



    
  
    

    