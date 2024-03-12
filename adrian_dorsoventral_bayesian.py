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
from random import sample
from scipy.ndimage import gaussian_filter1d
import matplotlib.cm as cm
import seaborn as sns
import pycircstat

#%% 

def smoothAngularTuningCurves(tuning_curves, sigma=2):

    tmp = np.concatenate((tuning_curves.values, tuning_curves.values, tuning_curves.values))
    tmp = gaussian_filter1d(tmp, sigma=sigma, axis=0)

    return pd.DataFrame(index = tuning_curves.index,
        data = tmp[tuning_curves.shape[0]:tuning_curves.shape[0]*2], 
        columns = tuning_curves.columns
        )


def my_decoder(tuning_curves, group, ep, bin_size, time_units="s", feature=None):
    
        if isinstance(group, dict):
            newgroup = nap.TsGroup(group, time_support=ep)
        elif isinstance(group, nap.TsGroup):
            newgroup = group.restrict(ep)
        else:
            raise RuntimeError("Unknown format for group")
    
        if tuning_curves.shape[1] != len(newgroup):
            raise RuntimeError("Different shapes for tuning_curves and group")
    
        if not np.all(tuning_curves.columns.values == np.array(newgroup.keys())):
            raise RuntimeError("Difference indexes for tuning curves and group keys")
               
        
        count = newgroup.count(bin_size, ep, time_units)
        count = count.smooth(1,1)
        # Occupancy
        if feature is None:
            occupancy = np.ones(tuning_curves.shape[0])
        elif isinstance(feature, nap.Tsd):
            diff = np.diff(tuning_curves.index.values)
            bins = tuning_curves.index.values[:-1] - diff / 2
            bins = np.hstack(
                (bins, [bins[-1] + diff[-1], bins[-1] + 2 * diff[-1]])
            )  # assuming the size of the last 2 bins is equal
            occupancy, _ = np.histogram(feature.values, bins)
        else:
            raise RuntimeError("Unknown format for feature in decode_1d")
    
        # Transforming to pure numpy array
        tc = tuning_curves.values
        ct = count.values
    
        bin_size_s = nap.TsIndex.format_timestamps(
            np.array([bin_size], dtype=np.float64), time_units
        )[0]
    
        p1 = np.exp(-bin_size_s * tc.sum(1))
        p2 = occupancy / occupancy.sum()
    
        ct2 = np.tile(ct[:, np.newaxis, :], (1, tc.shape[0], 1))
    
        p3 = np.prod(tc**ct2, -1)
    
        p = p1 * p2 * p3
        p = p / p.sum(1)[:, np.newaxis]
    
        idxmax = np.argmax(p, 1)
    
        p = nap.TsdFrame(
            t=count.index, d=p, time_support=ep, columns=tuning_curves.index.values
        )
    
        decoded = nap.Tsd(
            t=count.index, d=tuning_curves.index.values[idxmax], time_support=ep
        )

        return decoded, p


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
    position = position.restrict(epochs['wake'])    

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

    tcurves_D = nap.compute_1d_tuning_curves(dorsal_spikes[dorsal_hd], feature = position['ang'].restrict(epochs['wake'].loc[[0]]), nb_bins = 361)
    smoothcurves_D = smoothAngularTuningCurves(tcurves_D, sigma=3)
    
    tcurves_V = nap.compute_1d_tuning_curves(ventral_spikes[ventral_hd], feature = position['ang'].restrict(epochs['wake'].loc[[0]]), nb_bins = 361)
    smoothcurves_V = smoothAngularTuningCurves(tcurves_V, sigma=3)
    
#%% UP state epochs where at least 5 HD cells from each population are active
    
    tmp1 = dorsal_spikes[dorsal_hd].count(0.05, up_ep).as_dataframe()
    tmp2 = ventral_spikes[ventral_hd].count(0.05, up_ep).as_dataframe()
    
    tmp1 = tmp1.sum(axis=1)
    tmp2 = tmp2.sum(axis=1)
         
    tmp1 = tmp1[tmp1 >= 5]
    tmp2 = tmp1[tmp2 >= 5]
    
    active_up = np.intersect1d(tmp1.index.values, tmp2.index.values)
       
    
#%% Decode during sleep
 
    d_D, p_feature_D = nap.decode_1d(tuning_curves = smoothcurves_D,  group = dorsal_spikes[dorsal_hd], ep = up_ep, bin_size = 0.05, 
                                           feature = position['ang'].restrict(epochs['wake'].loc[[0]]))
    
    d_V, p_feature_V = nap.decode_1d(tuning_curves = smoothcurves_V,  group = ventral_spikes[ventral_hd], ep = up_ep, bin_size = 0.05, 
                                           feature = position['ang'].restrict(epochs['wake'].loc[[0]]))
    
    decoded_D = nap.Ts(active_up).value_from(d_D)
    decoded_V = nap.Ts(active_up).value_from(d_V)

#%% Compute histogram of angular differences 

    # lin_error = np.abs(decoded_D.values - decoded_V.values)
    # lin_error = decoded_D.values - decoded_V.values
    
    
    # decode_error =  np.minimum((2*np.pi - abs(lin_error)), abs(lin_error))
    decode_error = np.abs(pycircstat.cdiff(decoded_D.values, decoded_V.values))
    allerr.append(np.mean(decode_error))
    
#%% Plotting rasters
    
    # active_ep = nap.IntervalSet(start = active_up[0:-2], end = active_up[1:-1])
    # ex_ep = nap.IntervalSet(start = 535, end = 538)
    # ex_ep = nap.IntervalSet(start = 1680, end = 1700)
    # ex_ep = nap.IntervalSet(start = 1800, end = 1820)
    
    # d_D, p_feature_D = nap.decode_1d(tuning_curves = tcurves_D,  group = dorsal_spikes[dorsal_hd], ep = epochs['wake'], bin_size = 0.25, 
    #                                         feature = position['ang'])
    
    # d_V, p_feature_V = nap.decode_1d(tuning_curves = tcurves_V,  group = ventral_spikes[ventral_hd], ep = epochs['wake'], bin_size = 0.25, 
    #                                         feature = position['ang'])
    
    # decoded_D = d_D
    # decoded_V = d_V
    
    # pref_ang_D = []
    # pref_ang_V = []
    
    # # ex_ep = nap.IntervalSet(start = 3116, end = 3125)

    # for i in tcurves_D.columns:
    #     pref_ang_D.append(tcurves_D.loc[:,i].idxmax())
        
    # for i in tcurves_V.columns:
    #     pref_ang_V.append(tcurves_V.loc[:,i].idxmax())
    
    # norm = plt.Normalize() #Normalizes data into the range [0,1]      
    # color_D = plt.cm.hsv(norm([i/(2*np.pi) for i in pref_ang_D]))
    # color_V = plt.cm.hsv(norm([i/(2*np.pi) for i in pref_ang_V]))#Assigns a colour in the HSV colourmap for each value of preferred angle 
    

    # plt.figure()
    # plt.tight_layout()
    # plt.subplot(311)
    # plt.ylabel('Ang (rad)')
    # plt.plot(decoded_D.restrict(ex_ep), 'o-', label = 'dorsal')
    # plt.plot(decoded_V.restrict(ex_ep),  'o-', label = 'ventral')
    # plt.xlim([ex_ep['start'], ex_ep['end']])
    # plt.legend(loc = 'upper right')
    
    # plt.subplot(312)
    # plt.title('dorsal')
    # plt.ylabel('Ang (rad)')
    # plt.xlim([ex_ep['start'], ex_ep['end']])
    # for i,n in enumerate(dorsal_spikes[dorsal_hd]):
    #     plt.plot(dorsal_spikes[dorsal_hd][n].restrict(ex_ep).fillna(pref_ang_D[i]), '|', color = color_D[i])
        
    # plt.subplot(313)
    # plt.title('ventral')
    # plt.xlabel('time(s)')
    # plt.ylabel('Ang (rad)')
    # plt.xlim([ex_ep['start'], ex_ep['end']])
    # for i,n in enumerate(ventral_spikes[ventral_hd]):
    #     plt.plot(ventral_spikes[ventral_hd][n].restrict(ex_ep).fillna(pref_ang_V[i]), '|', color = color_V[i])
   

#%% SANITY CHECK - DECODE WAKE
    
    # center = position.restrict(epochs['wake'].loc[[0]]).time_support.get_intervals_center()

    # halves = nap.IntervalSet(start = [position.restrict(epochs['wake'].loc[[0]]).start, center.t[0]],
    # end = [center.t[0], position.restrict(epochs['wake'].loc[[0]]).end])


    # tcurves = nap.compute_1d_tuning_curves(spikes[hd], feature = position['ang'].restrict(epochs['wake'].loc[[0]]), nb_bins = 361)
    # smoothcurves = smoothAngularTuningCurves(tcurves, sigma=3)

    # d, pfeature = my_decoder(tuning_curves = smoothcurves,  group = spikes[hd], ep = epochs['wake'].loc[[0]], bin_size = 0.25, 
    #                                         feature = position['ang'])

 
    # prefang = []
        
    # ex_ep = nap.IntervalSet(start = 2880, end = 2885)

    # for i in tcurves.columns:
    #     prefang.append(tcurves.loc[:,i].idxmax())
        
    # norm = plt.Normalize() #Normalizes data into the range [0,1]      
    # color = plt.cm.hsv(norm([i/(2*np.pi) for i in prefang]))

    # plt.figure(figsize=(12, 9))
    # for i, n in enumerate(spikes[hd]): 
    #     plt.subplot(10, 9, i + 1, projection='polar')  # Plot the curves in 8 rows and 4 columns
    #     plt.plot(smoothcurves[n], color=color[i])  # Colour of the curves determined by preferred angle    
    #     plt.xlabel("Angle (rad)")  # Angle in radian, on the X-axis
    #     plt.ylabel("Firing Rate (Hz)")  # Firing rate in Hz, on the Y-axis
    #     plt.xticks([])
    #     plt.show()

    # angdiff = np.abs(pycircstat.cdiff(d.values, d.value_from(position['ang']).values))
    # plt.figure()
    # plt.subplot(211)
    # plt.plot(d.value_from(position['ang']).restrict(ex_ep), 'o-', label = 'true')
    # plt.plot(d.restrict(ex_ep), 'o-', label = 'decoded')
    # plt.legend(loc = 'upper right')
    # plt.subplot(212)
    # for i,n in enumerate(spikes[hd]):
    #     plt.plot(spikes[hd][n].restrict(ex_ep).fillna(prefang[i]), '|', color = color[i])
    
#%% 

    # bins = np.linspace(0, np.pi, 61)    
    # bins = np.linspace(0, np.pi, 61)    
    
    # relcounts_all,_ = np.histogram(decode_error, bins)     
    # relcounts_all,_ = np.histogram(lin_error, bins)     
    # p_rel = relcounts_all/sum(relcounts_all)
    # angprop.append(p_rel)
    
    # h, mu, ci = pycircstat.mtest(decode_error,0)
    # h, mu, ci = pycircstat.mtest(lin_error,0)
    # allh.append(h)
    # allmu.append(mu)
    # allci.append(ci)
    # print(h, mu, ci)
    
#%% Plotting for each session

    # plt.figure()
    # plt.title(s)
    # plt.stairs(p_rel, bins, linewidth = 2, color = 'k')
    # plt.xlabel('DV decoded ang diff (rad)')
    # plt.ylabel('% events')
    # plt.gca().set_box_aspect(1)
    
#%% Shuffle the DV location of each cell

    derr_shu = []

    for i in range(100):
        depth_shu = sample(list(spkdata['depth'].values), len(spkdata))
        spk_shu = pd.DataFrame(index = spkdata.index)   
        spk_shu['depth'] = depth_shu
        spk_shu['level'] = pd.cut(spk_shu['depth'],2, precision=0, labels=[1,0]) #0 is dorsal, 1 is ventral
         
        dorsal_shu = spikes[spk_shu.index[spk_shu['level'] == 0]]
        ventral_shu = spikes[spk_shu.index[spk_shu['level'] == 1]]      

        dshu_hd = np.intersect1d(hd, dorsal_shu.index)
        vshu_hd = np.intersect1d(hd, ventral_shu.index)    
        
        tc_D_shu = nap.compute_1d_tuning_curves(dorsal_shu[dshu_hd], feature = position['ang'], nb_bins = 61)
        tc_V_shu = nap.compute_1d_tuning_curves(ventral_shu[vshu_hd], feature = position['ang'], nb_bins = 61)
        
        tmp1_shu = dorsal_shu[dshu_hd].count(0.025, up_ep).as_dataframe().sum(axis=1)
        tmp2_shu = ventral_shu[vshu_hd].count(0.025, up_ep).as_dataframe().sum(axis=1)
        
        tmp1_shu = tmp1_shu[tmp1_shu >= 5]
        tmp2_shu = tmp2_shu[tmp2_shu >= 5]
        
        active_up_shu = np.intersect1d(tmp1_shu.index.values, tmp2_shu.index.values)
        
        d_D_shu, _ = nap.decode_1d(tuning_curves = tc_D_shu,  group = dorsal_shu[dshu_hd], ep = up_ep, bin_size = 0.025, 
                                               feature = position['ang'])
        
        d_V_shu, _ = nap.decode_1d(tuning_curves = tc_V_shu,  group = ventral_shu[vshu_hd], ep = up_ep, bin_size = 0.025, 
                                               feature = position['ang'])
        
        decoded_D_shu = nap.Ts(active_up_shu).value_from(d_D_shu)
        decoded_V_shu = nap.Ts(active_up_shu).value_from(d_V_shu)
        
        lin_error_shu = np.abs(decoded_D_shu.values - decoded_V_shu.values)
                
        decode_error_shu =  np.minimum((2*np.pi - abs(lin_error_shu)), abs(lin_error_shu))
        derr_shu.append(np.mean(decode_error_shu))
        
        bins = np.linspace(0, np.pi, 61)    
        relcounts_all,_ = np.histogram(derr_shu, bins)     
        p_rel = relcounts_all/sum(relcounts_all)
        np.save(rawpath + '/' + s + '_shuffle.npy', np.array(derr_shu))
        
#%% Plot for each session shuffles and data 

    # plt.figure()
    # plt.title(s)
    # plt.hist(derr_shu, label = 'shuffle')
    # # plt.stairs(p_rel, bins,  label = 'shuffle')
    # plt.axvline(np.mean(decode_error), color = 'k', label = 'data')
    # plt.xlabel('DV decoded ang diff (rad)')
    # plt.ylabel('Counts')
    # plt.legend(loc = 'upper right')
        
#%% 

# testdata = spikes[hd].restrict(epochs['wake'].loc[[0]])
# testcount = testdata.count(0.25)[:,0]

# sm1 = testcount.smooth(1,2)
# sm2 = testcount.smooth(1,3)
# sm3 = testcount.smooth(1,100)
# sm4 = testcount.smooth(3,100)
# sm5 = testcount.smooth(10,100)

# plt.figure()
# plt.plot(testcount, label = 'binned')
# plt.plot(sm1, label = '1,2')
# plt.plot(sm2, label = '1,3')
# plt.plot(sm3, label = '1,100')
# plt.plot(sm4, label = '3,100')
# plt.plot(sm5, label = '10,100')
# plt.legend(loc = 'upper right')
        
        