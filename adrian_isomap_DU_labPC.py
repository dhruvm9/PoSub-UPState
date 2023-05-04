#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 13:05:05 2023

@author: dhruv
"""

import numpy as np 
import pandas as pd 
import scipy.io
import pynapple as nap 
import os, sys
import time 
import matplotlib.pyplot as plt 
from random import sample, choice, uniform
from math import sin, cos
from matplotlib.colors import hsv_to_rgb
from scipy.stats import kendalltau, pearsonr, wilcoxon, mannwhitneyu
from itertools import combinations
import seaborn as sns
import scipy.signal
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

#%% 

def full_ang(ep,position, nb_bins):
    
    starts = []
    ends = []
    count = np.zeros(nb_bins-1)
    bins = np.linspace(0,2*np.pi,nb_bins)
    ang_pos = position.restrict(ep)
    ang_time = ang_pos.times()

    idx = np.digitize(ang_pos,bins)-1
    
    start = 0
    for i,j in enumerate(idx):
        count[j] += 1
        if np.all(count >= 1):
            starts.append(start)
            ends.append(i)
            count = np.zeros(nb_bins-1)
            start = i+1
            
    t_start = ang_time[starts]
    t_end = ang_time[ends]
    
    full_ang_ep = nap.IntervalSet(start = t_start, end = t_end)
    
    return full_ang_ep

#%% 

def perievent_Tsd(data, tref,  minmax):
    peth = {}
    
    a = data.index[data.index.get_indexer(tref.index.values, method='nearest')]
    
    tmp = nap.compute_perievent(data, nap.Ts(a.values) , minmax = minmax, time_unit = 's')
    peth_all = []
    for j in range(len(tmp)):
        #if len(tmp[j]) >= 400: #TODO: Fix this - don't hard code
        peth_all.append(tmp[j].as_series())
    peth['all'] = pd.concat(peth_all, axis = 1, join = 'outer')
    peth['mean'] = peth['all'].mean(axis = 1)
    return peth


#%% 

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'

for s in datasets:
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    rawpath = os.path.join(rwpath,s)

    data = nap.load_session(rawpath, 'neurosuite')
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
    
    file = [f for f in listdir if 'Layers' in f]
    lyr = scipy.io.loadmat(os.path.join(filepath,file[0]))
    layer = lyr['l']
    
    file = [f for f in listdir if 'Velocity' in f]
    vl = scipy.io.loadmat(os.path.join(filepath,file[0]), simplify_cells = True)
    vel = nap.Tsd(t = vl['vel']['t'], d = vl['vel']['data']  )
    
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


#%% 

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
        
#%% 

    filepath = os.path.join(path, 'Analysis')
    data = pd.read_csv(filepath + '/Tracking_data.csv', header = None)
    position = pd.DataFrame(index = data[0].values, data = data[[1,2,3]].values, columns=['x', 'y', 'ang'])
    position = position.loc[~position.index.duplicated(keep='first')]
    position['ang'] = position['ang'] *(np.pi/180) #convert degrees to radian
    position['ang'] = (position['ang'] + 2*np.pi) % (2*np.pi) #convert [-pi, pi] to [0, 2pi]
    position = nap.TsdFrame(position) 
    position = position.restrict(epochs['wake'])
    
    angle = position['ang'].loc[vel.index.values]
    
#%%  Visualization
    
### Wake binning 

    # wake_bin = 0.2 #200ms binwidth
    
    # ep = full_ang(angle.time_support, angle, 60)
      
    # dx = position['x'].bin_average(wake_bin,ep)
    # dy = position['y'].bin_average(wake_bin,ep)
    # ang = angle.bin_average(wake_bin, ep)
    # v = vel.bin_average(wake_bin,ep)

    # v = v.threshold(2)
       
    # ang = ang.loc[v.index.values]
    
    # wake_count = spikes[hd].count(wake_bin, ep) 
    # wake_count = wake_count.loc[v.index.values]
           
    # wake_count = wake_count.as_dataframe()
    # wake_rate = np.sqrt(wake_count/wake_bin)
    # wake_rate = wake_rate.rolling(window = 50, win_type = 'gaussian', center = True, min_periods = 1, axis = 0).mean(std = 3)
        
    # samples = sample(list(np.arange(0,len(ep)-1)), round(len(ep)/5))
    # samples = np.sort(samples)
    
    # sub_ep = nap.IntervalSet(start = ep.iloc[samples]['start'], end = ep.iloc[samples]['end'])    
    
    # ang = ang.restrict(sub_ep)
    # ang = ang.dropna()
      
    # wake_rate = wake_rate.loc[ang.index.values]

### Sleep binning 

    # du = nap.IntervalSet(start = up_ep['start'] - 0.025, end = up_ep['start'] + 0.15) 
    
    # sleep_dt = 0.025 
    # sleep_binwidth = 0.1

    # num_overlapping_bins = int(sleep_binwidth/sleep_dt)    
    
    # sleep_count = spikes[hd].count(sleep_dt, du.loc[[0]]) 
    # sleep_count = sleep_count.as_dataframe()
    # sleep_rate = np.sqrt(sleep_count/sleep_dt)
    # sleep_rate = sleep_rate.rolling(window = num_overlapping_bins, win_type = 'gaussian', center = True, min_periods = 1, axis = 0).mean(std = 3)
        
    # rate = np.vstack([wake_rate.values, sleep_rate.values])
               
    # projection = Isomap(n_components = 2, n_neighbors = 20).fit_transform(rate)
        
    # H = ang.values/(2*np.pi)
    # HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T
    # RGB = hsv_to_rgb(HSV)    
       
    # cmap = plt.colormaps['Greys']
    # col = np.linspace(0,1,len(projection[:,0][len(wake_rate)+1:]))
    
    
    # dx = np.diff(projection[:,0][len(wake_rate)+1:])
    # dy = np.diff(projection[:,1][len(wake_rate)+1:])
    
    # plt.figure(figsize = (8,8))
    # plt.scatter(projection[:,0][0:len(wake_rate)], projection[:,1][0:len(wake_rate)], c = RGB)
    
    # for i in range(len(projection[:,0][len(wake_rate)+1:])):
    #     plt.plot(projection[:,0][i], projection[:,1][i], 'o-', color = cmap(col)[i])
        
    
    
    # plt.xticks([])
    # plt.yticks([])
    
    
#%% Analysis of radial and angular velocity 
   
### Wake binning 

    wake_bin =  0.2 #0.4 #400ms binwidth
    
    ep = epochs['wake']
      
    dx = position['x'].bin_average(wake_bin,ep)
    dy = position['y'].bin_average(wake_bin,ep)
    ang = angle.bin_average(wake_bin, ep)
    v = vel.bin_average(wake_bin,ep)

    v = v.threshold(2)
       
    ang = ang.loc[v.index.values]
    
    wake_count = spikes[hd].count(wake_bin, ep) 
    wake_count = wake_count.loc[v.index.values]
           
    wake_count = wake_count.as_dataframe()
    wake_rate = np.sqrt(wake_count/wake_bin)
    wake_rate = wake_rate.rolling(window = 50, win_type = 'gaussian', center = True, min_periods = 1, axis = 0).mean(std = 3)
        
    ang = ang.dropna()
    ang = ang.rename('ang')
      
    wake_rate = wake_rate.loc[ang.index.values]
        
    wake_rate = pd.concat([wake_rate, pd.DataFrame(ang)], axis = 1)
    wake_rate = wake_rate.sample(frac = 0.5)

### Sleep binning 

    sleep_dt = 0.01 #0.015 #0.025 #0.015 
    sleep_binwidth = 0.05 #0.1 #0.03

    tokeep = np.where((up_ep['end'] - up_ep['start']) > 0.5)
    
    du = nap.IntervalSet(start = up_ep.iloc[tokeep]['start'] - 0.25, end = up_ep.iloc[tokeep]['start'] + 0.5) 
       
    refs = []
    
    for i in tokeep[0]:
        rng = uniform(0,1)
        times = (up_ep.iloc[i]['start'] + (rng * (up_ep.iloc[i]['end'] - up_ep.iloc[i]['start'])))
        refs.append(times)
     
    rnd = nap.IntervalSet(start = pd.Series(refs).values - 0.25, end = pd.Series(refs).values + 0.25)
    
    
    num_overlapping_bins = int(sleep_binwidth/sleep_dt)    
    
    sleep_count = spikes[hd].count(sleep_dt,du) 
    sleep_count = sleep_count.as_dataframe()
    sleep_rate = np.sqrt(sleep_count/sleep_dt)
    sleep_rate = sleep_rate.rolling(window = num_overlapping_bins, win_type = 'gaussian', center = True, min_periods = 1, axis = 0).mean(std = 3)
        
    rnd_count = spikes[hd].count(sleep_dt,rnd) 
    rnd_count = rnd_count.as_dataframe()
    rnd_rate = np.sqrt(rnd_count/sleep_dt)
    rnd_rate = rnd_rate.rolling(window = num_overlapping_bins, win_type = 'gaussian', center = True, min_periods = 1, axis = 0).mean(std = 3)
    
    rate = np.vstack([wake_rate.loc[:, wake_rate.columns != 'ang'].values, sleep_rate.sample(frac = 0.01).values]) #Take 1000 random values 
               
    fit = Isomap(n_components = 2, n_neighbors = 200).fit(rate) 
    p_wake = fit.transform(wake_rate.loc[:, wake_rate.columns != 'ang'])    
    p_sleep = fit.transform(sleep_rate)
    p_rnd = fit.transform(rnd_rate)
    

    H = wake_rate['ang'].values/(2*np.pi)
    HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T
    RGB = hsv_to_rgb(HSV)    
    
    truex = p_sleep[np.where(sleep_rate.sum(axis=1)==0)][0][0]
    truey = p_sleep[np.where(sleep_rate.sum(axis=1)==0)][0][1]
    
    p_wake = p_wake - [truex, truey]
    p_sleep = p_sleep - [truex, truey]
    p_rnd = p_rnd - [truex, truey]
                
    projection = nap.TsdFrame(t = sleep_rate.index.values, d = p_sleep, columns = ['x', 'y'])
    ctrl = nap.TsdFrame(t = rnd_rate.index.values, d = p_rnd, columns = ['x', 'y'] )

#%% 

##Angular direction, radius and velocity     
    # angdir = nap.Tsd(t = projection.index.values, d = np.arctan2(projection['y'].values, projection['x'].values))
    # radius = nap.Tsd(t = projection.index.values, 
    #                   d = np.sqrt((projection['x'].values**2) 
    #                   + (projection['y'].values**2)))
    
    # peth_radius = perievent_Tsd(radius, nap.Tsd(up_ep.iloc[tokeep]['start'].values), (-0.025, 0.5))
    # peth_radius['all'] = peth_radius['all'][0:0.48]
    # peth_radius['mean'] = peth_radius['mean'][0:0.48]
    
    # peth_angle = perievent_Tsd(nap.Tsd(angdir), nap.Tsd(up_ep.iloc[tokeep]['start'].values), (-0.025, 0.5))
    # peth_angle['all'] = peth_angle['all'][0:0.48]
    # peth_angle['mean'] = peth_angle['mean'][0:0.48]
    
        
#%% 
    angdiff = (angdir + 2*np.pi)%(2*np.pi)
    angdiff = np.unwrap(angdiff)
    
    angs = pd.Series(index = projection.index.values, data = angdiff)
    angs = angs.rolling(window = 20, win_type='gaussian', center=True, min_periods=1).mean(std=1)
    
    # angdiff = np.minimum((2*np.pi - abs(angdiff.values)), abs(angdiff.values))
    
    angvel = nap.Tsd(t = projection.index.values, d = np.abs(np.gradient(angs.values)))    
    # radvel = nap.Tsd(t = projection.index.values[:-1], d = (np.diff(radius.values)))
    
    
    
    
        
    # peth_angvel = perievent_Tsd(angvel, nap.Tsd(up_ep.iloc[tokeep]['start'].values), (-0.025, 0.5))
    # peth_angvel['all'] = peth_angvel['all'][0:0.48]
    # peth_angvel['mean'] = peth_angvel['mean'][0:0.48]
   
   #%%  
   
    # plt.figure(figsize = (20,10))
    # plt.subplot(131)
    # plt.scatter(p_wake[:,0], p_wake[:,1], c = RGB[0:3000], zorder = 2)
    # plt.scatter(p_sleep[:,0], p_sleep[:,1], c = 'silver') 
   
    # plt.subplot(132)
    # plt.plot(peth_radius['all'], color = 'silver', linewidth = 0.5) 
    # plt.plot(peth_radius['mean'], color = 'k', linewidth = 2)
   
    # plt.subplot(133)
    # plt.plot(peth_angvel['all'], color = 'silver', linewidth = 0.5) 
    # plt.plot(peth_angvel['mean'], color = 'k', linewidth = 2)

    #%% 
    
    
    
### Select Individual examples and plot them 
           
    examples = [0,1,4,7,9,69,420, 616, 666, 786]
    tokeep = []
    
    angkeep = []
    akeep2 = []
            
    #Pick radius from histogram here
    radius = nap.Tsd(t = projection.index.values, 
                       d = np.sqrt((projection['x'].values**2) 
                      + (projection['y'].values**2)))
    
    ctrl_radius = nap.Tsd(t = ctrl.index.values, 
                       d = np.sqrt((ctrl['x'].values**2) 
                      + (ctrl['y'].values**2)))
    
    
    bins = np.arange(0, np.ceil(radius.max()), 1)
    
    counts, bins = np.histogram(radius,bins)
    radius_index = scipy.signal.argrelmin(counts)
    
    ringradius = int(bins[radius_index][0])
    
    # plt.figure()
    # plt.plot(counts)
    # plt.axvline(ringradius, color = 'k')
      
    
    plt.figure()    
    # plt.scatter(p_wake[:,0], p_wake[:,1], c = RGB)
    
    for k in range(len(du)): #examples: #range(len(du)): #examples: #range(len(du)):  

        traj = projection.restrict(nap.IntervalSet(start = du.loc[[k]]['start'], 
                                                   end = du.loc[[k]]['end']))
        traj.index = traj.index.values - (du.loc[[k]]['start'].values + 0.25)
        traj.index = [round (j,4) for j in traj.index.values]                               
                                   
        traj = traj.as_dataframe()
                  
        vec = traj[-0.25:0.15]
        # tx = vec['x'] - vec['x'].loc[0]
        # ty = vec['y'] - vec['y'].loc[0]
        # tr = nap.Tsd(t = tx. index. values, d = np.sqrt((tx.values**2) + (ty.values**2)))
        tr = nap.Tsd(t = vec. index. values, d = np.sqrt((vec['x'].values**2) + (vec['y'].values**2)))
  
                
        ix = np.where(tr[0:0.15].values > ringradius)[0]
        iy = np.where(tr[-0.25:0].values > ringradius)[0]
        
        if (len(ix) > 0) and (len(iy) > 0):
           
            ix1 = ix[0]
            
            tokeep.append(k)     
            
            
            
        # if tr.loc[0.05] > 5:
                
        #     tjs = pd.DataFrame(data = [tx, ty, tr], index = ['x', 'y', 'r'])
        #     tjs = tjs.T
            # tjs['x'] = tjs['x'] / tjs['r'].max()
            # tjs['y'] = tjs['y'] / tjs['r'].max()
               
        
    # theta = - np.arctan2(tjs.loc[tjs['r'].idxmax()]['y'], tjs.loc[tjs['r'].idxmax()]['x']) 
            theta = - np.arctan2(vec['y'][0:0.15].iloc[ix1], vec['x'][0:0.15].iloc[ix1]) 
            akeep2.append(-theta)
            rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
            rotated = np.dot(rot, [vec['x'], vec['y']])
            
            rotvec = pd.DataFrame(index = ['x', 'y'], data = [rotated[0,:], rotated[1,:]] ).T
            rotframe = nap.TsdFrame(t = vec.index.values, d = rotvec.values, columns = ['x', 'y'])
            
            angkeep.append(np.arctan2(rotframe['y'].iloc[iy[-1]],rotframe['x'].iloc[iy[-1]]))
        # plt.plot(tjs['x'],tjs['y'],'o-')
       
            plt.plot(rotframe['x'][-0.25:0], rotframe['y'][-0.25:0], color = 'k', alpha = 0.5)
            plt.plot(rotframe['x'][0:0.15], rotframe['y'][0:0.15], color = 'silver', alpha = 0.5)
            # plt.plot(rotframe['x'][-0.25:0], rotframe['y'][-0.25:0], color = 'k', alpha = 0.2, zorder = 5, linewidth = 1)
            # plt.plot(rotframe['x'][0:0.15], rotframe['y'][0:0.15], color = 'silver', alpha = 0.5, zorder = 2)
     
#%% 

    angles = np.arctan2(vec['y'][0:0.15].iloc[ix], vec['x'][0:0.15].iloc[ix]) 
    ctrl_angles = np.arctan2(vec['y'][0:0.15].iloc[ix], vec['x'][0:0.15].iloc[ix]) 



#%%     
    angles = pd.DataFrame(data = [akeep2, angkeep], index = ['DU', 'UD']).T
    
    shu = pd.DataFrame()
    for i in range(1000):
        shuffang = pd.DataFrame(data = [angles['DU'].sample(frac=1).values, angles['UD'].sample(frac=1).values], index = ['DU', 'UD']).T
        shu = pd.concat([shu,shuffang])
    
    bins = np.linspace(-np.pi, np.pi, 20)
    plt.figure()
    plt.hist(np.minimum((2*np.pi - (shu['DU']-shu['UD'])),(shu['DU']-shu['UD']) ), bins, alpha = 0.5)
    
    plt.figure()
    plt.hist(np.minimum((2*np.pi - (angles['DU']-angles['UD'])),(angles['DU']-angles['UD']) ), bins)
    
    # plt.hist(angkeep)
    
#%% 
        ### Plotting 
    
    cmap = plt.colormaps['Greys']
    
    for k in range(6):
        print(k)
        traj = projection.restrict(nap.IntervalSet(start = du.loc[[k]]['start'], 
                                                   end = du.loc[[k]]['end']))
        traj.index = traj.index.values - (du.loc[[k]]['start'].values + 0.25)
        traj.index = [round (j,4) for j in traj.index.values]                               
                                   
        traj = traj.as_dataframe()    
        
    
        exl = len(traj)
        col = np.linspace(0,1, exl)
        dx = np.diff(traj['x'].values)
        dy = np.diff(traj['y'].values)
        
        tuning_curves = nap.compute_1d_tuning_curves(group=spikes[hd], 
                                                     feature=position['ang'],                                              
                                                     nb_bins=31, 
                                                     minmax=(0, 2*np.pi))
        pref_ang = []
         
        for i in tuning_curves.columns:
           pref_ang.append(tuning_curves.loc[:,i].idxmax())
    
        norm = plt.Normalize()        
        color = plt.cm.hsv(norm([i/(2*np.pi) for i in pref_ang]))
        
    
        
    
        plt.figure(figsize = (8,8))
        plt.subplot(121)
        plt.scatter(p_wake[:,0], p_wake[:,1], c = RGB[0:3000], zorder = 2)
        plt.scatter(p_sleep[:,0], p_sleep[:,1], c = 'silver') 
        
        for i in range(exl-1):
            print(traj['x'].iloc[i], traj['y'].iloc[i])
            plt.plot(traj['x'].iloc[i], traj['y'].iloc[i], 'o-', color = cmap(col)[i])
        
        for i in range(exl-2):
            plt.arrow(traj['x'].iloc[i], traj['y'].iloc[i],
                  dx[i], dy[i], color = cmap(col)[i],
                  head_width = 0.1, head_length = 0.1, linewidth = 4)
        
            
        # plt.subplot(142)
        # plt.plot(peth_radius['all'][k], color = 'silver') 
               
        # plt.subplot(143)
        # plt.plot(peth_angvel['all'][k], color = 'silver')
        
        plt.subplot(122)
        for i,n in enumerate(spikes[hd].keys()):
            plt.plot(spikes[hd][n].restrict(du.loc[[k]]).fillna(pref_ang[i]), '|',color = color[i])
            plt.axvline(du.loc[[k]]['start'][0] + 0.25, color = 'r')
    
        #%% 
        
        plt.scatter(p_wake[:,0], p_wake[:,1], c = RGB[0:3000], zorder = 2)
        plt.scatter(projection['x'].restrict(up_ep).values, projection['y'].restrict(up_ep).values, color = 'k')
        plt.scatter(projection['x'].restrict(down_ep).values, projection['y'].restrict(down_ep).values, color = 'silver')
        
       #%% 
     
#Shuffled Controls

        tokeep = []
        
        angkeep = []
        akeep2 = []
                
        plt.figure()    
        # plt.scatter(p_wake[:,0], p_wake[:,1], c = RGB[0:3000])
        
        for k in range(len(du)): #examples: #range(len(du)): #examples: #range(len(du)):  
       
            traj = projection.restrict(nap.IntervalSet(start = du.loc[[k]]['start'], 
                                                       end = du.loc[[k]]['end']))
            traj.index = traj.index.values - (du.loc[[k]]['start'].values + 0.25)
            traj.index = [round (j,4) for j in traj.index.values]                               
                                       
            traj = traj.as_dataframe()
            
            trand = traj.sample(frac = 1)
            trand.index = traj.index.values
            
                      
            vec = trand[-0.25:0.15]
            tr = nap.Tsd(t = vec. index. values, d = np.sqrt((vec['x'].values**2) + (vec['y'].values**2)))
            
            ix = np.where(tr[0:0.15].values > 10)[0]
            iy = np.where(tr[-0.25:0].values > 10)[0]
            
            if (len(ix) > 0) and (len(iy) > 0):
               
                ix1 = ix[0]
                
                tokeep.append(k)     
                     
                
                theta = - np.arctan2(vec['y'][0:0.15].iloc[ix1], vec['x'][0:0.15].iloc[ix1]) 
                akeep2.append(-theta)
                rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
                rotated = np.dot(rot, [vec['x'], vec['y']])
                
                rotvec = pd.DataFrame(index = ['x', 'y'], data = [rotated[0,:], rotated[1,:]] ).T
                rotframe = nap.TsdFrame(t = vec.index.values, d = rotvec.values, columns = ['x', 'y'])
                
                angkeep.append(np.arctan2(rotframe['y'].iloc[iy[-1]],rotframe['x'].iloc[iy[-1]]))
            
               
                plt.plot(rotframe['x'][-0.25:0], rotframe['y'][-0.25:0], color = 'k', alpha = 0.5)
                plt.plot(rotframe['x'][0:0.15], rotframe['y'][0:0.15], color = 'silver', alpha = 0.5)
                
        angles = pd.DataFrame(data = [akeep2, angkeep], index = ['DU', 'UD']).T
        bins = np.linspace(-np.pi, np.pi, 20)
        plt.figure()
        plt.hist(np.minimum((2*np.pi - abs(angles['DU']-angles['UD'])),abs(angles['DU']-angles['UD']) ), bins)
        