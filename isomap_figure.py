#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 16:39:58 2023
@author: Dhruv
"""

import numpy as np 
import pandas as pd 
import scipy.io
import pynapple as nap 
import os, sys
import matplotlib
import matplotlib.pyplot as plt 
from matplotlib.colors import hsv_to_rgb
import scipy.signal
from sklearn.manifold import Isomap

#%% 

# data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
data_directory = '/media/adrien/LaCie/PoSub-UPState/Data/###AllPoSub'
# datasets = np.genfromtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')

rwpath = '/media/adrien/LaCie/PoSub-UPState/Project/Data'

alltruex = []
alltruey = []

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
    
#%%  Wake binning 
   
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
    wake_rate = wake_rate.sample(frac = 0.5).sort_index()

#%% Sleep binning 
    
    sleep_dt = 0.01 #0.015 #0.025 #0.015 
    sleep_binwidth = 0.05 #0.1 #0.03

    tokeep = np.where((up_ep['end'] - up_ep['start']) > 0.5)
    
    du = nap.IntervalSet(start = up_ep.iloc[tokeep]['start'] - 0.25, end = up_ep.iloc[tokeep]['start'] + 0.25) 
    longdu = nap.IntervalSet(start = up_ep.iloc[tokeep]['start'] - 0.25, end = up_ep.iloc[tokeep]['start'] + 0.25) 
        
    num_overlapping_bins = int(sleep_binwidth/sleep_dt)    
        
    sleep_count = spikes[hd].count(sleep_dt,longdu) 
    sleep_count = sleep_count.as_dataframe()
    sleep_rate = np.sqrt(sleep_count/sleep_dt)
    sleep_rate = sleep_rate.rolling(window = num_overlapping_bins, win_type = 'gaussian', center = True, min_periods = 1, axis = 0).mean(std = 3)
    
    fit_rate = nap.TsdFrame(sleep_rate).restrict(du)
       
    rate = np.vstack([wake_rate.loc[:, wake_rate.columns != 'ang'].values, fit_rate.as_dataframe().sample(frac = 0.01).values]) #Take 1000 random values 
               
    fit = Isomap(n_components = 2, n_neighbors = 200).fit(rate) 
    p_wake = fit.transform(wake_rate.loc[:, wake_rate.columns != 'ang'])    
    p_sleep = fit.transform(sleep_rate)
    

    H = wake_rate['ang'].values/(2*np.pi)
    HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T
    RGB = hsv_to_rgb(HSV)    
    
    truex = p_sleep[np.where(sleep_rate.sum(axis=1)==0)][0][0]
    truey = p_sleep[np.where(sleep_rate.sum(axis=1)==0)][0][1]
    
    alltruex.append(truex)
    alltruey.append(truey)
    
    p_wake = p_wake - [truex, truey]
    p_sleep = p_sleep - [truex, truey]
                    
    projection = nap.TsdFrame(t = sleep_rate.index.values, d = p_sleep, columns = ['x', 'y'])

#%% 

    wake_radius = np.sqrt((p_wake[:,0]**2) 
                       + (p_wake[:,1]**2))
    
    ringradius = np.mean(wake_radius)

    # cmap = plt.colormaps['Greys']
    cmap = plt.colormaps['copper']
    
    examples = [470, 741, 991]  
    # [ 470KEEP, 741KEEP, 950 ]  
    # goodeps = []
    
    
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
    plt.scatter(p_wake[:,0], p_wake[:,1], c = RGB, zorder = 2)
    # plt.scatter(p_sleep[:,0], p_sleep[:,1], c = 'silver') 
    
    for k in examples: #goodeps[91:100]: #range(len(du)): #examples:
        traj = projection.restrict(nap.IntervalSet(start = du.loc[[k]]['start'], 
                                                   end = du.loc[[k]]['end']))
        traj.index = traj.index.values - (du.loc[[k]]['start'].values + 0.25)
        traj.index = [round (j,4) for j in traj.index.values]                               
                                   
        traj = traj.as_dataframe()    
        
        radius = nap.Tsd(t = traj.index.values, 
                           d = np.sqrt((traj['x'].values**2) 
                          + (traj['y'].values**2)))
        
        ix = np.where(radius[0:0.1].values > ringradius/3)[0]
        iy = np.where(radius[-0.25:0].values > ringradius)[0]
        
        if (len(ix) > 0)and (len(iy) > 0):
            # goodeps.append(k)
            
            
    
            exl = len(traj[0:0.15])
            col = np.linspace(0,1, exl)
            dx = np.diff(traj['x'][0:0.15].values)
            dy = np.diff(traj['y'][0:0.15].values)
            
            # plt.figure(figsize = (8,8))
            # plt.subplot(121)
            # plt.scatter(p_wake[:,0], p_wake[:,1], c = RGB, zorder = 2)
            # plt.scatter(p_sleep[:,0], p_sleep[:,1], c = 'silver') 
            
            
            for i in range(exl-1):
                plt.plot(traj['x'][0:0.15].iloc[i], traj['y'][0:0.15].iloc[i], 'o-', color = cmap(col)[i], zorder = 5)
            
            for i in range(exl-2):
                plt.arrow(traj['x'][0:0.15].iloc[i], traj['y'][0:0.15].iloc[i],
                      dx[i], dy[i], color = cmap(col)[i],
                      head_width = 0.1, head_length = 0.1, linewidth = 4, zorder = 5)
                
            # plt.subplot(122)
            # for i,n in enumerate(spikes[hd].keys()):
            #     # plt.plot(spikes[hd][n].restrict(du.loc[[k]]).fillna(pref_ang[i]), '|',color = color[i])
            #     plt.plot(spikes[n].restrict(du.loc[[k]]).fillna(pref_ang[i]), '|',color = 'k')
            #     plt.axvline(du.loc[[k]]['start'][0] + 0.25, color = 'r')
         


#%% Circular colorbar 

fg = plt.figure(figsize=(8,8))
ax = fg.add_axes([0.1,0.1,0.8,0.8], projection='polar')

# Define colormap normalization for 0 to 2*pi
norm = matplotlib.colors.Normalize(0, 2*np.pi) 

# Plot a color mesh on the polar plot
# with the color set by the angle

n = 200  #the number of secants for the mesh
t = np.linspace(0,2*np.pi,n)   #theta values
r = np.linspace(.8,1,2)        #radius values change 0.6 to 0 for full circle
rg, tg = np.meshgrid(r,t)      #create a r,theta meshgrid
c = tg                         #define color values as theta value
im = ax.pcolormesh(t, r, c.T,norm=norm, cmap = plt.cm.hsv)  #plot the colormesh on axis with colormap
ax.set_yticklabels([])                   #turn of radial tick labels (yticks)
ax.tick_params(pad=15,labelsize=24)      #cosmetic changes to tick labels
ax.spines['polar'].set_visible(False)    #turn off the axis spine.
    
    
        