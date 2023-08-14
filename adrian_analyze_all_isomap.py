#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 18:05:36 2023

@author: dhruv
"""

import numpy as np 
import pandas as pd 
import scipy.io
import pynapple as nap 
import os, sys
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import circvar, mannwhitneyu, wilcoxon
import scipy.signal
from math import sin, cos
from matplotlib.colors import hsv_to_rgb
from random import uniform, sample

#%% 

data_directory = '/media/dhruv/LaCie1/PoSub-UPState/Data/###AllPoSub'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')
# datasets = np.genfromtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')

rwpath = '/media/dhruv/LaCie1/PoSub-UPState/Project/Data'

crosstimes = []
alldiffs = []
angvars = []
stats = []
UDang_all = [] 
radii = []

slope = []
cslope =[]
allwave_angles = []
allctrl_angles = []
radspeed = []

sess_vars = pd.DataFrame()
ctrl_vars = pd.DataFrame() 


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
            
#%% LOAD UP AND DOWN STATE
        
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

#%% 

    RGB = np.load(rawpath + '/' + s + '_RGB.npy')
    p_wake = np.load(rawpath + '/' + s + '_pwake.npy')
    
    projection = nap.TsdFrame(pd.read_pickle(rawpath + '/' + s + '_projection.pkl'))
    
    wake_radius = np.sqrt(p_wake[:,0]**2 + p_wake[:,1]**2)
    ringradius = np.mean(wake_radius)
    radii.append(ringradius)
    
    # plt.figure()
    # plt.scatter(p_wake[:,0], p_wake[:,1], color = RGB)

#%% 
   
    angdir = nap.Tsd(t = projection.index.values, d = np.arctan2(projection['y'].values, projection['x'].values))
    
    radius = nap.Tsd(t = projection.index.values, 
                        d = np.sqrt((projection['x'].values**2) 
                      + (projection['y'].values**2)))
    
    bins = np.arange(0, np.ceil(radius.max()), 1)
    
    counts, bins = np.histogram(radius,bins)
    # radius_index = scipy.signal.argrelmin(counts)
    
    # if len(radius_index) > 1:
    #     ringradius = int(bins[radius_index][0])
    # else: ringradius = 5
           
    # plt.figure()
    # plt.plot(counts)
    # plt.axvline(ringradius/3, color = 'k')
    
#%%
    
    tokeep = np.where((up_ep['end'] - up_ep['start']) > 0.5)
    longdu = nap.IntervalSet(start = up_ep.iloc[tokeep]['start'] - 0.25, end = up_ep.iloc[tokeep]['start'] + 0.5) 
    
    
    goodeps = []
    sess_angdiffs = []
    DUang = []
    UDang = []
    
    tmp2 = []
    ix2 = []
    
    ctmp2 = []
    cix2 = []
        
    sess_radspeed = []
    
    # plt.figure()    
    # plt.scatter(p_wake[:,0], p_wake[:,1], c = RGB)
    
    examples = [48, 580, 12, 264, 445] 
    
    for k in range(len(longdu)): #examples:
    
        traj = projection.restrict(nap.IntervalSet(start = longdu.loc[[k]]['start'], 
                                                    end = longdu.loc[[k]]['end']))
        traj.index = traj.index.values - (longdu.loc[[k]]['start'].values + 0.25)
        traj.index = [round (j,4) for j in traj.index.values]                               
                                   
        traj = traj.as_dataframe()
                  
        vec = traj[-0.25:0.5]
        vx = vec[0:0.15]
                
        tr = nap.Tsd(t = vec. index. values, d = np.sqrt((vec['x'].values**2) + (vec['y'].values**2)))
                      
        ix = np.where(tr[0:0.1].values > (ringradius/3))[0]
        iy = np.where(tr[-0.25:0].values > (ringradius/3))[0]
        
        if (len(ix) > 0) and (len(iy) > 0):
                       
            ix1 = ix[0]
            crosstimes.append(vx.index[ix1])
            winlength = len(vx[vx.index[ix1]:0.15])
            
            goodeps.append(k)     
            
            rx = nap.Tsd(t = vx['x'][vx.index[ix1]:0.15].index.values,
                d = np.sqrt(vx['x'][vx.index[ix1]:0.15]**2) + (vx['y'][vx.index[ix1]:0.15]**2))

            sess_radspeed.append(np.mean(np.gradient((rx))))
            radspeed.append(np.mean(np.gradient((rx))))
            
            
            theta = - np.arctan2(vx['y'].iloc[ix1], vx['x'].iloc[ix1]) 
            DUang.append(theta)
            rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
            rotated = np.dot(rot, [vec['x'], vec['y']])
            
            rotvec = pd.DataFrame(index = ['x', 'y'], data = [rotated[0,:], rotated[1,:]] ).T
            rotframe = nap.TsdFrame(t = vec.index.values, d = rotvec.values, columns = ['x', 'y'])
            
            UDang.append(np.arctan2(rotframe['y'].iloc[iy[-1]],rotframe['x'].iloc[iy[-1]]))
               
            # plt.plot(rotframe['x'][-0.25:0], rotframe['y'][-0.25:0], color = 'violet', zorder = 5)
            # plt.plot(rotframe['x'][0:0.15], rotframe['y'][0:0.15], color = 'mediumorchid')
           
            # plt.hist(UDang) 
            
            # wave_angles = np.arctan2(vx['y'][vx.index[ix1]:0.15], vx['x'][vx.index[ix1]:0.15])
            
            
                
            
#%% #Compute the angles for first and last third 
            
            thirds = int(np.ceil(len(vx['y'][vx.index[ix1]:0.15])/3))
            
            angle_first_part = np.arctan2(vx['y'][vx.index[ix1]:0.15].iloc[0:thirds], vx['x'][vx.index[ix1]:0.15].iloc[0:thirds]) 
            angle_last_part = np.arctan2(vx['y'][vx.index[ix1]:0.15].iloc[2*thirds-1:], vx['x'][vx.index[ix1]:0.15].iloc[2*thirds-1:]) 
            
            diffs = np.abs(np.mean(angle_last_part) - np.mean(angle_first_part))
            lindiff = np.minimum((2*np.pi - abs(diffs)), abs(diffs))
            sess_angdiffs.append(lindiff)
            alldiffs.append(lindiff)
            
                    
            wave_angles = pd.DataFrame(index = vec['y'][vx.index[ix1]:0.15].index.values,  
                                            data = np.arctan2(vec['y'][vx.index[ix1]:0.15], vec['x'][vx.index[ix1]:0.15])) 
            
            # ctrl_angles = pd.DataFrame(index = vec['y'][0.3:0.45].index.values,  
            #                                 data = np.arctan2(vec['y'][0.3:0.45], vec['x'][0.3:0.45])) 
            
            ctrl_angles = pd.DataFrame(index = vec['y'][0.15:0.3].index.values,  
                                            data = np.arctan2(vec['y'][0.15:0.3], vec['x'][0.15:0.3])) 
            
            
            allwave_angles.append(wave_angles)
            allctrl_angles.append(ctrl_angles)
          
            tmp = []
            ix = wave_angles.index.values[:-2]
            for i in range(len(wave_angles)-2):
                tmp.append(circvar(wave_angles.iloc[0:i+3]))
                  
            tmp2.append(pd.DataFrame(index = ix, data = tmp))
            
            ctmp = []
            cix = ctrl_angles.index.values[:-2]
            for i in range(len(ctrl_angles)-2):
                ctmp.append(circvar(ctrl_angles.iloc[0:i+3]))
                  
            ctmp2.append(pd.DataFrame(index = cix, data = ctmp))
            
                
    data_var = pd.concat(tmp2, axis = 1)
    data_var = data_var.sort_index()
    
    ctrl_var = pd.concat(ctmp2, axis = 1)
    ctrl_var = ctrl_var.sort_index()
    
    slope.append(np.mean(np.gradient(data_var.mean(axis = 1).loc[0.035:]))/np.diff(data_var.index.values)[0])
    # slope.append(np.mean(np.gradient(data_var.mean(axis = 1).loc[0.02:]))/np.diff(data_var.index.values)[0])
    cslope.append(np.mean(np.gradient(ctrl_var.mean(axis = 1)))/np.diff(data_var.index.values)[0])
   
    plt.figure()
    plt.title(s)
    plt.plot(data_var.index.values*1e3, data_var.mean(axis = 1), 'o-', color = 'lightcoral', label = 'Data')
    plt.plot(data_var.index.values*1e3, ctrl_var.mean(axis = 1), 'o-', color = 'rosybrown', label = 'Control')
    
    sess_vars = pd.concat([sess_vars, data_var.mean(axis=1)], axis = 1)
    ctrl_vars = pd.concat([ctrl_vars, ctrl_var.mean(axis=1)], axis = 1)
    
    # angvars.append(np.var(sess_angdiffs))
 
#%%     
    
varwave = []

for i in range(len(allwave_angles)):
    ix = allwave_angles[i].index.values[:-2]
    tmpall = []
    
    for j in range(len(allwave_angles[i])-2):
        tmpall.append(circvar(allwave_angles[i].iloc[0:j+3]))    
    
    varwave.append(pd.DataFrame(index = ix, data = tmpall))

varall = pd.concat(varwave, axis = 1)
varall = varall.sort_index()
    
               
ctrlwave = []

for i in range(len(allctrl_angles)):
    cix = allctrl_angles[i].index.values[:-2]
    ctmpall = []
    
    for j in range(len(allctrl_angles[i])-2):
        ctmpall.append(circvar(allctrl_angles[i].iloc[0:j+3]))
    
    ctrlwave.append(pd.DataFrame(index = cix, data = ctmpall))

ctrlall = pd.concat(ctrlwave, axis = 1)
ctrlall = ctrlall.sort_index()
        
                
    
#%% Simulated Data 
   
simdiffs = [] 
tmp2 = []
   
while len(simdiffs) <= (len(alldiffs)):     
     rvals = []
     thetavals = []

     r0 = 0 
     theta0 = 0 

     times = np.arange(0, 145, 10)
     rprime = np.mean(radii)/150 
         
     for i in times:
     
         rvals.append(r0)
         thetavals.append(theta0)
         
         thetaprime = np.random.normal(loc = 0, scale = 1)
                        
         r0 += rprime
         theta0 += thetaprime
         
                 
     firstpart = thetavals[0:int(len(thetavals)/3)] 
     lastpart = thetavals[2*int(len(thetavals)/3)-1:] 
     
     diffs = np.abs(np.mean(lastpart) - np.mean(firstpart))
     lindiff = np.minimum((2*np.pi - abs(diffs)), abs(diffs))
     
     if 0 <= lindiff <= np.pi:
     
         simdiffs.append(lindiff)
       
         tmp = []
     
         for i in range(len(thetavals)-2):
            
             tmp.append(circvar(thetavals[0:i+3]))
         tmp2.append(pd.DataFrame(tmp))
             
angvar = pd.concat(tmp2, axis = 1)

simdiffs2 = [] 
tmp2 = []
      
while len(simdiffs2) <= (len(alldiffs)):     
     rvals = []
     thetavals = []

     r0 = 0 
     theta0 = 0 

     times = np.arange(0, 145, 10)
     rprime = np.mean(radii)/150 
         
     for i in times:
     
         rvals.append(r0)
         thetavals.append(theta0)
         
         thetaprime = np.random.normal(loc = 0, scale = 0.5)
                        
         r0 += rprime
         theta0 += thetaprime
         
                 
     firstpart = thetavals[0:int(len(thetavals)/3)] 
     lastpart = thetavals[2*int(len(thetavals)/3)-1:] 
     
     diffs = np.abs(np.mean(lastpart) - np.mean(firstpart))
     lindiff = np.minimum((2*np.pi - abs(diffs)), abs(diffs))
     
     if 0 <= lindiff <= np.pi:
     
         simdiffs2.append(lindiff)
       
         tmp = []
     
         for i in range(len(thetavals)-2):
            
             tmp.append(circvar(thetavals[0:i+3]))
         tmp2.append(pd.DataFrame(tmp))
             
angvar2 = pd.concat(tmp2, axis = 1)
         
simdiffs3 = [] 
tmp2 = []
   
while len(simdiffs3) <= (len(alldiffs)):     
     rvals = []
     thetavals = []

     r0 = 0 
     theta0 = 0 

     times = np.arange(0, 145, 10)
     rprime = np.mean(radii)/150 
         
     for i in times:
     
         rvals.append(r0)
         thetavals.append(theta0)
         
         thetaprime = np.random.normal(loc = 0, scale = 0)
                        
         r0 += rprime
         theta0 += thetaprime
         
                 
     firstpart = thetavals[0:int(len(thetavals)/3)] 
     lastpart = thetavals[2*int(len(thetavals)/3)-1:] 
     
     diffs = np.abs(np.mean(lastpart) - np.mean(firstpart))
     lindiff = np.minimum((2*np.pi - abs(diffs)), abs(diffs))
     
     if 0 <= lindiff <= np.pi:
     
         simdiffs3.append(lindiff)
       
         tmp = []
     
         for i in range(len(thetavals)-2):
            
             tmp.append(circvar(thetavals[0:i+3]))
         tmp2.append(pd.DataFrame(tmp))
             
angvar3 = pd.concat(tmp2, axis = 1)
    
    
  
   
   
#%% 

bins = np.linspace(0, np.pi, 30)
xcenters = np.mean(np.vstack([bins[:-1], bins[1:]]), axis = 0)
 
counts,_ = np.histogram(alldiffs, bins)
simcounts,_ = np.histogram(simdiffs, bins) 
simcounts2,_ = np.histogram(simdiffs2, bins) 
simcounts3,_ = np.histogram(simdiffs3, bins) 
 
# plt.figure()
# plt.plot(xcenters, np.cumsum(counts/sum(counts)), color = 'rosybrown', label  = 'Data')
# plt.plot(xcenters, np.cumsum(simcounts3/sum(simcounts3)), linestyle = '--', color = 'k', label = 'SD = 0')
# plt.plot(xcenters, np.cumsum(simcounts2/sum(simcounts2)), linestyle = '--', color = 'dimgrey', label = 'SD = 0.5')
# plt.plot(xcenters, np.cumsum(simcounts/sum(simcounts)),linestyle = '--', color = 'darkgrey', label = 'SD = 1')
# plt.xlabel('Ang diff')
# plt.ylabel('Proportion of events')
# plt. legend(loc = 'lower right')                

plt.figure()
# plt.title(s)
# plt.plot(times[:-3],data_var.mean(axis = 1), 'o-', color = 'rosybrown', label = 'Data')
plt.plot(varall.index.values*1e3, varall.mean(axis = 1), 'o-', color = 'lightcoral', label = 'Data')

# err = varall.std(axis=1)
# plt.fill_between(varall.index.values*1e3, varall.mean(axis = 1)-err, varall.mean(axis = 1)+err)

plt.plot(varall.index.values*1e3, ctrlall.mean(axis = 1), 'o-', color = 'rosybrown', label = 'Control')

# err = ctrlall.std(axis=1)
# plt.fill_between(varall.index.values*1e3, ctrlall.mean(axis = 1)-err, ctrlall.mean(axis = 1)+err)


plt.plot(times[:-2],angvar3.mean(axis = 1), 'o-', color = 'k', label = 'SD = 0')
plt.plot(times[:-2],angvar2.mean(axis = 1), 'o-', color = 'dimgrey', label = 'SD = 0.5')
plt.plot(times[:-2],angvar.mean(axis = 1), 'o-', color = 'darkgrey', label = 'SD = 1')
plt.xlabel('Time (ms)')
plt.ylabel('Circular Variance')
plt. legend(loc = 'upper left')       
 
#%% 

DUtype = pd.DataFrame(['DU' for x in range(sess_vars.shape[1])])
ctrltype = pd.DataFrame(['Ctrl' for x in range(sess_vars.shape[1])])

slopedf = pd.DataFrame()
slopedf['type'] = pd.concat([DUtype, ctrltype])
slopedf['slope'] = pd.concat([pd.Series(slope), pd.Series(cslope)])

sns.set_style('white')
palette = ['lightcoral', 'rosybrown']
ax = sns.violinplot( x = slopedf['type'], y = slopedf['slope'] , data = slopedf, dodge=False,
                    palette = palette,cut = 2,
                      scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = slopedf['type'], y = slopedf['slope'] , data = slopedf, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.swarmplot(x = slopedf['type'], y = slopedf['slope'], data = slopedf, color = 'k', dodge=False, ax=ax)
# # sns.stripplot(x = b['type'], y=b['corr'], data=b, color = 'k', dodge=False, ax=ax)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Diffusion Constant (rad^2/s)')
ax.set_box_aspect(1)



#%% 

plt.figure()
plt.xlabel('Time (ms)')
plt.ylabel('Circular Variance')
plt.plot(np.array(sess_vars.index.values).astype(np.float64)*1e3, sess_vars.mean(axis = 1), color = 'lightcoral', label = 'Data')
err = sess_vars.sem(axis=1)
plt.fill_between(np.array(sess_vars.index.values).astype(np.float64)*1e3, sess_vars.mean(axis = 1)-err, sess_vars.mean(axis = 1)+err, color = 'lightcoral', alpha = 0.2)
plt.plot(np.array(sess_vars.index.values).astype(np.float64)*1e3, ctrl_vars.mean(axis = 1), color = 'rosybrown', label = 'Control')
err = ctrl_vars.sem(axis=1)
plt.fill_between(np.array(sess_vars.index.values).astype(np.float64)*1e3, ctrl_vars.mean(axis = 1)-err, ctrl_vars.mean(axis = 1)+err, color = 'rosybrown', alpha = 0.2)
plt.legend(loc = 'upper left')

