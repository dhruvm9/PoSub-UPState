#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 15:12:36 2024

@author: dhruv
"""

import numpy as np 
import pandas as pd 
import scipy.io
import nwbmatic as ntm
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

    RGB_D = np.load(rawpath + '/' + s + '_RGB_D.npy')
    RGB_V = np.load(rawpath + '/' + s + '_RGB_V.npy')
    
    p_wake_D = np.load(rawpath + '/' + s + '_pwake_D.npy')
    p_wake_V = np.load(rawpath + '/' + s + '_pwake_V.npy')
    
    projection_D = nap.TsdFrame(pd.read_pickle(rawpath + '/' + s + '_projection_D.pkl'))
    projection_V = nap.TsdFrame(pd.read_pickle(rawpath + '/' + s + '_projection_V.pkl'))
    
#%% 
    
    plt.figure()
    plt.suptitle(s)
    plt.subplot(121)
    plt.title('Dorsal')
    plt.scatter(p_wake_D[:,0], p_wake_D[:,1], color = RGB_D)
    plt.subplot(122)
    plt.title('Ventral')
    plt.scatter(p_wake_V[:,0], p_wake_V[:,1], color = RGB_V)
    
#%%
    # wake_radius = np.sqrt(p_wake[:,0]**2 + p_wake[:,1]**2)
    # ringradius = np.mean(wake_radius)
    # radii.append(ringradius)
    
