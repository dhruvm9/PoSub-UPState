#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 13:34:26 2023

@author: dhruv
"""
import numpy as np 
import pandas as pd 
import scipy.io
import pynapple as nap 
import os, sys
import time 
import matplotlib.pyplot as plt 
from matplotlib.colors import hsv_to_rgb
from scipy.stats import kendalltau, pearsonr, wilcoxon, mannwhitneyu
import seaborn as sns
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

#%% 

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
# datasets = np.loadtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')
# datasets = np.loadtxt(os.path.join(data_directory,'dataset_Ver_DM.list'), delimiter = '\n', dtype = str, comments = '#')

rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'

allPETH = pd.DataFrame()
peak_to_mean = []

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
        
#%% 

############################################################################################### 
    # COMPUTE EVENT CROSS CORRS
###############################################################################################  
      
## Peak firing       
    cc2 = nap.compute_eventcorrelogram(spikes, nap.Tsd(up_ep['start'].values), binsize = 0.005, windowsize = 0.255, ep = up_ep, norm = True)
    tmp = pd.DataFrame(cc2)
    tmp = tmp.rolling(window=8, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
    dd2 = tmp[0:0.155]  
    ee = dd2[pyr] 
   
    
    if len(ee.columns) > 0:
                    
        tokeep = []
        depths_keeping_ex = []
        peaks_keeping_ex = []
                                  
            
        for i in range(len(ee.columns)):
            a = np.where(ee.iloc[:,i] > 0.5)
            if len(a[0]) > 0:
              tokeep.append(ee.columns[i])  
              peaks_keeping_ex.append(ee.iloc[:,i].max())
              peak_to_mean.append(ee.iloc[:,i].max())
              
    allPETH = pd.concat([allPETH, ee[tokeep]], axis = 1)
    
 #%% Isomap

projection = Isomap(n_components = 2, n_neighbors = 50).fit_transform(allPETH.T.values)   

H = peak_to_mean/(max(peak_to_mean))
cmap = plt.cm.OrRd

plt.figure(figsize = (8,8))
plt.scatter(projection[:,0], projection[:,1], c = cmap(H))
plt.gca().set_box_aspect(1)
plt.xlabel('Isomap component 1')
plt.ylabel('Isomap component 2')

#%% Validation

r, p = pearsonr(projection[:,0],peak_to_mean)

plt.figure()
plt.scatter(projection[:,0],peak_to_mean, label = 'r = ' + str(round(r,2)))
plt.gca().set_box_aspect(1)
plt.xlabel('Isomap component 1')
plt.ylabel('Peak-maean rate ratio')
plt.legend(loc = 'upper right')
              
#%% Explained Variance 

# r = allPETH.corr()
# norm = np.matrix(peak_to_mean / np.linalg.norm(peak_to_mean))
# EV = np.dot(np.dot(norm,r),norm.T) / len(norm)

#%% PCA on isomap

# isomapdata = pd.DataFrame(columns = ['x','y'], data = projection)

# isomapdata['x'] = (isomapdata['x'] - isomapdata['x'].mean()) / isomapdata['x'].std()
# isomapdata['y'] = (isomapdata['y'] - isomapdata['y'].mean()) / isomapdata['y'].std()

# Z = isomapdata.T.dot(isomapdata)
# eigenvalues, eigenvectors = np.linalg.eig(Z)

# PCA = isomapdata.dot(eigenvectors)

# sum_eigenvalues = np.sum(eigenvalues)
# prop_var = [i/sum_eigenvalues for i in eigenvalues]