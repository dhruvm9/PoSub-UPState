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
from itertools import combinations
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
uponset_hdc = []
layers_all = []
UD_onset = [] 
alldepths = []

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
    
    cc2 = nap.compute_eventcorrelogram(spikes, nap.Tsd(down_ep['start'].values), binsize = 0.005, windowsize = 0.255, ep = up_ep, norm = True)    
    dd2 = cc2[-0.25:0.25]
    tmp2 = pd.DataFrame(dd2)
    tmp2 = tmp2.rolling(window=8, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
    ee2 = dd2[pyr]

    if len(ee2.columns) > 0:
        tmp = ee2.loc[0.005:] > 0.5
        
        tokeep = tmp.columns[tmp.sum(0) > 0]
        ends = np.array([tmp.index[np.where(tmp[i])[0][0]] for i in tokeep])
        es = pd.Series(index = tokeep, data = ends)
        
        tmp2 = ee2.loc[-0.1:-0.005] > 0.5
    
        tokeep2 = tmp2.columns[tmp2.sum(0) > 0]
        start = np.array([tmp2.index[np.where(tmp2[i])[0][-1]] for i in tokeep2])
        st = pd.Series(index = tokeep2, data = start)
            
        ix = np.intersect1d(tokeep,tokeep2)
        ix = [int(i) for i in ix]
        stk = st[ix]
        
    for i in stk.index.values:
        UD_onset.append(stk[i])
    
#%% 

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
              alldepths.append(depth.flatten()[ee.columns[i]])
              peak_to_mean.append(ee.iloc[:,i].max())
              layers_all.append((np.float64(layer[i])))
              res = ee.iloc[:,i].index[a]
              uponset_hdc.append(res[0])
              
    allPETH = pd.concat([allPETH, ee[tokeep]], axis = 1)

#%% Isomap 

projection = Isomap(n_components = 2, n_neighbors = 50).fit_transform(allPETH.T.values)   

H = peak_to_mean/(max(peak_to_mean))

# H = layers_all/(max(layers_all))

# lower, upper = 2, 10
# norm_onset = (upper - lower)*((uponset_hdc - min(uponset_hdc)) / (max(uponset_hdc) - min(uponset_hdc))) + lower

norm_onset = uponset_hdc/max(uponset_hdc)
cmap = plt.cm.OrRd

plt.figure(figsize = (8,8))
plt.scatter(projection[:,0], projection[:,1], c = cmap(H)) #s = [150*i + 3 for i in norm_onset]) #[20*20**i for i in norm_onset] )
plt.gca().set_box_aspect(1)
plt.xlabel('Isomap component 1')
plt.ylabel('Isomap component 2')

#%% Xcorr sorted by distance 

# a = nap.compute_crosscorrelogram(spikes, binsize = 0.005, windowsize = 1, ep = up_ep)
    
 #%% Isomap on AD and PoSub (FIRST RUN AD PETH, THEN HDC PETH)

# projection = Isomap(n_components = 2, n_neighbors = 50).fit_transform(allPETH.T.values)   

# H_AD = peak_to_mean[0:83]/(max(peak_to_mean[0:83]))
# H_HDC = peak_to_mean[84:]/(max(peak_to_mean[84:]))

# # cmap_ad = plt.cm.OrRd
# # cmap_hdc = plt.cm.inferno

# cmap_ad = plt.cm.Greys
# cmap_hdc = plt.cm.Reds


# plt.figure(figsize = (8,8))
# plt.scatter(projection[:,0][0:83], projection[:,1][0:83], c = cmap_ad(H_AD))
# plt.scatter(projection[:,0][84:], projection[:,1][84:], c = cmap_hdc(H_HDC))
# plt.gca().set_box_aspect(1)
# plt.xlabel('Isomap component 1')
# plt.ylabel('Isomap component 2')

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

#%% Isomap points and PETH traces 
    
allPETH.columns = range(allPETH.columns.size)

summ = {}
summ['peth'] = allPETH
summ['p1'] = projection[:,0]
summ['p2'] = projection[:,1]

# np.argmax(summ['p2'])
# np.argmin(summ['p2'])
# a = np.where(np.logical_and(summ['p1']>=-4, summ['p1']<=-2))
# b = np.where(np.logical_and(summ['p2']>=-2, summ['p2']<=0))
# np.intersect1d(a,b)

examples = [2, 523, 528, 1081, 895,828,1061,226,208,379,1018, 378]

for i in examples:

    plt.figure()
    plt.plot(summ['peth'][i])
    plt.axhline(y = 1, linestyle = '--', color = 'silver')
    plt.plot(uponset_hdc[i],0.5)

    
#%% 

#bins = np.arange(0,0.15,0.005);plt.hist(uponset_adn, bins, density = True); plt.hist(uponset_hdc, bins, alpha = 0.8, density = True)

#%% 

pairs = list(combinations(summ['peth'].columns,2))

F_pmrr = []
F_uponset = []

depth_diff = []

for i,p in enumerate(pairs):
    diff_pmrr = peak_to_mean[p[0]] - peak_to_mean[p[1]]
    diff_uponset = uponset_hdc[p[0]] - uponset_hdc[p[1]]
           
    dx = summ['p1'][p[0]] - summ['p1'][p[1]]
    dy = summ['p2'][p[0]] - summ['p2'][p[1]]
    
    depth_diff.append(alldepths[p[0]] - alldepths[p[1]])    
    
    F_pmrr.append([diff_pmrr/dx, diff_pmrr/dy])
    F_uponset.append([diff_uponset/dx, diff_uponset/dy])
    
mean_pmrr = np.mean(F_pmrr, axis = 0)
mean_uponset = np.mean(F_uponset, axis = 0)

origin = np.array([[0, 0],[0, 0]])
plt.figure()
plt.xlim(0,0.05)
plt.ylim(0,0.05)
plt.quiver(origin[0], origin[1],  mean_pmrr[0] ,  mean_pmrr[1], angles = 'xy', scale_units = 'xy', scale = 1)


#%% 


    
    