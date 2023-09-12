#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 13:54:03 2021

@author: dhruv
"""

#loading the dataset
import numpy as np 
import pandas as pd 
import scipy.io
from functions import *
from wrappers import *
import os, sys
import time 
import matplotlib.pyplot as plt 
from Wavelets import MyMorlet as Morlet
from scipy.stats import kendalltau, pearsonr, wilcoxon, mannwhitneyu
import seaborn as sns

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
# datasets = np.loadtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'

allcoefs_up = []
allcoefs_up_ex = []
allspeeds_up = []
allspeeds_up_ex = []
pvals = []
pvals_ex = []
N_units = []
N_ex = []
N_hd = [] 

range_DUonset = []
allDU = []

for s in datasets:
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    rawpath = os.path.join(rwpath,s)

    spikes, shank = loadSpikeData(path)
    N_units.append(len(spikes))
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

    N_ex.append(len(pyr))
    N_hd.append(len(hd))
# ############################################################################################### 
#     # LOAD UP AND DOWN STATE, NEW SWS AND NEW WAKE EPOCHS
# ###############################################################################################   

    
    # file = os.path.join(rawpath, name +'.evt.py.dow')
    # file = os.path.join(rawpath, name +'.evt.py.d3w')
    file = os.path.join(rawpath, name +'.evt.py.d1w')
    
    if os.path.exists(file):
        tmp = np.genfromtxt(file)[:,0]
        tmp = tmp.reshape(len(tmp)//2,2)/1000
        down_ep = nts.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
    
    # file = os.path.join(rawpath, name +'.evt.py.upp')
    # file = os.path.join(rawpath, name +'.evt.py.u3p')
    file = os.path.join(rawpath, name +'.evt.py.u1p')
    
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

############################################################################################### 
    # COMPUTE EVENT CROSS CORRS
###############################################################################################  
    binsize = 5
    nbins = 1000        
    neurons = list(spikes.keys())
    times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2        
    cc = pd.DataFrame(index = times, columns = neurons)
    tsd_up = up_ep.as_units('ms').start.values
  
#UP State    
    ep_U = nts.IntervalSet(start = up_ep.start[0], end = up_ep.end.values[-1])
                  
    rates = []
    
    sess_DU = []
               
    for i in neurons:
        # spk2 = spikes[i].restrict(ep_U).as_units('ms').index.values
        spk2 = spikes[i].restrict(up_ep).as_units('ms').index.values
        tmp = crossCorr(tsd_up, spk2, binsize, nbins)
        tmp = pd.DataFrame(tmp)
        tmp = tmp.rolling(window=8, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
        
        # fr = len(spk2)/ep_U.tot_length('s')
        fr = len(spk2)/up_ep.tot_length('s')
        rates.append(fr)
        cc[i] = tmp.values
        cc[i] = tmp.values/fr
                     
        dd = cc[0:150]
        # dd = cc[0:250]
               
    #Cell types 
    ee = dd[pyr]
    ff = dd[interneuron]
       
#######All cells 
#     if len(dd.columns) > 0:
#         indexplot = []
#         depths_keeping = []
        
#     for i in range(len(dd.columns)):
#         a = np.where(dd.iloc[:,i] > 0.5)
            
#         if len(a[0]) > 0:
#             depths_keeping.append(depth.flatten()[dd.columns[i]])
#             res = dd.iloc[:,i].index[a]
#             indexplot.append(res[0])
           
#     coef, p = kendalltau(indexplot,depths_keeping)
#     pvals.append(p)
#     allcoefs_up.append(coef)
    
# ####ALL CELLS SPEED

    
    

#     y_est = np.zeros(len(indexplot))
#     m, b = np.polyfit(indexplot, depths_keeping, 1)
#     allspeeds_up.append(m)
        
#     for i in range(len(indexplot)):
#         y_est[i] = m*indexplot[i]
        
        
#######Ex cells 
    if len(ee.columns) > 0:
                    
        indexplot_ex = []
        depths_keeping_ex = []
                
            
        for i in range(len(ee.columns)):
            a = np.where(ee.iloc[:,i] > 0.5)
            if len(a[0]) > 0:
                depths_keeping_ex.append(depth.flatten()[ee.columns[i]])
                res = ee.iloc[:,i].index[a]
                indexplot_ex.append(res[0])
                allDU.append(res[0])
            
        sess_DU.append(indexplot_ex)
        
        
        range_DUonset.append(np.std(sess_DU))
        
        #Latency v/s depth 
        coef_ex, p_ex = kendalltau(indexplot_ex,depths_keeping_ex)
        
        pvals_ex.append(p_ex)
        allcoefs_up_ex.append(coef_ex)
         
        
#######FS cells 
    # if len(ff.columns) > 0:
                    
    #     indexplot_fs = []
    #     depths_keeping_fs = []
         
                
            
    #     for i in range(len(ff.columns)):
    #         b = np.where(ff.iloc[:,i] > 0.5)
    #         if len(b[0]) > 0:
    #             depths_keeping_fs.append(depth.flatten()[ff.columns[i]])
    #             res = ee.iloc[:,i].index[b]
    #             indexplot_fs.append(res[0])
            
        
    #     #Latency v/s depth 
        # coef_fs, p_fs = kendalltau(indexplot_fs,depths_keeping_fs)
            
        # allcoefs_up_fs.append(coef_fs)
        
        ###SPEED 
        
        y_est_ex = np.zeros(len(depths_keeping_ex))
        m_ex, b_ex = np.polyfit(indexplot_ex, depths_keeping_ex, 1)
        allspeeds_up_ex.append(m_ex)
        
        for i in range(len(indexplot_ex)):
            y_est_ex[i] = m_ex*indexplot_ex[i]
            
        # ##FS cells speed
        
        # y_est_fs = np.zeros(len(indexplot_fs))
        # m_fs, b_fs = np.polyfit(indexplot_fs, depths_keeping_fs, 1)
        # allspeeds_fs.append((m_fs)/10)
        
        # for i in range(len(indexplot_fs)):
        #     y_est_fs[i] = m_fs*indexplot_fs[i]
            
    ###PLOTS
    plt.figure()
    plt.scatter(indexplot_ex, depths_keeping_ex, color = 'cornflowerblue')
    # plt.scatter(indexplot_fs,depths_keeping_fs, alpha = 0.8, color = 'indianred')
    plt.plot(indexplot_ex, y_est_ex + b_ex, color = 'cornflowerblue')
    # plt.plot(indexplot_fs, y_est_fs + b_fs, color = 'orange')
    # sns.regplot(x = depths_keeping_ex, y = indexplot_ex, ci = 95)
    plt.title('Bin where FR > 50% baseline rate_' + s)
    plt.ylabel('Depth from top of probe (um)')
    plt.yticks([0, -400, -800])
    plt.xlabel('Lag (ms)')
  
        
        # sys.exit()
    

# Out of loop 

z_up, p_up = wilcoxon(np.array(allcoefs_up)-0)
z_up_ex, p_up_ex = wilcoxon(np.array(allcoefs_up_ex)-0)
# z_up_fs, p_up_fs = wilcoxon(np.array(allcoefs_up_fs)-0)

# np.save('allspeeds_up_ex.npy', allspeeds_up_ex)
# np.save('allcoefs_up_ex.npy', allcoefs_up_ex)
speedmag_up = [x * -1 for x in allspeeds_up_ex]
speedmag_ex = [abs(x) for x in allspeeds_up_ex]
speedmag_up = [abs(x)  for x in allspeeds_up]

regs = pd.DataFrame()
regs['corr'] = allcoefs_up
regs['pval'] = pvals
regs['vel'] = speedmag_up
regs['corr_ex'] = allcoefs_up_ex
regs['pval_ex'] = pvals_ex
regs['spd_ex'] = speedmag_ex
regs['vel_ex'] = allspeeds_up_ex

plt.figure()
plt.hist([regs['corr_ex'][regs['pval_ex'] < 0.05], regs['corr_ex'][regs['pval_ex'] >= 0.05]], label = ['p < 0.05', 'p >= 0.05'], stacked = True, color = ['darkslategray','cadetblue'], linewidth = 2)
plt.legend(loc = 'upper right')
plt.xlabel('Tau value')
plt.yticks([0,1,2,3,4])
plt.ylabel('Number of sessions')

plt.figure()
plt.boxplot(regs['spd_ex'][regs['pval_ex'] < 0.05], positions = [0], showfliers= False, patch_artist=True,boxprops=dict(facecolor='darkslategray', color='darkslategray'),
            capprops=dict(color='darkslategray'),
            whiskerprops=dict(color='darkslategray'),
            medianprops=dict(color='white', linewidth = 2))
x1 = np.random.normal(0, 0.01, size=len(regs['vel_ex'][regs['pval_ex'] < 0.05]))
plt.plot(x1, regs['spd_ex'][regs['pval_ex'] < 0.05] , '.', color = 'k', fillstyle = 'none', markersize = 8, zorder =3)
plt.ylabel('Speed (mm/s)')
plt.xticks([])
plt.title('DU speed')


# plt.figure()
# plt.boxplot(allcoefs_up_ex, positions=[0], showfliers=False, patch_artist=True, boxprops=dict(facecolor='cornflowerblue', color='cornflowerblue'),
#             capprops=dict(color='cornflowerblue'),
#             whiskerprops=dict(color='cornflowerblue'),
#             medianprops=dict(color='white', linewidth = 2))
# plt.boxplot(allcoefs_up_fs, positions=[0.3], showfliers=False, patch_artist=True, boxprops=dict(facecolor='indianred', color='indianred'),
#             capprops=dict(color='indianred'),
#             whiskerprops=dict(color='indianred'),
#             medianprops=dict(color='white', linewidth = 2))

# x1 = np.random.normal(0, 0.01, size=len(allcoefs_up_ex))
# x2 = np.random.normal(0.3, 0.01, size=len(allcoefs_up_fs))
# plt.plot(x1, allcoefs_up_ex, '.', color = 'k', fillstyle = 'none', markersize = 8, zorder =3)
# plt.plot(x2, allcoefs_up_fs, '.', color = 'k', fillstyle = 'none', markersize = 8, zorder =3)
# plt.axhline(0, color = 'silver')
# plt.xticks([0, 0.3],['Ex', 'FS'])
# plt.title('Sequential activation of UP-state onset')
# plt.ylabel('Tau value')


# plt.figure()
# plt.boxplot(regs['vel_ex'], positions=[0], showfliers=False, patch_artist=True, boxprops=dict(facecolor='cornflowerblue', color='cornflowerblue'),
#             capprops=dict(color='cornflowerblue'),
#             whiskerprops=dict(color='cornflowerblue'),
#             medianprops=dict(color='white', linewidth = 2))
# plt.boxplot(allspeeds_dn_ex, positions=[0.3], showfliers=False, patch_artist=True, boxprops=dict(facecolor='indianred', color='indianred'),
#             capprops=dict(color='indianred'),
#             whiskerprops=dict(color='indianred'),
#             medianprops=dict(color='white', linewidth = 2))

# x1 = np.random.normal(0, 0.01, size=len(regs['vel_ex']))
# x2 = np.random.normal(0.3, 0.01, size=len(allspeeds_dn_ex))
# plt.plot(x1, regs['vel_ex'], '.', color = 'k', fillstyle = 'none', markersize = 8, zorder =3)
# plt.plot(x2, allspeeds_dn_ex, '.', color = 'k', fillstyle = 'none', markersize = 8, zorder =3)
# plt.axhline(0, color = 'silver')
# plt.xticks([0, 0.3],['DU', 'UD'])
# plt.title('Slope distribution')
# plt.ylabel('1/speed (s/mm)')

