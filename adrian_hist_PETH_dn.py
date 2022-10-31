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
import neuroseries as nts 
import time 
import matplotlib.pyplot as plt 
from Wavelets import MyMorlet as Morlet
from scipy.stats import kendalltau, pearsonr, wilcoxon, mannwhitneyu
import seaborn as sns 

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
# datasets = np.loadtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
datasets = np.loadtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'

allcoefs_dn = []
allcoefs_dn_ex = []
durcoeffs_D = []
durcoeffs_V = []
allspeeds_dn_ex = []
p_dn_ex = []
allspeeds_fs = []
allspeeds_dn = []

DD = []
DV = []
ratios = []

n_pyr = []
n_int = []

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

    n_pyr.append(len(pyr))
    n_int.append(len(interneuron))
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

############################################################################################### 
    # COMPUTE EVENT CROSS CORRS
###############################################################################################  
                 
    binsize = 5
    nbins = 1000        
    neurons = list(spikes.keys())
    times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2        
    cc = pd.DataFrame(index = times, columns = neurons)
    tsd_dn = down_ep.as_units('ms').start.values
  
#DOWN State    
    ep_D = nts.IntervalSet(start = down_ep.start[0], end = down_ep.end.values[-1])
               
    rates = []
    reslist = []
            
    for i in neurons:
        # spk2 = spikes[i].restrict(ep_D).as_units('ms').index.values
        spk2 = spikes[i].restrict(up_ep).as_units('ms').index.values
        tmp = crossCorr(tsd_dn, spk2, binsize, nbins)
        tmp = pd.DataFrame(tmp)
        tmp = tmp.rolling(window=4, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
        
        # fr = len(spk2)/ep_D.tot_length('s')
        fr = len(spk2)/up_ep.tot_length('s')
        rates.append(fr)
        cc[i] = tmp.values
        cc[i] = tmp.values/fr
       
        dd = cc[-250:250]
        
    #Cell types 
    ee = dd[pyr]
    ff = dd[interneuron]
   
#######All cells 
    # if len(dd.columns) > 0:
    #     tmp = dd.loc[5:] > 0.5
        
    #     tokeep = tmp.columns[tmp.sum(0) > 0]
    #     ends = np.array([tmp.index[np.where(tmp[i])[0][0]] for i in tokeep])
    #     es = pd.Series(index = tokeep, data = ends)
        
    #     tmp2 = dd.loc[-100:-5] > 0.5
    
    #     tokeep2 = tmp2.columns[tmp2.sum(0) > 0]
    #     start = np.array([tmp2.index[np.where(tmp2[i])[0][-1]] for i in tokeep2])
    #     st = pd.Series(index = tokeep2, data = start)
            
    #     ix = np.intersect1d(tokeep,tokeep2)
    #     ix = [int(i) for i in ix]
        
        
    #     depths_keeping = depth[ix]
    #     stk = st[ix]
        
    # coef, p = kendalltau(stk,depths_keeping)
    # allcoefs_dn.append(coef)
    # pvals.append(p)
           
####ALL CELLS SPEED
    # y_est = np.zeros(len(stk))
    # m, b = np.polyfit(stk, depths_keeping, 1)
    # allspeeds_dn.append(m[0]/10)
        
    # for i in range(len(stk)):
    #     y_est[i] = m*stk.values[i]
   
   
    
#####EX cells    
    if len(ee.columns) > 0: 
        
    #     mn = np.mean(ee[-250:-100])
        
    #     tmp = dd.loc[5:] > 0.2
        
        
        tmp_ex = ee.loc[5:] > 0.5
        
        tokeep_ex = tmp_ex.columns[tmp_ex.sum(0) > 0]
        ends_ex = np.array([tmp_ex.index[np.where(tmp_ex[i])[0][0]] for i in tokeep_ex])
        es_ex = pd.Series(index = tokeep_ex, data = ends_ex)
        
        tmp2_ex = dd.loc[-250:-5] > 0.2
        # tmp2_ex = ee.loc[-100:-5] > 0.5
    
        tokeep2_ex = tmp2_ex.columns[tmp2_ex.sum(0) > 0]
        start_ex = np.array([tmp2_ex.index[np.where(tmp2_ex[i])[0][-1]] for i in tokeep2_ex])
        st_ex = pd.Series(index = tokeep2_ex, data = start_ex)
            
        ix_ex = np.intersect1d(tokeep_ex,tokeep2_ex)
        ix_ex = [int(i) for i in ix_ex]
        
        
        depths_keeping_ex = depth[ix_ex]
        # dur_ex = np.zeros(len(ix_ex))
         
        
        # for i,n in enumerate(ix_ex):
        #         dur_ex[i] = es_ex[ix_ex][n] - st_ex[ix_ex][n]
     
    #Plot for threshold crossing           
    coef_ex, p_ex = kendalltau(st_ex[ix_ex],depths_keeping_ex)
    allcoefs_dn_ex.append(coef_ex)
    p_dn_ex.append(p_ex)
        
    stk_ex = st_ex[ix_ex]
    
    # data_ex = pd.DataFrame()
    # data_ex['depth'] = depths_keeping_ex.flatten()
    # data_ex['dur'] = dur_ex
    # data_ex['level'] = pd.cut(data_ex['depth'],3, precision=0, labels=[2,1,0]) #0 is dorsal, 2 is ventral
    
        
#####FS cells    
    # if len(ff.columns) > 0: 
    
    #     # mn2 = np.mean(ff[-250:-100])
    #     # tmp = dd.loc[5:] > 0.2
    #     tmp_fs = ff.loc[5:] > 0.5
        
    #     tokeep_fs = tmp_fs.columns[tmp_fs.sum(0) > 0]
    #     ends_fs = np.array([tmp_fs.index[np.where(tmp_fs[i])[0][0]] for i in tokeep_fs])
    #     es_fs = pd.Series(index = tokeep_fs, data = ends_fs)
        
    #     # tmp2 = dd.loc[-250:-5] > 0.2
    #     tmp2_fs = ff.loc[-100:-5] > 0.5
        
    #     tokeep2_fs = tmp2_fs.columns[tmp2_fs.sum(0) > 0]
    #     start_fs = np.array([tmp2_fs.index[np.where(tmp2_fs[i])[0][-1]] for i in tokeep2_fs])
    #     st_fs = pd.Series(index = tokeep2_fs, data = start_fs)
            
    #     ix_fs = np.intersect1d(tokeep_fs,tokeep2_fs)
    #     ix_fs = [int(i) for i in ix_fs]
        
        
    #     depths_keeping_fs = depth[ix_fs]
    #     dur_fs = np.zeros(len(ix_fs))
         
        
    #     for i,n in enumerate(ix_fs):
    #             dur_fs[i] = es_fs[ix_fs][n] - st_fs[ix_fs][n]
     
    # #Plot for threshold crossing           
    # coef_fs, p_fs = kendalltau(st_fs[ix_fs],depths_keeping_fs)
    # allcoefs_dn_fs.append(coef_fs)
    
    # stk_fs = st_fs[ix_fs]
 
####SPEED COMPUTATION 

    y_est_ex = np.zeros(len(stk_ex.values))
    m_ex, b_ex = np.polyfit(stk_ex.values, depths_keeping_ex.flatten(), 1)
    allspeeds_dn_ex.append(m_ex)
        
    for i in range(len(stk_ex)):
        y_est_ex[i] = m_ex*stk_ex.values[i]
    

    # y_est_fs = np.zeros(len(stk_fs))
    # m_fs, b_fs = np.polyfit(stk_fs, depths_keeping_fs, 1)
    # allspeeds_fs.append((m_fs[0])/10)
        
    # for i in range(len(stk_fs)):
    #     y_est_fs[i] = m_fs*stk_fs.values[i]


####PLOT        
    plt.figure()
    plt.scatter(stk_ex.values, depths_keeping_ex.flatten(), color = 'cornflowerblue', alpha = 0.8, label = 'R(ex) = ' + str(round(coef_ex,4)))
    # plt.scatter(stk_fs, depths_keeping_fs, alpha = 0.8, color = 'indianred')
    plt.plot(stk_ex, y_est_ex + b_ex, color = 'cornflowerblue')
    # # plt.plot(stk_fs, y_est_fs + b_fs, color = 'orange')
    # sns.regplot(x = depths_keeping_ex.flatten(), y = stk_ex.values, ci = 95)
    plt.title('Last bin before DOWN where FR > 50% baseline rate_' + s)
    plt.xlabel('Depth from top of probe (um)')
    plt.yticks([0, -400, -800])
    plt.xlabel('Lag (ms)')
    plt.legend(loc = 'upper right')
            
#####Plot for duration
    # dur_ventral = data_ex[data_ex['level'] ==2].dur.values
    # dur_dorsal = data_ex[data_ex['level'] ==0].dur.values
    
    # DD.append(dur_dorsal)
    # DV.append(dur_ventral)
    
    # ratio = np.mean(dur_ventral) / np.mean(dur_dorsal)
    # ratios.append(ratio)    

    # coef2_V, p2_V = kendalltau(dur_ventral, data_ex[data_ex['level'] ==2].depth.values)
    # coef2_D, p2_D = kendalltau(dur_dorsal, data_ex[data_ex['level'] ==0].depth.values)
            
    # durcoeffs_D.append(coef2_D)
    # durcoeffs_V.append(coef2_V)
 
    
    # plt.figure()
    # plt.title('DOWN duration_' + s)
    # # bins = np.linspace(min(min(dur_dorsal),min(dur_ventral)),(max(max(dur_dorsal),max(dur_ventral))),20)
    # plt.boxplot(dur_dorsal, positions = [0])
    # plt.boxplot(dur_ventral, positions = [0.3])
    # plt.xticks([0, 0.3],['Dorsal', 'Ventral'])
    # plt.ylabel('DOWN duration (ms)')
    # x1 = np.random.normal(0, 0.01, size=len(dur_dorsal))
    # x2 = np.random.normal(0.3, 0.01, size=len(dur_ventral))
    # plt.plot(x1, dur_dorsal, '.', color = 'k', fillstyle = 'none', markersize = 8, zorder =3)
    # plt.plot(x2, dur_ventral, '.', color = 'k', fillstyle = 'none', markersize = 8, zorder =3)
    
    
    # plt.figure()
    # plt.scatter(dur_dorsal,data_ex[data_ex['level'] ==0].depth.values, c= 'orange', label = 'R(dorsal) = ' + str(round(coef2_D,4)))
    # plt.scatter(dur_ventral,data_ex[data_ex['level'] ==2].depth.values, c= 'black', label = 'R(ventral) = ' + str(round(coef2_V,4)))
    # plt.title('Duration of < 20% baseline firing v/s depth_' + s)
    # plt.ylabel('Depth from top of probe (um)')
    # plt.yticks([0, -400, -800])
    # plt.xlabel('Duration of cell firing < 20% (ms)')
    # plt.legend(loc = 'upper right')
          
    # sys.exit()
    
#Out of loop 

# z_dn, p_dn = wilcoxon(np.array(allcoefs_dn)-0)
# z_dn_ex, p_dn_ex = wilcoxon(np.array(allcoefs_dn_ex)-0)
# z_dn_fs, p_dn_fs = wilcoxon(np.array(allcoefs_dn_fs)-0)

# np.save('allcoefs_dn_ex.npy', allcoefs_dn_ex)

regs = pd.DataFrame()
regs['corr'] = allcoefs_dn

# ############################################################################################### 
#     # CUMULATIVE LAG v/s DEPTH PLOT (Run adrian_hist_PETH_up before running this segment)
# ###############################################################################################   

a = pd.DataFrame()
a['allcoefs_up_ex'] = allcoefs_up_ex
a['pval_up_ex'] = pvals_ex
a['allcoefs_dn_ex'] = allcoefs_dn_ex
a['pval_dn_ex'] = p_dn_ex

plt.figure()
plt.boxplot(allcoefs_up_ex, positions=[0], showfliers=False, patch_artist=True, boxprops=dict(facecolor='royalblue', color='royalblue'),
              capprops=dict(color='royalblue'),
              whiskerprops=dict(color='royalblue'),
              medianprops=dict(color='white', linewidth = 2))
plt.boxplot(allcoefs_dn_ex, positions=[0.3], showfliers=False, patch_artist=True, boxprops=dict(facecolor='lightsteelblue', color='lightsteelblue'),
              capprops=dict(color='lightsteelblue'),
              whiskerprops=dict(color='lightsteelblue'),
              medianprops=dict(color='white', linewidth = 2))

x1 = np.random.normal(0, 0.01, size=len(a['allcoefs_up_ex'][a['pval_up_ex'] >= 0.05]))
x2 = np.random.normal(0.3, 0.01, size=len(a['allcoefs_dn_ex'][a['pval_dn_ex'] >= 0.05]))
x3 = np.random.normal(0, 0.01, size=len(a['allcoefs_up_ex'][a['pval_up_ex'] < 0.05]))
x4 = np.random.normal(0.3, 0.01, size=len(a['allcoefs_dn_ex'][a['pval_dn_ex'] < 0.05]))

plt.plot(x1, a['allcoefs_up_ex'][a['pval_up_ex'] >= 0.05], '.', color = 'k', fillstyle = 'none', markersize = 6, zorder =3)
plt.plot(x2, a['allcoefs_dn_ex'][a['pval_dn_ex'] >= 0.05], '.', color = 'k', fillstyle = 'none', markersize = 6, zorder =3, label = 'p >= 0.05')
plt.plot(x3, a['allcoefs_up_ex'][a['pval_up_ex'] < 0.05], 'x', color = 'k', fillstyle = 'none', markersize = 6, zorder =3, label = 'p < 0.05')
plt.plot(x4, a['allcoefs_dn_ex'][a['pval_dn_ex'] < 0.05], 'x', color = 'k', fillstyle = 'none', markersize = 6, zorder =3)
plt.axhline(0, color = 'silver')
plt.xticks([0, 0.3],['DOWN-UP', 'UP-DOWN'])
plt.title('Sequential activation of Ex cells')
plt.ylabel('Lag v/s depth (R)')
plt.legend(loc = 'upper right')

# ############################################################################################### 


# z_dn_D, p_dn_D = wilcoxon(np.array(durcoeffs_D)-0)
# z_dn_V, p_dn_V = wilcoxon(np.array(durcoeffs_V)-0)
# # bins = np.linspace(min(min(durcoeffs_D), min(durcoeffs_V)),max(max(durcoeffs_D), max(durcoeffs_V)),20)

# plt.figure()
# # plt.hist(durcoeffs_D,bins, color = 'orange',alpha = 0.5, linewidth = 2, label = 'p-value (dorsal) =' + str(round(p_dn_D,4)))
# # plt.axvline(np.mean(durcoeffs_ex),color = 'blue', linestyle='dashed', linewidth = 1)
# # plt.hist(durcoeffs_V,bins, color = 'k',alpha = 0.5, linewidth = 2, label = 'p-value (ventral) =' + str(round(p_dn_V,4)))
# plt.title('DOWN-duration as a function of depth (ex cells)')
# # plt.xlabel('Tau value') 
# # plt.ylabel('Number of sessions')
# # plt.legend(loc = 'upper right')  

# plt.boxplot(durcoeffs_D, positions = [0], showfliers=False,patch_artist=True, boxprops=dict(facecolor='royalblue', color='royalblue'),
#             capprops=dict(color='royalblue'),
#             whiskerprops=dict(color='royalblue'),
#             medianprops=dict(color='white', linewidth = 2))

# plt.boxplot(durcoeffs_V, positions = [0.3], showfliers=False,patch_artist=True, boxprops=dict(facecolor='lightsteelblue', color='lightsteelblue'),
#             capprops=dict(color='lightsteelblue'),
#             whiskerprops=dict(color='lightsteelblue'),
#             medianprops=dict(color='white', linewidth = 2))
# plt.xticks([0, 0.3],['Dorsal', 'Ventral'])
# plt.ylabel('Tau value')
# x1 = np.random.normal(0, 0.01, size=len((durcoeffs_D)))
# x2 = np.random.normal(0.3, 0.01, size=len((durcoeffs_V)))
# plt.axhline(0, color = 'silver')
# plt.plot(x1, durcoeffs_D  , '.', color = 'k', fillstyle = 'none', markersize = 8, zorder =3)
# plt.plot(x2, durcoeffs_V , '.', color = 'k', fillstyle = 'none', markersize = 8, zorder =3)



# z_spd_ex, p_spd_ex = wilcoxon(np.array(allspeeds_ex)-0)
# z_spd_fs, p_spd_fs = wilcoxon(np.array(allspeeds_fs)-0)
# t, pvalue = mannwhitneyu(allspeeds_ex, allspeeds_fs)

# plt.figure()
# speedmag_ex = [x * -1 for x in allspeeds_dn_ex]
# speedmag_fs = [x * -1 for x in allspeeds_fs]

plt.figure()
plt.boxplot(allspeeds_dn_ex, positions=[0], showfliers=False, patch_artist=True, boxprops=dict(facecolor='cornflowerblue', color='cornflowerblue'),
            capprops=dict(color='cornflowerblue'),
            whiskerprops=dict(color='cornflowerblue'),
            medianprops=dict(color='white', linewidth = 2))
# plt.boxplot(speedmag_fs, positions=[0.3], showfliers=False, patch_artist=True, boxprops=dict(facecolor='indianred', color='indianred'),
#             capprops=dict(color='indianred'),
#             whiskerprops=dict(color='indianred'),
#             medianprops=dict(color='white', linewidth = 2))

x1 = np.random.normal(0, 0.01, size=len(allspeeds_dn_ex))
# x2 = np.random.normal(0.3, 0.01, size=len(speedmag_fs))
plt.plot(x1, allspeeds_dn_ex, '.', color = 'k', fillstyle = 'none', markersize = 8, zorder =3)
# plt.plot(x2, speedmag_fs, '.', color = 'k', fillstyle = 'none', markersize = 8, zorder =3)
plt.axhline(0, color = 'silver')
plt.xticks([0, 0.3],['Ex', 'FS'])
plt.title('Velocity of DOWN propagation')
plt.ylabel('Velocity (cm/s)')   


# plt.figure()
# plt.title('DOWN duration (ex cells)')
# # bins = np.linspace(min(min(dur_dorsal),min(dur_ventral)),(max(max(dur_dorsal),max(dur_ventral))),20)
# plt.boxplot(np.concatenate( DD, axis=0 ), positions = [0], showfliers=False,patch_artist=True, boxprops=dict(facecolor='royalblue', color='royalblue'),
#             capprops=dict(color='royalblue'),
#             whiskerprops=dict(color='royalblue'),
#             medianprops=dict(color='white', linewidth = 2))

# plt.boxplot(np.concatenate( DV, axis=0 ), positions = [0.3], showfliers=False,patch_artist=True, boxprops=dict(facecolor='lightsteelblue', color='lightsteelblue'),
#             capprops=dict(color='lightsteelblue'),
#             whiskerprops=dict(color='lightsteelblue'),
#             medianprops=dict(color='white', linewidth = 2))
# plt.xticks([0, 0.3],['Dorsal', 'Ventral'])
# plt.ylabel('DOWN duration (ms)')
# x1 = np.random.normal(0, 0.01, size=len(np.concatenate( DD, axis=0 )))
# x2 = np.random.normal(0.3, 0.01, size=len(np.concatenate( DV, axis=0 )))
# plt.plot(x1, np.concatenate( DD, axis=0 )  , '.', color = 'k', fillstyle = 'none', markersize = 8, zorder =3, alpha=0.3)
# plt.plot(x2, np.concatenate( DV, axis=0 ) , '.', color = 'k', fillstyle = 'none', markersize = 8, zorder =3, alpha = 0.3)

# plt.figure()
# z,p = wilcoxon(np.array(ratios)-1)
# plt.hist(ratios, label = 'p-value =' + str(round(p,4)))
# plt.legend(loc = 'upper right')

#Raster and LFP code 

# fig, ax = plt.subplots()
# [plot(spikes[n].restrict(per).as_units('s').fillna(n), '|', color = 'k') for n in spikes.keys()]
# # plot(lfp.restrict(per).as_units('s'), color = 'k')
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.axes.get_yaxis().set_visible(False)


# per = nts.IntervalSet(start = 1251288000, end = 1251688000) #20min 51s 288ms

# n = len(depth)
# tmp = np.argsort(depth.flatten())
# desc = tmp[::-1][:n]


# fig, ax = plt.subplots()
# [plt.plot(spikes[i].restrict(per).as_units('ms').fillna(i), '|', color = 'k') for i in desc]
