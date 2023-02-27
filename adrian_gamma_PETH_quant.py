#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 15:35:38 2022

@author: dhruv
"""
import numpy as np 
import pandas as pd 
import scipy.io
from functions import *
from wrappers import *
import os, sys
import neuroseries as nts 
import time 
import matplotlib.pyplot as plt 
import pynapple as nap
from Wavelets import MyMorlet as Morlet
import seaborn as sns
from scipy.stats import wilcoxon, pearsonr, kendalltau, mannwhitneyu
import matplotlib.cm as cm
import seaborn as sns

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')
# datasets = np.loadtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'

posall = []
negall = []
minall = []
posp_all = []
negp_all = []
minp_all = []
pos_slope = []
neg_slope = []
min_slope = []

diffcorr_DU = []
diffp_DU = []

diffcorr_UD = []
diffp_UD = []

for s in datasets:
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    rawpath = os.path.join(rwpath,s)
    
    data = nap.load_session(rawpath, 'neurosuite')
    data.load_neurosuite_xml(rawpath)
    
#%%    
############################################################################################### 
    # LOADING DATA
###############################################################################################
    spikes = data.spikes  
    epochs = data.epochs
    channelorder = data.group_to_channel[0]
    depth = np.arange(0, -800, -12.5)
    seq = channelorder[::8].tolist()
    
#%%
# ############################################################################################### 
#     # LOAD UP AND DOWN STATE, NEW SWS AND NEW WAKE EPOCHS
# ###############################################################################################   
    file = os.path.join(rawpath, name +'.DM.new_sws.evt')
    new_sws_ep  = data.read_neuroscope_intervals(name = 'new_sws', path2file = file)
    
    file = os.path.join(rawpath, name +'.DM.new_wake.evt')
    new_wake_ep  = data.read_neuroscope_intervals(name = 'new_wake', path2file = file)
    
    # peaks = pd.read_pickle(rawpath + '/' + s + '_LFP_peaks.pkl')
    # lfp_all = pd.read_pickle(rawpath + '/' + s + '_LFP_all.pkl')
    
    peaks = pd.read_pickle(rawpath + '/' + s + '_LFP_peaks1.pkl')
    lfp_all = pd.read_pickle(rawpath + '/' + s + '_LFP_all1.pkl')
    
    # highgamma_all = pd.read_pickle(rawpath + '/' + s + '_highgamma_all.pkl') 
    highgamma_all = pd.read_pickle(rawpath + '/' + s + '_highgamma_all1.pkl') 
    
    gamma_all = pd.read_pickle(rawpath + '/' + s + '_gamma_all.pkl') 
    
#%%  
#Positive crossing of 0.5 value
 
    # poscross = highgamma_all.loc[0.0016:] > 0.5
    # poscross = gamma_all.loc[0.0016:] > 0.5
    
    # poskeep = poscross.columns[poscross.sum(0) > 0]
    
    # depths_keeping = []
    # for i in range(len(poscross.columns)): 
    #     if poscross.columns[i] in poskeep:
    #         depths_keeping.append(i)            
    
    # depth2 = depth[depths_keeping]
    
    # ends = np.array([poscross.index[np.where(poscross[i])[0][0]] for i in poskeep])
    # ends = ends*1e3
   
    
#Negative crossing of 0.5 value
 
    # negcross = highgamma_all.loc[-1:0.0016] > 0.5
    # negcross = gamma_all.loc[-1:0.0016] > 0.5
    
    # negkeep = poscross.columns[negcross.sum(0) > 0]
    
    # st = np.array([negcross.index[np.where(negcross[i])[0][-1]] for i in negkeep])
    # st = st*1e3
    
#Minimum of PETH
    # mins = []
    # for i in range(len(highgamma_all.columns)):
    #     mins.append(highgamma_all[i].idxmin())
    
    # for i in range(len(gamma_all.columns)):
    #     mins.append(gamma_all[i].idxmin())
    
    
    # mins = np.array(mins)
    # mins = mins*1e3
    
#Correlation for each session 
    
    # poscorr, posp = kendalltau(ends, depth2) 
    # posall.append(poscorr)
    # posp_all.append(posp)

    # negcorr, negp = kendalltau(st, depth2) 
    # negall.append(negcorr)
    # negp_all.append(negp)
    
    # mincorr, minp = kendalltau(mins, depth)
    # minall.append(mincorr)
    # minp_all.append(minp)

#Best fit line

#DU transition
    # ypos = np.zeros(len(ends))
    # mpos, bpos = np.polyfit(ends, depth2, 1)
    # pos_slope.append(abs(-(mpos)))
        
    # for i in range(len(ends)):
    #       ypos[i] = mpos*ends[i]
         
    # plt.figure()
    # plt.scatter(ends, depth2, alpha = 0.8, label = 'R = ' + str(round(poscorr,2)), color = 'cornflowerblue')
    # plt.plot(ends, ypos+bpos, label = 'speed = ' + str(round((abs(-(mpos))),2)), color = 'cornflowerblue')
    # plt.title('DU threshold crossing_' + s)
    # plt.xlabel('Lag (ms)')
    # plt.yticks([0, -400, -800])
    # plt.ylabel('Depth from top of probe (um)')
    # plt.legend(loc = 'upper right')
    
#UD transition    
    # yneg = np.zeros(len(st))
    # mneg, bneg = np.polyfit(st, depth2, 1)
    # neg_slope.append(abs(-(mneg)))
        
    # for i in range(len(st)):
    #       yneg[i] = mneg*st[i]
         
    # plt.figure()
    # plt.scatter(st, depth2, alpha = 0.8, label = 'R = ' + str(round(negcorr,4)))
    # plt.plot(yneg + bneg, depth2)
    # plt.title('UD threshold crossing_' + s)
    # plt.xlabel('Lag (ms)')
    # plt.yticks([0, -400, -800])
    # plt.ylabel('Depth from top of probe (um)')
    # plt.legend(loc = 'upper right')
    
#Minima
    # ymin = np.zeros(len(mins))
    # mmin, bmin = np.polyfit(mins, depth, 1)
    # min_slope.append(abs(-(mmin)))
        
    # for i in range(len(mins)):
    #       ymin[i] = mmin*mins[i]
         
    # plt.figure()
    # plt.scatter(mins, depth, alpha = 0.8, label = 'R = ' + str(round(negcorr,4)))
    # plt.plot(ymin + bmin, depth)
    # plt.title('Minima of Gamma Power_' + s)
    # plt.xlabel('Lag (ms)')
    # plt.yticks([0, -400, -800])
    # plt.ylabel('Depth from top of probe (um)')
    # plt.legend(loc = 'upper right')
        

    #sns.regplot(x = depth, y = ends, ci = 95)
    
#%%
#Speed stats

# posdf = pd.DataFrame(data = (np.vstack([pos_slope, posp_all])).T, columns = ['speed', 'pval'])
# negdf = pd.DataFrame(data = (np.vstack([neg_slope, negp_all])).T, columns = ['speed', 'pval'])
# mindf = pd.DataFrame(data = (np.vstack([min_slope, minp_all])).T, columns = ['speed', 'pval'])

# #Median correlation
# m1 = np.load(rwpath + '/' + 'mediancorr_highgamma_1.npy')
# m2 = np.load(rwpath + '/' + 'mediancorr_highgamma_2.npy')
# m3 = np.load(rwpath + '/' + 'mediancorr_highgamma_3.npy')

# p1 = np.load(rwpath + '/' + 'medianp_highgamma_1.npy')
# p2 = np.load(rwpath + '/' + 'medianp_highgamma_2.npy')
# p3 = np.load(rwpath + '/' + 'medianp_highgamma_3.npy')

# m1 = np.load(rwpath + '/' + 'mediancorr_gamma_1.npy')
# m2 = np.load(rwpath + '/' + 'mediancorr_gamma_2.npy')
# m3 = np.load(rwpath + '/' + 'mediancorr_gamma_3.npy')

# p1 = np.load(rwpath + '/' + 'medianp_gamma_1.npy')
# p2 = np.load(rwpath + '/' + 'medianp_gamma_2.npy')
# p3 = np.load(rwpath + '/' + 'medianp_gamma_3.npy')

# medcorr = np.concatenate([m1,m2,m3])
# medp = np.concatenate([p1,p2,p3])

# median_df = pd.DataFrame(data = (np.vstack([medcorr, medp])).T, columns = ['mediancorr', 'pval'])
# poscorr_df = pd.DataFrame(data = (np.vstack([posall, posp_all])).T, columns = ['poscorr', 'pval'])
# negcorr_df = pd.DataFrame(data = (np.vstack([negall, negp_all])).T, columns = ['negcorr', 'pval'])
# mincorr_df = pd.DataFrame(data = (np.vstack([minall, minp_all])).T, columns = ['mincorr', 'pval'])

#%%
#Speed plots

# plt.figure()    
# plt.title('Speed')
# plt.boxplot(posdf['speed'], positions = [0], showfliers=False, patch_artist=True, boxprops=dict(facecolor='cornflowerblue', color='cornflowerblue'),
#             capprops=dict(color='cornflowerblue'),
#             whiskerprops=dict(color='cornflowerblue'),
#             medianprops=dict(color='white', linewidth = 2))
# plt.boxplot(negdf['speed'], positions = [0.3], showfliers=False, patch_artist=True, boxprops=dict(facecolor='cornflowerblue', color='cornflowerblue'),
#             capprops=dict(color='cornflowerblue'),
#             whiskerprops=dict(color='cornflowerblue'),
#             medianprops=dict(color='white', linewidth = 2))
# plt.boxplot(mindf['speed'], positions = [0.6], showfliers=False, patch_artist=True, boxprops=dict(facecolor='cornflowerblue', color='cornflowerblue'),
#             capprops=dict(color='cornflowerblue'),
#             whiskerprops=dict(color='cornflowerblue'),
#             medianprops=dict(color='white', linewidth = 2))

# x1 = np.random.normal(0, 0.01, size=len(posdf['speed'][posdf['pval'] < 0.05]))
# x2 = np.random.normal(0.3, 0.01, size=len(negdf['speed'][negdf['pval'] < 0.05]))
# x3 = np.random.normal(0.6, 0.01, size=len(mindf['speed'][mindf['pval'] < 0.05]))

# x4 = np.random.normal(0, 0.01, size=len(posdf['speed'][posdf['pval'] > 0.05]))
# x5 = np.random.normal(0.3, 0.01, size=len(negdf['speed'][negdf['pval'] > 0.05]))
# x6 = np.random.normal(0.6, 0.01, size=len(mindf['speed'][mindf['pval'] > 0.05]))

# plt.plot(x1, posdf['speed'][posdf['pval'] < 0.05], '.', color = 'k', fillstyle = 'none', markersize = 8, zorder =3)
# plt.plot(x2, negdf['speed'][negdf['pval'] < 0.05], '.', color = 'k', fillstyle = 'none', markersize = 8, zorder =3)
# plt.plot(x3, mindf['speed'][mindf['pval'] < 0.05], '.', color = 'k', fillstyle = 'none', markersize = 8, zorder =3)

# plt.plot(x4, posdf['speed'][posdf['pval'] > 0.05], 'x', color = 'k', fillstyle = 'none', markersize = 6, zorder =3)
# plt.plot(x5, negdf['speed'][negdf['pval'] > 0.05], 'x', color = 'k', fillstyle = 'none', markersize = 6, zorder =3)
# plt.plot(x6, mindf['speed'][mindf['pval'] > 0.05], 'x', color = 'k', fillstyle = 'none', markersize = 6, zorder =3)

# plt.xticks([0, 0.3, 0.6],['DU', 'UD', 'Minima'])
# plt.ylabel('Velocity (mm/s)')   

#Correlation distribution  

# plt.figure()    
# plt.title('DU correlations')
# plt.boxplot(posall, positions = [0],showfliers=False, patch_artist=True, boxprops=dict(facecolor='cornflowerblue', color='cornflowerblue'),
#             capprops=dict(color='cornflowerblue'),
#             whiskerprops=dict(color='cornflowerblue'),
#             medianprops=dict(color='white', linewidth = 2))
# plt.boxplot(negall, positions = [0.3],showfliers=False, patch_artist=True, boxprops=dict(facecolor='cornflowerblue', color='cornflowerblue'),
#             capprops=dict(color='cornflowerblue'),
#             whiskerprops=dict(color='cornflowerblue'),
#             medianprops=dict(color='white', linewidth = 2))
# plt.boxplot(minall, positions = [0.6],showfliers=False, patch_artist=True, boxprops=dict(facecolor='cornflowerblue', color='cornflowerblue'),
#             capprops=dict(color='cornflowerblue'),
#             whiskerprops=dict(color='cornflowerblue'),
#             medianprops=dict(color='white', linewidth = 2))
# x1 = np.random.normal(0, 0.01, size=len(posall))
# x2 = np.random.normal(0.3, 0.01, size=len(negall))
# x3 = np.random.normal(0.6, 0.01, size=len(minall))

# plt.plot(x1, posall, '.', color = 'k', fillstyle = 'none', markersize = 8, zorder =3)
# plt.plot(x2, negall, '.', color = 'k', fillstyle = 'none', markersize = 8, zorder =3)
# plt.plot(x3, minall, '.', color = 'k', fillstyle = 'none', markersize = 8, zorder =3)
# plt.xticks([0, 0.3, 0.6],['DU', 'UD', 'Minima'])
# plt.ylabel('Lag v/s depth (R)')   

# z_DU_speed, p_DU_speed = wilcoxon(np.array(pos_slope)-0)
# z_UD_speed, p_UD_speed = wilcoxon(np.array(neg_slope)-0)
# z_min_speed, p_min_speed = wilcoxon(np.array(min_slope)-0)

# z_DU_corr, p_DU_corr = wilcoxon(np.array(posall)-0)
# z_UD_corr, p_UD_corr = wilcoxon(np.array(negall)-0)
# z_min_corr, p_min_corr = wilcoxon(np.array(minall)-0)

# t_speed, p_speed = mannwhitneyu(pos_slope, neg_slope)
# t_corr, p_corr = mannwhitneyu(posall, negall)



# fig, ax = plt.subplots()
# ax.scatter(negdf['speed'].values,unit_speed) 
# lims = [
#     np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
#     np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
# ]
# ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
# ax.set_aspect('equal')
# ax.set_xlim(lims)
# ax.set_ylim(lims)
# plt.xlabel('Gamma power UD speed') 
# plt.ylabel('Unit speed')


# fig, ax = plt.subplots()
# ax.scatter(mindf['speed'].values,unit_speed) 
# lims = [
#     np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
#     np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
# ]
# ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
# ax.set_aspect('equal')
# ax.set_xlim(lims)
# ax.set_ylim(lims)
# plt.xlabel('Gamma power minima speed')  
# plt.ylabel('Unit speed')

#%%
#Medians versus DU, UD, minima

# plt.figure()
# plt.scatter(median_df['mediancorr'],poscorr_df['poscorr'])
# plt.xlabel('Median R')
# plt.ylabel('DU corr')

# plt.figure()
# plt.scatter(median_df['mediancorr'],negcorr_df['negcorr'])
# plt.xlabel('Median R')
# plt.ylabel('UD corr')

# plt.figure()
# plt.scatter(median_df['mediancorr'],mincorr_df['mincorr'])
# plt.xlabel('Median R')
# plt.ylabel('Minima corr')

#%%
#Differential of Gamma PETH

    w = pd.DataFrame(index = highgamma_all.index.values[0:-1], columns = highgamma_all.columns) #Drop end values
    
    # w = pd.DataFrame(index = gamma_all.index.values[0:-1], columns = gamma_all.columns) #Drop end values


    peakval = []
    troughs = []
    
    for i in highgamma_all.columns:
        w[i] = np.diff(highgamma_all[i].values)
        w[i] = w[i].rolling(window=60,win_type='gaussian',center=True,min_periods=1).mean(std=80)
        peakval.append(w[i][0:1].idxmax()*1e3)
        troughs.append(w[i][-1:0].idxmin()*1e3)
    
    # for i in gamma_all.columns:
    #     w[i] = np.diff(gamma_all[i].values)
    #     w[i] = w[i].rolling(window=60,win_type='gaussian',center=True,min_periods=1).mean(std=80)
    #     peakval.append(w[i].idxmax()*1e3)
    #     troughs.append(w[i].idxmin()*1e3)
    
    
    corr_DU, p_DU = kendalltau(peakval,depth)
    diffcorr_DU.append(corr_DU)
    diffp_DU.append(p_DU)    
    
    corr_UD, p_UD = kendalltau(troughs,depth)
    diffcorr_UD.append(corr_UD)
    diffp_UD.append(p_UD) 
        
    # plt.plot(w[-0.2:0.3][channelorder[31]], color='cornflowerblue')
    # plt.axvline(peakval[31]/1e3, color = 'silver', linestyle = '--')
    # plt.axvline(troughs[31]/1e3, color = 'silver', linestyle = '--')
    # plt.xticks([-0.2, 0, 0.3])
    # plt.gca().set_box_aspect(1)
    
    mpos, bpos = np.polyfit(peakval,depth,1)
    pos_slope.append(abs(-(mpos)))
    
    ypos = np.zeros(len(peakval))
    for i in range(len(peakval)):
           ypos[i] = (mpos*peakval[i]) + bpos
    
    mneg, bneg = np.polyfit(troughs,depth,1)
    neg_slope.append(abs(-(mneg)))
    
    yneg = np.zeros(len(troughs))
    for i in range(len(troughs)):
           yneg[i] = (mneg*troughs[i]) + bneg
    
    plt.figure()
    plt.title(s)
    plt.scatter(peakval, depth, label = 'R = ' + str(round(corr_DU,2)), color = 'cornflowerblue')   
    plt.plot(peakval,ypos, label = 'speed = ' + str(round((abs(-(mpos))),2)), color = 'cornflowerblue')
    plt.xlabel('DU Lag (ms)')
    plt.ylabel('Depth (um)')
    plt.yticks([0, -400, -800])
    plt.legend(loc = 'upper right')
    plt.gca().set_box_aspect(1)

#%%
    # plt.figure()
    # plt.title(s)
    # plt.scatter(troughs, depth, label = 'R = ' + str(round(corr_UD,2)), color = 'orange')   
    # plt.plot(troughs,yneg, label = 'speed = ' + str(round((abs(-(mneg))),2)), color = 'orange')
    # plt.xlabel('UD Lag (ms)')
    # plt.ylabel('Depth (um)')
    # plt.yticks([0, -400, -800])
    # plt.legend(loc = 'upper right')

#%% 
    
    # plt.figure(figsize = (24,12))
    # plt.suptitle(s)
    # plt.subplot(131)
    # j = 0
    # for i in seq:
    #     plt.plot(w[-0.75:0.75][i], color=cm.inferno(j/8), label = j)
    #     plt.axvline(0, color = 'k', linestyle = '--')
    #     plt.legend(loc = 'upper right')
    #     j+=1
    # plt.subplot(132)
    # plt.scatter(peakval, depth, label = 'R = ' + str(round(corr_DU,2)), color = 'cornflowerblue')   
    # plt.plot(peakval,ypos, label = 'speed = ' + str(round((abs(-(mpos))),2)), color = 'cornflowerblue')
    # plt.xlabel('DU Lag (ms)')
    # plt.ylabel('Depth (um)')
    # plt.legend(loc = 'upper right')
    # plt.subplot(133)
    # plt.scatter(troughs, depth, label = 'R = ' + str(round(corr_UD,2)), color = 'cornflowerblue') 
    # plt.plot(troughs,yneg, label = 'speed = ' + str(round((abs(-(mneg))),2)), color = 'cornflowerblue')
    # plt.xlabel('UD Lag (ms)')
    # plt.ylabel('Depth (um)')
    # plt.legend(loc = 'upper right')

#%%

DUcorr = pd.DataFrame(diffcorr_DU)
UDcorr = pd.DataFrame(diffcorr_UD)
DUtype = pd.DataFrame(['DU' for x in range(len(diffcorr_DU))])
UDtype = pd.DataFrame(['UD' for x in range(len(diffcorr_UD))])

b = pd.DataFrame()
b['corr'] = pd.concat([DUcorr,UDcorr])
b['type'] = pd.concat([DUtype,UDtype])

#%% 

plt.figure()
sns.set_style('white')
palette = ['royalblue', 'lightsteelblue']
ax = sns.violinplot( x = b['type'], y=b['corr'] , data = b, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = b['type'], y=b['corr'] , data = b, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.swarmplot(x = b['type'], y=b['corr'], data=b, color = 'k', dodge=False, ax=ax)
# sns.stripplot(x = b['type'], y=b['corr'], data=b, color = 'k', dodge=False, ax=ax)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.axhline(0, color = 'silver')
plt.ylabel('Delay v/s depth (R)')
ax.set_box_aspect(1)


z_du, p_du = wilcoxon(np.array(DUcorr)-0)
z_ud, p_ud = wilcoxon(np.array(UDcorr)-0)
t, p = mannwhitneyu(DUcorr, UDcorr)

#%%

# DU_df = pd.DataFrame(data = (np.vstack([diffcorr_DU, diffp_DU])).T, columns = ['DU', 'pval'])
# UD_df = pd.DataFrame(data = (np.vstack([diffcorr_UD, diffp_UD])).T, columns = ['UD', 'pval'])


# plt.figure()    
# plt.title('Differental correlations')
# plt.boxplot(DU_df['DU'], positions = [0],showfliers=False, patch_artist=True, boxprops=dict(facecolor='cornflowerblue', color='cornflowerblue'),
#             capprops=dict(color='cornflowerblue'),
#             whiskerprops=dict(color='cornflowerblue'),
#             medianprops=dict(color='white', linewidth = 2))
# plt.boxplot(UD_df['UD'], positions = [0.3],showfliers=False, patch_artist=True, boxprops=dict(facecolor='lightsteelblue', color='lightsteelblue'),
#             capprops=dict(color='lightsteelblue'),
#             whiskerprops=dict(color='lightsteelblue'),
#             medianprops=dict(color='white', linewidth = 2))

# x1 = np.random.normal(0, 0.01, size=len(DU_df['DU'][DU_df['pval'] < 0.05]))
# x2 = np.random.normal(0.3, 0.01, size=len(UD_df['UD'][UD_df['pval'] < 0.05]))
# x3 = np.random.normal(0, 0.01, size=len(DU_df['DU'][DU_df['pval'] >= 0.05]))
# x4 = np.random.normal(0.3, 0.01, size=len(UD_df['UD'][UD_df['pval'] >= 0.05]))

# plt.plot(x1, DU_df['DU'][DU_df['pval'] < 0.05] , 'x', color = 'k', fillstyle = 'none', markersize = 6, zorder =3, label = 'p < 0.05')
# plt.plot(x2, UD_df['UD'][UD_df['pval'] < 0.05] , 'x', color = 'k', fillstyle = 'none', markersize = 6, zorder =3)
# plt.plot(x3, DU_df['DU'][DU_df['pval'] > 0.05] , '.', color = 'k', fillstyle = 'none', markersize = 6, zorder =3, label = 'p >= 0.05')
# plt.plot(x4, UD_df['UD'][UD_df['pval'] > 0.05] , '.', color = 'k', fillstyle = 'none', markersize = 6, zorder =3)
# plt.xticks([0, 0.3],['DU', 'UD'])
# plt.axhline(0, color = 'silver')
# plt.ylabel('Lag v/s depth (R)')   
# plt.legend(loc = 'upper right')

# posdf = pd.DataFrame(data = (np.vstack([pos_slope, diffp_DU])).T, columns = ['speed', 'pval'])
# negdf = pd.DataFrame(data = (np.vstack([neg_slope, diffp_UD])).T, columns = ['speed', 'pval'])

# plt.figure()    
# plt.title('DU Speed')
# plt.boxplot(posdf['speed'][posdf['pval'] < 0.05], positions = [0], showfliers=False, patch_artist=True, boxprops=dict(facecolor='darkslategray', color='darkslategray'),
#             capprops=dict(color='darkslategray'),
#             whiskerprops=dict(color='darkslategray'),
#             medianprops=dict(color='white', linewidth = 2))

# x1 = np.random.normal(0, 0.01, size=len(posdf['speed'][posdf['pval'] < 0.05]))
# plt.plot(x1, posdf['speed'][posdf['pval'] < 0.05], '.', color = 'k', fillstyle = 'none', markersize = 8, zorder =3)
# plt.xticks([])
# plt.ylabel('Velocity (mm/s)')   


#%% 

#Run adrian_hist_PETH_up before running this

unit_speed = regs['spd_ex']

corr, p = pearsonr(posdf['speed'],unit_speed)
m,b = np.polyfit(posdf['speed'],unit_speed,1)
y = m*posdf['speed'] + b
plt.figure()
plt.scatter(posdf['speed'].values,unit_speed, label = 'R = ' + str(round(corr,2)), color = 'k')
plt.plot(posdf['speed'].values,y, color = 'k')
plt.xlabel('LFP speed (mm/s)')
plt.ylabel('Unit speed (mm/s)')
plt.legend(loc = 'upper right')

#%% 

#Run adrian_hist_PETH_up before running this

unit_speed = regs['spd_ex'][regs['pval_ex'] < 0.05]

plt.figure()    
plt.title('DU Speed')
plt.boxplot(unit_speed, positions = [0], showfliers=False, patch_artist=True, boxprops=dict(facecolor='darkslategray', color='darkslategray'),
            capprops=dict(color='darkslategray'),
            whiskerprops=dict(color='darkslategray'),
            medianprops=dict(color='white', linewidth = 2))
plt.boxplot(posdf['speed'][posdf['pval'] < 0.05], positions = [0.3], showfliers=False, patch_artist=True, boxprops=dict(facecolor='darkslategray', color='darkslategray'),
            capprops=dict(color='darkslategray'),
            whiskerprops=dict(color='darkslategray'),
            medianprops=dict(color='white', linewidth = 2))

x1 = np.random.normal(0.3, 0.01, size=len(posdf['speed'][posdf['pval'] < 0.05]))
x2 = np.random.normal(0, 0.01, size=len(unit_speed))
plt.plot(x1, posdf['speed'][posdf['pval'] < 0.05], '.', color = 'k', fillstyle = 'none', markersize = 8, zorder =3)
plt.plot(x2, unit_speed, '.', color = 'k', fillstyle = 'none', markersize = 8, zorder =3)
plt.xticks([0, 0.3],['Spikes', 'LFP'])
plt.ylabel('Velocity (mm/s)')  

t,p = mannwhitneyu(unit_speed,posdf['speed'][posdf['pval'] < 0.05])

    
#%%
#Units vs LFP

# plt.figure()
# plt.scatter(a['allcoefs_up_ex'],poscorr_df['poscorr'])
# plt.xlabel('Unit R (DU)')
# plt.ylabel('LFP R (DU)')

# plt.figure()
# plt.scatter(a['allcoefs_dn_ex'],negcorr_df['negcorr'])
# plt.xlabel('Unit R (UD)')
# plt.ylabel('LFP R (UD)')

#%% 

#Run adrian_hist_PETH_dn before running this

corr, p = pearsonr(a['allcoefs_up_ex'],DU_df['DU'])
m,b = np.polyfit(a['allcoefs_up_ex'],DU_df['DU'],1)
y = m*a['allcoefs_up_ex'] + b

plt.figure()
plt.scatter(a['allcoefs_up_ex'],DU_df['DU'], label = 'R = ' + str(round(corr,2)), color = 'k')
plt.plot(a['allcoefs_up_ex'],y, color = 'k')
plt.xlabel('Unit R (DU)')
plt.ylabel('LFP differential R (DU)')
plt.yticks([-1, -0.5, 0, 0.5, 1])
plt.xticks([-0.5, -0.25, 0, 0.25])
plt.legend(loc = 'upper right')

#%%

# plt.figure()
# plt.scatter(a['allcoefs_dn_ex'],UD_df['UD'])
# plt.xlabel('Unit R (UD)')
# plt.ylabel('LFP differential R (UD)')

    