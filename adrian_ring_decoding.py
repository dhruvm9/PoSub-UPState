#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 12:03:20 2021

@author: dhruv
"""
import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
from umap import UMAP
import sys
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import Isomap


path = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub/A3717-201021'
rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data/A3717-201021'

spikes, shank = loadSpikeData(path)
n_channels, fs, shank_to_channel = loadXML(rwpath)

data = pd.read_csv(path + '/Analysis/Tracking_data.csv', header = None)
data = pd.DataFrame(index = data[0].values*1e6, data = data[[1,2,3]].values, columns=['x', 'y', 'ang'])
data = data.loc[~data.index.duplicated(keep='first')]
data['ang'] = data['ang'] *(np.pi/180) #convert degrees to radian
data['ang'] = (data['ang'] + 2*np.pi) % (2*np.pi) #convert [-pi, pi] to [0, 2pi]
position = nts.TsdFrame(data)

#fishing out wake and sleep epochs
filepath = os.path.join(path, 'Analysis')
listdir    = os.listdir(filepath)
file = [f for f in listdir if 'BehavEpochs' in f]
behepochs = scipy.io.loadmat(os.path.join(filepath,file[0]))

# ############################################################################################### 
#     # LOAD UP AND DOWN STATE, NEW SWS AND NEW WAKE EPOCHS
# ###############################################################################################   
    
file = os.path.join(rwpath +'/A3717-201021' + '.evt.py.dow')
if os.path.exists(file):
    tmp = np.genfromtxt(file)[:,0]
    tmp = tmp.reshape(len(tmp)//2,2)/1000
    down_ep = nts.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')

file = os.path.join(rwpath + '/A3717-201021' + '.evt.py.upp')
if os.path.exists(file):
    tmp = np.genfromtxt(file)[:,0]
    tmp = tmp.reshape(len(tmp)//2,2)/1000
    up_ep = nts.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')

file = os.path.join(rwpath +'/A3717-201021'  +'.DM.new_sws.evt')
if os.path.exists(file):
    tmp = np.genfromtxt(file)[:,0]
    tmp = tmp.reshape(len(tmp)//2,2)/1000
    new_sws_ep = nts.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')

file = os.path.join(rwpath + '/A3717-201021'  +'.DM.new_wake.evt')
if os.path.exists(file):
    tmp = np.genfromtxt(file)[:,0]
    tmp = tmp.reshape(len(tmp)//2,2)/1000
    new_wake_ep = nts.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
    
wake_ep = np.hstack([behepochs['wake1Ep'][0][0][1],behepochs['wake1Ep'][0][0][2]])
wake_ep = nts.IntervalSet(wake_ep[:,0], wake_ep[:,1], time_units = 's').drop_short_intervals(0.0)

sleep2_ep = np.hstack([behepochs['sleep2Ep'][0][0][1],behepochs['sleep2Ep'][0][0][2]])
sleep2_ep = nts.IntervalSet(sleep2_ep[:,0], sleep2_ep[:,1], time_units = 's').drop_short_intervals(0.0)

file = os.path.join(rwpath + '/A3717-201021'  +'.rem.evt')
if os.path.exists(file):
    tmp = np.genfromtxt(file)[:,0]
    tmp = tmp.reshape(len(tmp)//2,2)/1000
    rem_ep = nts.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
    

# ############################################################################################### 
#     # LOAD MAT FILES
# ############################################################################################### 
filepath = os.path.join(path, 'Analysis')
listdir    = os.listdir(filepath)

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
		
#############################################################################################

bins = np.linspace(0, 2*np.pi, 180)
idx = bins[0:-1]+np.diff(bins)/2
angle = position['ang'].restrict(wake_ep)
neurons = np.sort(list(spikes.keys()))[hd]

####################################################################################################################
# BIN WAKE
####################################################################################################################
bin_size = 300
bins = np.arange(wake_ep.as_units('ms').start.iloc[0], wake_ep.as_units('ms').end.iloc[-1]+bin_size, bin_size)
spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = neurons)
for i in neurons:
	spks = spikes[i].as_units('ms').index.values
	spike_counts[i], _ = np.histogram(spks, bins)

rate_wake = np.sqrt(spike_counts/(bin_size*1e-3))
# rate_wake = spike_counts/(bin_size*1e-3)


# binning angle
angle = position['ang'].restrict(wake_ep.loc[[0]])
wakangle = pd.Series(index = np.arange(len(bins)-1),dtype = float64)
tmp = angle.groupby(np.digitize(angle.as_units('ms').index.values, bins)-1).mean()
wakangle.loc[tmp.index] = tmp
wakangle.index = pd.Index(bins[0:-1] + np.diff(bins)/2.)

# dropna
rate_wake = rate_wake[~wakangle.isna()]
wakangle = wakangle[~wakangle.isna()]


H = wakangle.values/(2*np.pi)
HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T
RGB = hsv_to_rgb(HSV)

tmp = rate_wake.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=3).values
ump = Isomap(n_components = 2, n_neighbors = 100).fit_transform(tmp)

scatter(ump[:,0], ump[:,1], c=RGB)

####################################################################################################################
# BIN SLEEP
####################################################################################################################
bin_size = 10

rates = []
timebins = pd.DataFrame(columns = ['bins', 'ix'])


for j in range(len(up_ep)):

    bins = np.arange(up_ep.as_units('ms').start.iloc[j], up_ep.as_units('ms').end.iloc[j]+bin_size, bin_size)
    spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = neurons)
    
    x = pd.DataFrame(columns = ['bins', 'ix'])
    x['bins'] = bins
    x['ix'] = j
        
    timebins = timebins.append(x,ignore_index=True)
    
    for i in neurons:
        spks = spikes[i].as_units('ms').index.values
        spike_counts[i], _ = np.histogram(spks, bins)

    rate_sleep = np.sqrt(spike_counts/(bin_size*1e-3))
    rates.append(rate_sleep)

r = pd.concat(rates)
r = r[sum(r,1) > np.percentile(sum(r,1),20)]
####################################################################################################################
# PROJECTION
####################################################################################################################

tmp2 = r.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=10).values

# from sklearn.decomposition import PCA

# tmp3 = PCA(n_components=10).fit_transform(tmp2)

ump = Isomap(n_components = 3, n_neighbors = 100).fit_transform(tmp2[0:20000])


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(ump[:,0], ump[:,1], ump[:,2])




scatter(ump[:,0], ump[:,1], c=RGB)




tmp3 = np.vstack((tmp, tmp2))


ump = UMAP(n_components = 2, n_neighbors = 5000, min_dist = 1).fit_transform(tmp3)

ump1 = ump[0:len(tmp)]
ump2 = ump[len(tmp):]


# ump2 = UMAP(n_components = 2, n_neighbors = 100, min_dist = 1).fit_transform(tmp2)


####################################################################################################################
# DECODING
####################################################################################################################
#center ring
# ump = ump - np.mean(ump,0)

# radius = 





figure()
scatter(ump1[:,0], ump1[:,1], s = 100, c= RGB, marker = '.', alpha = 0.8, linewidth = 0)


figure()
scatter(ump2[:,0], ump2[:,1], marker = '.', alpha = 0.5, linewidth = 0, s = 100)
show()