# -*- coding: utf-8 -*-
"""
Created on Mon May  4 17:16:24 2020

@author: Dhruv
"""

import numpy as np
import pandas as pd
import neuroseries as nts
import os 
from scipy.io import loadmat 
from pylab import * 
from wrappers import * 
from functions import * 
import sys


data_directory = '/media/DataAdrienBig/PeyracheLabData/Adrian/A3700/A3723/A3723-201115'

spikes,shank = loadSpikeData(data_directory)
n_channels, fs, shank_to_channel = loadXML(data_directory)

neuron_index = list(spikes.keys())

hd_neuron_index = loadHDCellInfo(data_directory + '/Analysis/HDCells.mat', neuron_index)

hd_spikes = {}
for neuron in hd_neuron_index: 
    hd_spikes[neuron] = spikes[neuron]

#start = []
#end = []

filepath = os.path.join(data_directory, 'Analysis')
listdir    = os.listdir(filepath)
file = [f for f in listdir if 'BehavEpochs' in f]
behepochs = scipy.io.loadmat(os.path.join(filepath,file[0]))

wake_ep = np.hstack([behepochs['wake1Ep'][0][0][1],behepochs['wake1Ep'][0][0][2]])
wake_ep = nts.IntervalSet(wake_ep[:,0], wake_ep[:,1], time_units = 's').drop_short_intervals(0.0)

#for k in behepochs.keys():
#    if 'wake' in k and 'Info' not in k: 
#        start.append(behepochs[k][0][0][1])
#        end.append(behepochs[k][0][0][2])



data = pd.read_csv('/media/DataAdrienBig/PeyracheLabData/Adrian/A3700/A3723/A3723-201115/Analysis/Tracking_data.csv', header = None)
data = pd.DataFrame(index = data[0].values*1e6, data = data[[1,2,3]].values, columns=['x', 'y', 'ang'])
data = data.loc[~data.index.duplicated(keep='first')]
data['ang'] = data['ang'] *(np.pi/180) #convert degrees to radian
data['ang'] = (data['ang'] + 2*np.pi) % (2*np.pi) #convert [-pi, pi] to [0, 2pi]
mouse_position = nts.TsdFrame(data)

figure()
plot(mouse_position['x'].restrict(wake_ep).values, mouse_position['y'].restrict(wake_ep).values)
xlabel("x position (cm)")
ylabel("y position (cm)")
show()

bins = np.linspace(0, 2*np.pi, 180)
idx = bins[0:-1]+np.diff(bins)/2
angle = mouse_position['ang'].restrict(wake_ep)

column_names = ['A3723-201115_'+str(k) for k in hd_spikes.keys()]
tuning_curves = pd.DataFrame(index = idx, columns = column_names)

for k in hd_spikes.keys():
    spks = hd_spikes[k]
    spks = spks.restrict(wake_ep)
    angle_spike = angle.restrict(wake_ep).realign(spks)
    spike_count, bin_edges = np.histogram(angle_spike, bins)
    occupancy, _ = np.histogram(angle, bins)
    spike_count = spike_count/occupancy		
    tcurves = spike_count*120	
    tuning_curves['A3723-201115_'+str(k)] = tcurves
    
tuning_curves = smoothAngularTuningCurves(tuning_curves, 10, 2)

figure()
for i, n in enumerate(tuning_curves.columns):
	subplot(11,11,i+1, projection = 'polar')
	plot(tuning_curves[n])	
show()
		
    

