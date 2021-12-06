# -*- coding: utf-8 -*-
"""
Created on Fri May  8 13:05:34 2020

@author: Dhruv
"""
#import libraries
import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys

#loading general info
data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/AllPoSubHor'
datasets = np.loadtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')
#datasets = np.loadtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
rwpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'

for s in datasets:
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    rawpath = os.path.join(rwpath,s)

############################################################################################### 
    # LOADING DATA
###############################################################################################

    spikes, shank = loadSpikeData(path)
    n_channels, fs, shank_to_channel = loadXML(rawpath)




#reading stuff
data = pd.read_csv('/media/DataAdrienBig/PeyracheLabData/Adrian/A3700/A3701/A3701-191119/Analysis/Tracking_data.csv', header = None)
data = pd.DataFrame(index = data[0].values*1e6, data = data[[1,2,3]].values, columns=['x', 'y', 'ang'])
data = data.loc[~data.index.duplicated(keep='first')]
data['ang'] = data['ang'] *(np.pi/180) #convert degrees to radian
data['ang'] = (data['ang'] + 2*np.pi) % (2*np.pi) #convert [-pi, pi] to [0, 2pi]
mouse_position = nts.TsdFrame(data)

#fishing out the sleep epoch
filepath = os.path.join(data_directory, 'Analysis')
listdir = os.listdir(filepath)
file = [f for f in listdir if 'BehavEpochs' in f]
behepochs = scipy.io.loadmat(os.path.join(filepath,file[0]))

sleep_ep = np.hstack([behepochs['sleepPostEp'][0][0][1],behepochs['sleepPostEp'][0][0][2]])
sleep_ep = nts.IntervalSet(sleep_ep[:,0], sleep_ep[:,1], time_units = 's').drop_short_intervals(0.0)

#get acceleration data
acceleration = loadAuxiliary(data_directory, 1, fs = 20000) #second argument = 1, third argument is fs in some cases

#make new sleep epoch
newsleep_ep = refineSleepFromAccel(acceleration, sleep_ep)

lfp = loadLFP(os.path.join(data_directory,'A3701-191119.lfp'), n_channels, 1, 1250, 'int16')
lfp = downsample(lfp, 1, 5)

lfp_filt_theta = nts.Tsd(lfp.index.values, butter_bandpass_filter(lfp, 4, 12, 1250/5, 2))
power_theta = nts.Tsd(lfp_filt_theta.index.values, np.abs(lfp_filt_theta.values))
power_theta = power_theta.rolling(window=1000,win_type='gaussian',center=True,min_periods=1).mean(std=80)

lfp_filt_delta = nts.Tsd(lfp.index.values, butter_bandpass_filter(lfp, 0.5, 4, 1250/5, 2))
power_delta = nts.Tsd(lfp_filt_delta.index.values, np.abs(lfp_filt_delta.values))
power_delta = power_delta.rolling(window=1000,win_type='gaussian',center=True,min_periods=1).mean(std=80)

ratio = nts.Tsd(t = power_theta.index.values, d = np.log(power_theta.values/power_delta.values))
ratio2 = ratio.rolling(window=10000,win_type='gaussian',center=True,min_periods=1).mean(std=200)
ratio2 = nts.Tsd(t = ratio2.index.values, d = ratio2.values)

index = (ratio2.as_series() > 0).values*1.0
start_cand = np.where((index[1:] - index[0:-1]) == 1)[0]+1
end_cand = np.where((index[1:] - index[0:-1]) == -1)[0]
if end_cand[0] < start_cand[0]: end_cand = end_cand[1:]
if end_cand[-1] < start_cand[-1]: start_cand = start_cand[0:-1]
tmp = np.where(end_cand != start_cand)
start_cand = ratio2.index.values[start_cand[tmp]]
end_cand = ratio2.index.values[end_cand[tmp]]

good_ep = nts.IntervalSet(start_cand, end_cand)
good_ep = newsleep_ep.intersect(good_ep)
good_ep = good_ep.merge_close_intervals(10, time_units = 's')
good_ep = good_ep.drop_short_intervals(20, time_units = 's')
good_ep = good_ep.reset_index(drop=True)

theta_rem_ep = good_ep
sws_ep = newsleep_ep.set_diff(theta_rem_ep)
sws_ep = sws_ep.merge_close_intervals(0).drop_short_intervals(0)

writepath = '/home/dhruv/PoSub_Upstate'
#writeNeuroscopeEvents(os.path.join(writepath,'A3701-191119-sleep2.rem.evt'), theta_rem_ep, "Theta")
#writeNeuroscopeEvents(os.path.join(writepath,'A3701-191119-sleep2.sws.evt'), sws_ep, "SWS")

sys.exit()

phase = getPhase(lfp, 6, 14, 16, fs/5.)

figure()
ax = subplot(311)
plt.title('LFP trace')
[plot(lfp.restrict(sws_ep.loc[[i]]), color = 'blue') for i in sws_ep.index]
plot(lfp_filt_delta.restrict(sws_ep), color = 'orange')
subplot(312, sharex = ax)
plt.title('Theta/Delta ratio')
[plot(ratio.restrict(sws_ep.loc[[i]]), color = 'blue') for i in sws_ep.index]
plot(ratio2.restrict(sws_ep), color = 'orange')
axhline(0)
subplot(313, sharex = ax)
plt.title('Acceleration')
plot(acceleration[0].restrict(sws_ep))
show()
