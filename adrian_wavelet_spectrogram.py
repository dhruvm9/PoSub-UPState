#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 15:04:19 2023

@author: dhruv
"""
import numpy as np 
import pandas as pd 
import scipy.io
from functions import *
from wrappers import *
import os, sys
import time 
import matplotlib.pyplot as plt 
import pynapple as nap
from Wavelets import MyMorlet as Morlet
import seaborn as sns
from scipy.fft import fft, ifft
from scipy.stats import wilcoxon, pearsonr
from scipy.signal import hilbert, fftconvolve
import matplotlib.cm as cm
import matplotlib.colors as colors
import math 
import pickle

#%% 

def MorletWavelet(f, ncyc, si):
    
    #Parameters
    s = ncyc/(2*np.pi*f)    #SD of the gaussian
    tbound = (4*s);   #time bounds - at least 4SD on each side, 0 in center
    tbound = si*np.floor(tbound/si)
    t = np.arange(-tbound,tbound,si) #time
    
    #Wavelet
    sinusoid = np.exp(2*np.pi*f*t*-1j)
    gauss = np.exp(-(t**2)/(2*(s**2)))
    
    A = 1
    wavelet = A * sinusoid * gauss
    wavelet = wavelet / np.linalg.norm(wavelet)
    return wavelet 

#%% 

# def FConv(kernel, signal):
    
#     #FFT parameters
#     n_kernel = len(kernel)
#     n_signal = len(signal)
#     n_convolution = n_kernel + n_signal-1
#     half_of_kernel_size = (n_kernel-1)/2
    
#     # FFT of wavelet and EEG data
#     fft_kernel = fft(kernel,n_convolution)
#     fft_signal = fft(signal,n_convolution)

#     convolution_result_fft = ifft((fft_kernel*fft_signal),n_convolution)
    
#     #cut off edges
#     convolution_result_fft = convolution_result_fft[int(half_of_kernel_size)+1:len(convolution_result_fft)-int(half_of_kernel_size)]
#     return convolution_result_fft


#%%
   
data_directory = '/media/dhruv/LaCie1/PoSub-UPState/Data/###AllPoSub'
rwpath = '/media/dhruv/LaCie1/PoSub-UPState/Project/Data'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

all_pspec_z = pd.DataFrame()
all_pspec_median = pd.DataFrame()

for s in datasets:
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    rawpath = os.path.join(rwpath,s)
    
    data = nap.load_session(rawpath, 'neurosuite')
    data.load_neurosuite_xml(rawpath)
    
    spikes = data.spikes  
    epochs = data.epochs
    channelorder = data.group_to_channel[0]
    seq = channelorder[::8].tolist()
    depth = np.arange(0, -800, -12.5)
        
    file = os.path.join(rawpath, name +'.DM.new_sws.evt')
    new_sws_ep  = data.read_neuroscope_intervals(name = 'new_sws', path2file = file)
    
    file = os.path.join(rawpath, name +'.DM.new_wake.evt')
    new_wake_ep  = data.read_neuroscope_intervals(name = 'new_wake', path2file = file)
    
    file = os.path.join(rawpath, name +'.evt.py.dow')
    down_ep = data.read_neuroscope_intervals(name = 'DOWN', path2file = file)
    
    file = os.path.join(rawpath, name +'.evt.py.upp')
    up_ep = data.read_neuroscope_intervals(name = 'UP', path2file = file)
    
    # peaks = pd.read_pickle(rawpath + '/' + s + '_LFP_peaks1.pkl')
    # lfp_all = pd.read_pickle(rawpath + '/' + s + '_LFP_all1.pkl')
        
     
    filepath = os.path.join(rwpath, name)
    listdir    = os.listdir(filepath)
    
    file = [f for f in listdir if  name + '.eeg' in f]
    filename = [name + '.eeg']
    matches = set(filename).intersection(set(file))
    
    if (matches == set()) is False:
        lfp = nap.load_eeg(filepath + '/' + name + '.eeg' , channel = seq[int(len(seq)/2)], n_channels = data.nChannels, frequency = 1250, precision ='int16', bytes_size = 2) 
    else: 
        lfp = nap.load_eeg(filepath + '/' + name + '.lfp' , channel = seq[int(len(seq)/2)] , n_channels = data.nChannels, frequency = 1250, precision ='int16', bytes_size = 2) 
    
#%%
    
    fs = 1250
    #window_ep = nap.IntervalSet(start = new_sws_ep['start'].values - 5, end = new_sws_ep['start'].values + 5)
    
    lfpsig = lfp #.restrict(window_ep)
    
    
   
#%%         
 
    fmin = 0.5
    fmax = 150
    nfreqs = 100
    ncyc = 3 #5
    si = 1/fs
    
    downsample = 10
    
    freqs = np.logspace(np.log10(fmin),np.log10(fmax),nfreqs)
    
    nfreqs = len(freqs)
    
    wavespec = nap.TsdFrame(t = lfpsig.index.values[::downsample], columns = freqs)
    powerspec = nap.TsdFrame(t = lfpsig.index.values[::downsample], columns = freqs)
        
    for f in range(len(freqs)):
         wavelet = MorletWavelet(freqs[f],ncyc,si)
         tmpspec = fftconvolve(lfpsig.values, wavelet, mode = 'same')
         wavespec[freqs[f]] = tmpspec [::downsample]
         temppower = abs(wavespec[freqs[f]]) #**2
         powerspec[freqs[f]] =  temppower #(temppower.values/np.median(temppower.values))
    
 #%%
        
    DU = nap.Tsd(up_ep['start'].values)
      
    realigned = powerspec.index[powerspec.index.get_indexer(DU.index.values, method='nearest')]
    
    pspec_median = pd.DataFrame()
    pspec_z = pd.DataFrame()
    
    for i in range(len(powerspec.columns)):
        tmp = nap.compute_perievent(powerspec[powerspec.columns[i]], nap.Ts(realigned.values) , minmax = (-1,1), time_unit = 's')
           
        peth_all = []
        for j in range(len(tmp)):
            peth_all.append(tmp[j].as_series())
            
        trials = pd.concat(peth_all, axis = 1, join = 'outer')
        
        z = ((trials - trials.mean()) / trials.std()).mean(axis = 1)    
        pspec_z[freqs[i]] = z
        
        mdn = (trials/trials.median()).mean(axis = 1)
        pspec_median[freqs[i]] = mdn
        
    all_pspec_median = pd.concat((pspec_median, all_pspec_median))
    all_pspec_z = pd.concat((pspec_z, all_pspec_z))
    
    
         
#%%

specgram_z = all_pspec_z.groupby(all_pspec_z.index).mean()
specgram_m = all_pspec_median.groupby(all_pspec_median.index).mean()
    
# ##Plotting 

## Z-scored 

labels = 2**np.arange(8)[3:]
norm = colors.TwoSlopeNorm(vmin=specgram_z[freqs[38:]][-0.1:0.5].values.min(),vcenter=0, vmax = specgram_z[freqs[38:]][-0.1:0.5].values.max())
       
fig, ax = plt.subplots()
plt.title('Z-scored spectrogram')
cax = ax.imshow(specgram_z[freqs[38:]][-0.1:0.5].T, aspect = 'auto', cmap = 'seismic', interpolation='bilinear', 
           origin = 'lower',
           extent = [specgram_z[freqs[38:]][-0.1:0.5].index.values[0], 
                     specgram_z[freqs[38:]][-0.1:0.5].index.values[-1],
                     np.log10(specgram_z[freqs[38:]].columns[0]),
                     np.log10(specgram_z[freqs[38:]].columns[-1])], 
           norm = norm)
plt.xlabel('Time from DU (s)')
plt.xticks([0, 0.25, 0.5])
plt.ylabel('Freq (Hz)')
plt.yticks(np.log10(labels), labels = labels)
cbar = fig.colorbar(cax, label = 'Power (z)')
# cbar = fig.colorbar(cax, ticks=[specgram_z[freqs[38:]][-0.1:0.5].values.min(),0, specgram_z[freqs[38:]][-0.1:0.5].values.max()], label = 'Power (z-scored)')
plt.axvline(0, color = 'k',linestyle = '--')
plt.gca().set_box_aspect(1)

## Median Normalized

norm = colors.TwoSlopeNorm(vmin=specgram_m[freqs[38:]][-0.1:0.5].values.min(),vcenter=1, vmax = specgram_m[freqs[38:]][-0.1:0.5].values.max())
       
plt.figure()
plt.title('Median-normalized spectrogram')
plt.imshow(specgram_m[freqs[38:]][-0.1:0.5].T, aspect = 'auto', cmap = 'seismic', interpolation='bilinear', 
           origin = 'lower',
           extent = [specgram_m[freqs[38:]][-0.1:0.5].index.values[0],
                     specgram_m[freqs[38:]][-0.1:0.5].index.values[-1],
                     np.log10(specgram_z[freqs[38:]].columns[0]),
                     np.log10(specgram_z[freqs[38:]].columns[-1])], 
           norm = norm)
plt.xlabel('Time from DU (s)')
plt.ylabel('Freq (Hz)')  
plt.yticks(np.log10(labels), labels = labels)
plt.colorbar()
plt.axvline(0, color = 'k', linestyle ='--')
plt.gca().set_box_aspect(1)

#%%

specgram_z.to_pickle(data_directory + '/specgram_z.pkl')
specgram_m.to_pickle(data_directory + '/specgram_m.pkl')


  
    
    

   
 

    
