# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:21:30 2023

@author: kasum
"""

#https://www.jneurosci.org/content/29/2/493
import numpy as np
import pandas as pd
import neuroseries as nts
from wrappers_KA import *
# from pylab import *
import os, sys
# from functions import remove_immobility
from place_cell_functions_KA import *
# from pycircstat.tests import rayleigh
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
# from astropy.visualization import hist
# import statsmodels.api as sm
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import glob, os 

import seaborn as sns
# from sklearn import datasets, linear_model
# from sklearn.metrics import mean_squared_error, r2_score
# import time

###############################################################
# PARAMETERS
###############################################################
# plt.close('all')
data_directory=r'/media/dhruv/Expansion/Processed/B2618-231020_test' #'C:\Users\kasum\Downloads\forKadjita\forKadjita'
all_files = glob.glob(os.path.join(data_directory, "*.csv"))

eps=[]
for i in range(len(all_files)):
    if all_files[i].split('_')[-1].split('.')[0] != 'TS':
        eps.append(int(all_files[i].split('_')[-1].split('.')[0]))

#To do: use the number and id of csvs to identify positiont-sleep events 
if 0 in eps:
    episodes=['wake']*len(eps)
else:
    episodes=['sleep']+(['wake']*len(eps))

        
events=[i for i,x in enumerate(episodes) if x!='sleep']    

n_analogin_channels = 1
channel_optitrack=0 #calls the second opened ch
spikes,shank= loadSpikeData(data_directory) #shank tells the number of cells on each shank
n_channels, fs, shank_to_channel = loadXML(data_directory)  #shank to channel 
position= loadPosition(data_directory,events,episodes,n_analogin_channels,channel_optitrack)
wake_ep=loadEpoch(data_directory,'wake',episodes)

# position_=remove_immobility(position,speed_threshold = 0.02) #stationary points excluded



###########################################################################################################
### PLACE CELL ANALYSIS
###########################################################################################################

# #Define Epochs
# ep = nts.IntervalSet(start=wake_ep.loc[0,'start'], end=wake_ep.loc[0,'end'])
# ep1 = nts.IntervalSet(start=wake_ep.loc[1,'start'], end=wake_ep.loc[1,'end'])
# ep2 = nts.IntervalSet(start=wake_ep.loc[2,'start'], end=wake_ep.loc[2,'end'])

# path_plot(ep,position)

# all_rates = all_frate_maps(spikes, position,ep,spikes.keys(),24) 
# all_rates1 = all_frate_maps(spikes, position_,ep1,spikes.keys(),24)
# all_rates2 = all_frate_maps(spikes, position,ep2,spikes.keys(),24)



# pcs, stats = find_place_cells(all_rates,spikes, position, ep)
# pcs1, stats1 = find_place_cells(all_rates1,spikes, position_, ep1)
# pcs2, stats2 = find_place_cells(all_rates2,spikes, position, ep2)

# pc=list(set(pcs) & set(pcs1))


# ref = spikes.keys()
# nrows = int(sqrt(len(ref)))
# ncols = int(len(ref)/nrows)+1


# figure()
ep=nts.IntervalSet(start=wake_ep.loc[0,'start'], end=wake_ep.loc[0,'end'])
all_rates = all_frate_maps(spikes, position,ep,spikes.keys(),24) 
pcs_main, stats = find_place_cells(all_rates,spikes, position, ep)



conds = ['A','B']
# plt.figure()
for ij in range(len(wake_ep)):
    ep=nts.IntervalSet(start=wake_ep.loc[ij,'start'], end=wake_ep.loc[ij,'end'])
    all_rates = all_frate_maps(spikes, position,ep,spikes.keys(),24) 
    pcs, stats = find_place_cells(all_rates,spikes, position, ep)

    ref = all_rates.keys()
    nrows = int(sqrt(len(ref)))
    ncols = int(len(ref)/nrows)+1
    
    plt.figure()
    plt.suptitle(data_directory.split('\\')[-1]+' '+conds[ij])
    
    for i,ii in enumerate(ref):
        subplot(nrows,ncols,i+1)
        # mr=round(len(spikes[ii].restrict(ep))/ep.tot_length('s'),1)
        # info = round(si.iloc[ii].values[0],2)
        # spk_ct = len(spikes[ii].restrict(ep))
        # c = round(corr.iloc[ii,0],1)
        
        # shank_tmp = shank[ii]
        # cell_tmp = np.where(np.where(shank==shank_tmp)[0]==ii)[0][0]
        
        tmp_id = ii#f'S{shank_tmp+1}_C{cell_tmp+2}'
    
        
        imshow(all_rates[ii], cmap='jet')#gist_yarg
        title(tmp_id, size=8) if ii not in pcs else title(tmp_id, color='red',size=8)
        # ylabel(f'#{ii}') if ii not in pcs1 else ylabel(f'PC #{ii}', color='red')
    
        # gca().set_ylim(0,10)
        #title(f'{round(sel.loc[ii].values[0],1)}')
        gca().invert_yaxis()
        # remove_box(4)

    # plt.savefig(data_directory+'/fig'+str(ij)+'.jpg.', format='jpg', bbox_inches="tight", pad_inches=0.05)
    

























# tc_ka127 = {'tcs': all_rates,'pos': position,'spk': spikes, 'ep':ep, 'stats': stats,'ids':ids, 'note': "KA127-ca3-gnat-osn"}
# np.save(r"C:\Users\kasum\Dropbox\figs4Stu\pc_tcKA127_gnat-ctl.npy",tc_ka127)


# sys.exit()


figure()
for i,ii in enumerate(ref):
    subplot(nrows,ncols,i+1)
    scatter(position['x'].realign(spikes[ii].restrict(ep)),position['z'].realign(spikes[ii].restrict(ep)),s=0.1,c='red',label=str(ii),zorder=5)
    plot(position['x'].restrict(ep),position['z'].restrict(ep),color='darkgrey',linewidth=1) 
    ylabel(f'#{ii}') if ii not in pcs else ylabel(f'PC #{ii}', color='red')

    # info = round(si.iloc[ii].values[0],2)
    # title(info, y=-0.05, size=14, color='black')
#     remove_box(4)

# figure();hist(stats.loc[pcs,'info'])




# oc=occupancy_prob(position, ep)


# compute_PVcorrs(all_rates,all_rates1,pc)
# r=pc_sessionStability(spikes, position, wake_ep)

# figure(); hist(r.loc[pcs1,'spatial_corr'].values)




###############################################################################
## Linear Track
#to do: id PCs on linear track, i have to isolate dir before defining cells as HD or non-HD
###############################################################################

# position_=remove_immobility(position,speed_threshold = 0.06)


# pos= position.restrict(ep2)[['x','z']]
# clean_frames = pos.z.values < 0.2

# pos_tmp = pos [clean_frames]


# pos_x = gaussian_filter(pos_tmp.x.values, sigma=100)
# pos_y = gaussian_filter(pos_tmp.z.values, sigma=100)


# figure();plot(pos_tmp.x,pos_tmp.z)



# figure();plot(pos_x)

# # needs fine-tuning to generalize across animals
# # if stationary ts are removed from the original position data, the trajectories will be more seperable
# peaks,_= scipy.signal.find_peaks(pos_x, height = 0.3,distance=4000)
# troughs,_ = scipy.signal.find_peaks((pos_x)*-1, height= 0.26,distance=2500)

# peaks_ts = pos_tmp.index[peaks]
# troughs_ts = pos_tmp.index[troughs] 

# # rightwards
# r_run_ep = nts.IntervalSet(start=troughs_ts, end=peaks_ts)


# # leftwards
# l_run_ep = nts.IntervalSet(start=peaks_ts, end=troughs_ts[1:])



###############################################################################
## Plots by direction
###############################################################################
# rights = pos_tmp.restrict(r_run_ep)[['x','z']]
# lefts = pos_tmp.restrict(l_run_ep)[['x','z']]


# eps = l_run_ep
# figure()
# for i in range(10):
#     subplot(10,1,i+1)
#     ep_tmp = nts.IntervalSet(start=eps.loc[i].start,end=eps.loc[i].end ) 
#     xpos = pos_tmp.restrict(ep_tmp)['x']
#     ypos = pos_tmp.restrict(ep_tmp)['z']
#     plot(xpos,ypos)




# all_rates_dir = all_frate_maps(spikes, rights,r_run_ep,spikes.keys(),30) 

# figure()
# for i,x in enumerate([3,31,32,40,46,54,50,78,96,99,147]):
#     subplot(4,3,i+1)
#     imshow(all_rates_dir[x], cmap='jet')
#     remove_box(4)
#     # gca().set_ylim(0,10)



# all_rates_left = all_frate_maps(spikes, lefts,l_run_ep,spikes.keys(),30) 

# figure()
# for i,x in enumerate([3,31,32,40,46,54,50,78,96,99,147]):
#     subplot(4,3,i+1)
#     imshow(all_rates_left[x], cmap='jet')
#     remove_box(4)
#     gca().set_ylim(5,25)







# figure(); 
# scatter(np.arange(len(lefts.x.values)),lefts.x.values)
    

#Verify peak and trough selections
# figure();plot(pos_x)   

# for i in range(len(troughs)):
#     scatter(troughs[i],-0.23, c='k', s=20)

# for i in range(len(peaks)):
#     scatter(peaks[i],0.4, c='r', s=20)

    


##################################################################
##### SORTED LINEAR TRACK TCURVES
##################################################################

# df = pd.DataFrame(columns=['mx_idx'])
# for i,x in enumerate(pcs2):
#     # subplot(12,4,i+1)
#     frates = all_rates2[x][0].data
#     mx_fr = frates.max()
#     idx = np.where(frates == mx_fr)[0][0]
#     df.loc[x] = idx
# df_tmp =df.sort_values(['mx_idx'])

# figure();
# for i,c in enumerate(df_tmp.index):
#     subplot(72,1,i+1)
#     frates = all_rates2[c][0].data
#     plot(frates,color='red')
#     remove_box(4)

    
    
    
    
    
    
    
    

# np.where(all_rates2[0][0].data == all_rates2[0][0].data.max()) 
    
# shank[6]


