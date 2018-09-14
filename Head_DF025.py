# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 16:08:36 2018

@author: admin
"""

import glob
import numpy as np
import pandas as pd
from numba import jit
import matplotlib.pyplot as plt
import time

@jit
def angle_diff(unit1, unit2):
    phi = abs(unit2-unit1) % 360
    sign = 1
    # used to calculate sign
    if not ((unit1-unit2 >= 0 and unit1-unit2 <= 180) or (
            unit1-unit2 <= -180 and unit1-unit2 >= -360)):
        sign = -1
    if phi > 180:
        result = 360-phi
    else:
        result = phi

    return result*sign


Heading_files = glob.glob('C:/Users/admin/Desktop/javimozo/MIROS/Project/DF037&DWR/DF025/*.DF025')
Heading_Params = pd.DataFrame({'Heading':[], 'HeadingMaxSpr':[]})

start_time = time.clock()

for Heading_file in Heading_files:

    with open(Heading_file, 'r') as Heading_data:

        Heading_serie = pd.read_table(Heading_data, header=None)

    # empty lists
    DateTime = []       
    Heading = []
    HeadingMaxSpr = []

    for i in range(int(len(Heading_serie)/75)):
    #    date = Heading_serie.iloc[3+75*i,[0]][0]
    #    time = Heading_serie.iloc[4+75*i,[0]][0]
        date_time = Heading_serie.iloc[3+75*i,[0]][0] + ' ' + Heading_serie.iloc[4+75*i,[0]][0]
        DateTime.append(pd.to_datetime(date_time, format='%d-%m-%Y %H:%M'))
        if Heading_serie.iloc[6+75*i,[0]][0][17:20] == '999':
            Heading.append(np.nan)
            HeadingMaxSpr.append(np.nan)
        else:
            Heading.append(float(Heading_serie.iloc[6+75*i,[0]][0][17:20]))
            HeadingMaxSpr.append(float(Heading_serie.iloc[6+75*i,[0]][0][21:24]))

    # create a dictionary with data from every parameter    
    data = {'Heading':Heading[:], 'HeadingMaxSpr':HeadingMaxSpr[:]}

    # create a dataframe where to include data as it is extracted 
    Heading_params = pd.DataFrame(data, index=DateTime)

    #concatenate DataFrame to previous (empty) generated ones    
    Heading_Params = pd.concat([Heading_Params, Heading_params])

# upsample data to fill gaps in final DataFrame
Heading_Params_asfreq = Heading_Params.asfreq('T')

end_time = time.clock()
print ('time elapsed: ', (end_time - start_time))

## then downsample data
#Heading_Params_asfreq_resamp_5 = Heading_Params_asfreq.resample('5T', closed='right').mean()
#Heading_Params_resamp_2_5 = Heading_Params.resample('2.5T', closed='right').mean()

#@jit
# def SECT45(Heading_Params_asfreq):

start_time2 = time.clock()

MAX_ = []
MIN_ = []
MAX_MIN_ = []
MAX_MIN_index =[]
# how to extract 45 min sections
for i in Heading_Params_asfreq.index:
#    print(Heading_Params_asfreq.loc[i,['Heading']])
    # i - timeDelta(45 m)
#timeStep2 = Heading_Params_asfreq.loc[Heading_Params_asfreq.index[0],['Heading']]
#timeDelta = pd.Timedelta(minutes=45)
#timeStep1 = timeStep2 - timeDelta
#    gap = pd.date_range(end=Heading_Params_asfreq.index[0], periods=45, freq='1T')
    gap = pd.date_range(end=i, periods=45, freq='1T')
#    try:
    sect45 = Heading_Params_asfreq.loc[gap,['Heading']]

    # extract a 45 m section
#        sect45 = Heading_Params_asfreq.loc[Heading_Params_asfreq.index[209000:209045],['Heading']]
    # create a square 2D array with equal columns (and its transpose)
    x = np.array(sect45)
    xx = np.ones([1,len(x)])*x
    yy = xx.T
    # substract angles 
    # xx(t1) - yy(t2) -> angle differences for whole interval and everything in between 
    # truly represents how the angle evolves in time 
    # that's what the function actually does
    ang_diff_vect = np.vectorize(angle_diff)
    adv_np = ang_diff_vect(xx,yy)
    # convert to an upper triangular array
    low_tri = np.tril_indices(45,k=-1)
    adv_np[low_tri] = np.nan
    # DON'T REALLY NEED THE DATA FRAME, JUST THE MAX WITHIN THE ARRAY
    # convert again to dataFrame to conserve timeStamp
#    adv_np_df = pd.DataFrame(adv_np, index=sect45.index, columns=sect45.index)
    # remove NaNs
#    adv_np_df_k = adv_np_df.dropna(axis=0,how='all')
#    adv_np_df_kk = adv_np_df_k.dropna(axis=1,how='all')
    
    # verify if there's any value above threshold
    #mask = np.any(adv_np_df > abs(10))
    # and which is that value (+ and -)
    try:
        max_ = np.nanmax(adv_np)
        min_ = np.nanmin(adv_np)
        
        if abs(max_) > abs(min_):
            MAX_.append(max_)
            MIN_.append(np.nan)
            MAX_MIN_.append(abs(max_))
            MAX_MIN_index.append(i)
            
        elif abs(max_) < abs(min_):
            MIN_.append(min_)
            MAX_.append(np.nan)
            MAX_MIN_.append(abs(min_))
            MAX_MIN_index.append(i)
        
        else:
            MAX_.append(max_)
            MIN_.append(min_)
            MAX_MIN_.append(abs(max_))
            MAX_MIN_index.append(i)

    except ValueError:
         MAX_.append(np.nan)
         MIN_.append(np.nan)
         MAX_MIN_index.append(i)
#    except KeyError:
#        pass

#   return MAX_MIN
        
#SECT45(Heading_Params_asfreq)
# IT IS HERE THAT I NEED TO RECONSTRUCT THE DATA FRAME
data = {'MAX':MAX_, 'MIN':MIN_, 'MAX_MIN':MAX_MIN_}         
RoT = pd.DataFrame(data, index=MAX_MIN_index)
# AND THEN REMOVE NaNs 

end_time2 = time.clock()
print ('time elapsed: ', (end_time2 - start_time2))

fig2, ax2 = plt.subplots(3, sharex=True)
fig2.subplots_adjust(hspace=0)
ax2[2].plot(RoT['MAX_MIN'], 'g.', markersize=2.5, label='Abs Max Turning Angle')
ax2[1].plot(RoT['MAX'], 'b.', markersize=2.5, label='Max Anticlockwise Turning Angle')
ax2[1].plot(RoT['MIN'], 'r.', markersize=2.5, label='Max Clockwise Turning Angle')
ax2[0].plot(Heading_Params_asfreq['Heading'], 'm.', markersize=2.5, label='Heading')
ax2[0].set_xlabel('Time', size=10)
ax2[2].set_ylabel('Degrees', size=10)
ax2[1].set_ylabel('Degrees', size=10)
ax2[1].yaxis.tick_right()
ax2[1].yaxis.set_label_position('right')
ax2[0].set_ylabel('Degrees', size=10)
ax2[2].legend(loc='upper left', shadow=False, fontsize='small')
ax2[1].legend(loc='upper left', shadow=False, fontsize='small')
ax2[0].legend(loc='upper left', shadow=False, fontsize='small')
ax2[0].set_title("Heading / Max Turning Angle / Abs Max Turning Angle")
ax2[2].axhline(10, linestyle='--', color='k', linewidth=1)
ax2[1].axhline(-10, linestyle='--', color='k', linewidth=1)
ax2[1].axhline(10, linestyle='--', color='k', linewidth=1)
ax2[2].grid(True)
ax2[1].grid(True)
ax2[0].grid(True)
#fig3.tight_layout()
plt.xticks(rotation=25)
plt.show()

fig4, ax4 = plt.subplots(2, sharex=True)
fig4.subplots_adjust(hspace=0)
ax4[0].plot(Heading_Params_asfreq['Heading'], 'm.', label='Heading')
ax4[1].plot(Heading_Params_asfreq['HeadingMaxSpr'], 'g.', label='Heading Max Spread')
ax4[1].set_xlabel('Time', size=14)
ax4[0].set_ylabel('Average Heading', size=14)
ax4[1].set_ylabel('Heading Spread', size=14)
#ax4[0].legend(loc='upper left', shadow=False, fontsize='large')
#ax4[1].legend(loc='upper left', shadow=False, fontsize='large')
ax4[0].set_title("Average Heading vs. Heading Spread")
#ax4[1].set_yticklabels({'N': 0,'NE':45,'E':90,'SE':135,'S':180,'SW':225,'W':270,'NW':315})
ax4[0].grid(True)
ax4[1].grid(True)
#fig3.tight_layout()
plt.xticks(rotation=25)
plt.show()

# -----------------------------------------------------------------------------
## transforming the 2D arrays to DataFrame and then applying ang_diff results 
## in a 2D array again that has to be transformed again to DataFrame
## REDUNDANT !!!
#x_df = pd.DataFrame(xx, index=sect45.index)
#y_df = x_df.T
#adv_df = ang_diff_vect(x_df,y_df)
#adv_df_df = pd.DataFrame(adv_df, index=sect45.index, columns=sect45.index)
#adv_df_df_k = adv_df_df.dropna(axis=0,how='all')
#adv_df_df_kk = adv_df_df_k.dropna(axis=1,how='all')
# -----------------------------------------------------------------------------    
## locate those values in new dataFrame and extract them with interval (initial 
## and final timeStamps) attached
## --- GET INTERVAL ---> TimeDelta
#i, j = np.where(adv_np_df > abs(10))
#np_vals = adv_np_df.values
#vals_plus10 = pd.Series(np_vals[i, j], [adv_np_df.index[i], adv_np_df.columns[j]])
#
## determine the longest interval in order to mark it on the plot
## all the individual time steps of every interval can be tracked down from
## the 2D array data (1st diagonal above the main one)
#
#timedelta = vals_plus10.index[5][1] - vals_plus10.index[5][0]
#
#timedelta_total_m = timedelta.total_seconds()/60
#
#rate = vals_plus10[14]/timedelta_total_m
#
#
#diag_1 = np.diag(adv_np_df_kk, k=1)
#
#
#np_vals = adv_np_df.values
#ii, jj = np.triu_indices_from(np_vals, 1)
#diag = pd.Series(np_vals[ii, jj], [adv_np_df.index[ii], adv_np_df.columns[jj]])
#
##triu_ind = np.triu_indices(45,1)
##adv_triu = adv_np[triu_ind]
#
#diag_k = diag.dropna(axis=0,how='all')
#adv_np_df_kk = adv_np_df_k.dropna(axis=1,how='all')
