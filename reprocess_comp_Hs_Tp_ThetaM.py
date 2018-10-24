# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 10:29:35 2018

@author: admin
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date
from netCDF4 import Dataset, num2date
from scipy.stats import gaussian_kde

#SM_050_files = []

LPNF = 'T'
LPNF_lenght = 21
ELC = 'T'
ELC_timeConst = 10
ELC_thres = 5
ELC_age = 2

year = 2018
month = 5

date = date(year, month, 1)
dato = date.strftime("%Y%m")
mes = date.strftime("%m")

file_path = 'C:/Users/admin/Desktop/javimozo/MIROS/Project/DF037&DWR'
#file_path = 'F:/MIROS/Project/DF037&DWR'

file_name = 'Reprocessed/{}.{}_{}.{}.{}.{}/DF037/WM/MON/PLU_WM1_{}_MON.DF037'.format(LPNF, LPNF_lenght, ELC, ELC_timeConst, ELC_thres, ELC_age, dato)
file_name_raw = 'Reprocessed/Raw_F_F/DF037/WM/MON/PLU_WM1_{}_MON.DF037'.format(dato)
file_nc = 'Datawell/NC/MON/Wave/preludeDwr4Wave74071-rudics.{}.{}.v3.nc'.format(year, mes)

file_path_filter = os.path.join(file_path, file_name)
file_path_raw = os.path.join(file_path, file_name_raw)
file_path_nc = os.path.join(file_path, file_nc)

SM_050_files = glob.glob(file_path_filter)
SM_050_files_raw = glob.glob(file_path_raw)
DWR_files = glob.glob(file_path_nc)

SM_050_Params = pd.DataFrame({'Hs':[], 'Tp':[], 'Tz':[], 'ThetaP':[], 'ThetaM':[]})
SM_050_Params_raw = pd.DataFrame({'Hs_r':[], 'Tp_r':[], 'Tz_r':[], 'ThetaP_r':[], 'ThetaM_r':[]})
DWR_Params = pd.DataFrame({'Hs':[], 'ThetaM':[], 'ThetaP':[], 'Tp':[], 'Tz':[]})


for DWR_file in DWR_files:

    DWR_data = Dataset(DWR_file, mode='r')

    Hs_DWR = DWR_data.variables['Hs']
    Tp_DWR = DWR_data.variables['Tp']
    Tz_DWR = DWR_data.variables['Tz']
    ThetaM_DWR = DWR_data.variables['ThetaM']
    ThetaP_DWR = DWR_data.variables['ThetaP']
    Time_DWR = DWR_data.variables['Time']

    data = {'Hs':Hs_DWR[:], 'Tp':Tp_DWR[:], 'Tz':Tz_DWR[:], 'ThetaM':ThetaM_DWR[:], 'ThetaP':ThetaP_DWR[:]}

    t = num2date(Time_DWR[:], Time_DWR.units)

    t = [j.replace(microsecond=0) for j in t]

    DWR_params = pd.DataFrame(data, index=t)

    DWR_Params = pd.concat([DWR_Params, DWR_params], sort=True)


for SM_050_file in SM_050_files:
    
    with open(SM_050_file, 'r') as SM_050_data:

        SM_050_params = pd.read_table(SM_050_data, skiprows=15, names=['Date','Time','Hs','Tp','Tz','ThetaP','ThetaM','Status'],
            usecols=(0,1,4,7,13,18,19,46), sep='\s+', parse_dates=[['Date', 'Time']], index_col=0 )

    for i in SM_050_params.index:
        if SM_050_params['Status'][i][1] != '1':
            SM_050_params.loc[i,'Hs'] = np.nan
        if SM_050_params['Status'][i][4] != '1':
            SM_050_params.loc[i,'Tp'] = np.nan
        if SM_050_params['Status'][i][10] != '1':
            SM_050_params.loc[i,'Tz'] = np.nan
        if SM_050_params['Status'][i][15] != '1':
            SM_050_params.loc[i,'ThetaP'] = np.nan
        if SM_050_params['Status'][i][16] != '1':
            SM_050_params.loc[i,'ThetaM'] = np.nan

    SM_050_params.drop('Status', axis=1, inplace=True)

    SM_050_Params = pd.concat([SM_050_Params, SM_050_params])


for SM_050_file_raw in SM_050_files_raw:
    
    with open(SM_050_file_raw, 'r') as SM_050_data_raw:

        SM_050_params_raw = pd.read_table(SM_050_data_raw, skiprows=15, names=['Date_r','Time_r','Hs_r','Tp_r','Tz_r','ThetaP_r','ThetaM_r','Status_r'],
            usecols=(0,1,4,7,13,18,19,46), sep='\s+', parse_dates=[['Date_r', 'Time_r']], index_col=0 )

    for i in SM_050_params_raw.index:
        if SM_050_params_raw['Status_r'][i][1] != '1':
            SM_050_params_raw.loc[i,'Hs_r'] = np.nan
        if SM_050_params_raw['Status_r'][i][4] != '1':
            SM_050_params_raw.loc[i,'Tp_r'] = np.nan
        if SM_050_params_raw['Status_r'][i][10] != '1':
            SM_050_params_raw.loc[i,'Tz_r'] = np.nan
        if SM_050_params_raw['Status_r'][i][15] != '1':
            SM_050_params_raw.loc[i,'ThetaP_r'] = np.nan
        if SM_050_params_raw['Status_r'][i][16] != '1':
            SM_050_params_raw.loc[i,'ThetaM_r'] = np.nan

    SM_050_params_raw.drop('Status_r', axis=1, inplace=True)

    SM_050_Params_raw = pd.concat([SM_050_Params_raw, SM_050_params_raw])


# timeSeries_plot

fig1, ax1 = plt.subplots(3, sharex=True)

fig1.subplots_adjust(hspace=0)

ax1[0].plot(SM_050_Params['Hs'], 'k-', markersize=2.5, label='SM-050 filtered data')
ax1[0].plot(SM_050_Params_raw['Hs_r'], 'b-', markersize=2.5, label='SM-050 non filtered data')
ax1[0].plot(DWR_Params['Hs'], 'g-', markersize=2.5, label='DWR data')
ax1[1].plot(SM_050_Params['Tp'], 'k-', markersize=2.5)
ax1[1].plot(SM_050_Params_raw['Tp_r'], 'b-', markersize=2.5)
ax1[1].plot(DWR_Params['Tp'], 'g-', markersize=2.5)
ax1[2].plot(SM_050_Params['ThetaP'], 'k-', markersize=2.5)
ax1[2].plot(SM_050_Params_raw['ThetaP_r'], 'b-', markersize=2.5)
ax1[2].plot(DWR_Params['ThetaP'], 'g-', markersize=2.5)

ax1[1].set_xlabel('Time', size=10)

ax1[0].set_ylabel('Hs [m]', size=10)
ax1[1].set_ylabel('Tp [s]', size=10)
ax1[1].yaxis.tick_right()
ax1[1].yaxis.set_label_position('right')
ax1[2].set_ylabel('ThetaP [°]', size=10)

ax1[0].legend(loc='upper right', shadow=False, fontsize='small')
#ax1[1].legend(loc='upper left', shadow=False, fontsize='small')
#ax1[2].legend(loc='upper left', shadow=False, fontsize='small')

ax1[0].set_title("TimeSeries _ LPN={}: Len={} _ ELC={}: Cons={}; Thres={}; Age={} _ Date={}-{}".format(LPNF, LPNF_lenght, ELC, ELC_timeConst, ELC_thres, ELC_age, month, year))

ax1[0].grid(True)
ax1[1].grid(True)
ax1[2].grid(True)
#fig3.tight_layout()
plt.xticks(rotation=25)
plt.show()


# Scatter plots
#--------------

fig3, ax3 = plt.subplots()

SM_050_Hs = SM_050_Params['Hs']
SM_050_raw_Hs = SM_050_Params_raw['Hs_r']

k1 = np.isnan(SM_050_Hs)
k2 = np.isnan(SM_050_raw_Hs)
SM_050_Hs = SM_050_Hs[~k1]
SM_050_Hs = SM_050_Hs[~k2]
SM_050_raw_Hs = SM_050_raw_Hs[~k2]
SM_050_raw_Hs = SM_050_raw_Hs[~k1]

SM_050_Hs_sort = SM_050_Hs.sort_values(ascending=False)
SM_050_raw_Hs_sort = SM_050_raw_Hs.sort_values(ascending=False)

maxval = max(SM_050_Hs_sort[-1],SM_050_raw_Hs_sort[-1])
minval = min(SM_050_Hs_sort[0],SM_050_raw_Hs_sort[0])
ax3.plot([minval,maxval],[minval,maxval],'k-')

# Colored by density --------------------

# Calculate the point density
xy = np.vstack([SM_050_Hs,SM_050_raw_Hs])
z = gaussian_kde(xy)(xy)

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
SM_050_Hs, SM_050_raw_Hs, z = SM_050_Hs[idx], SM_050_raw_Hs[idx], z[idx]

ax3.scatter(SM_050_Hs, SM_050_raw_Hs, c=z, s=25, marker='o', alpha=0.35, edgecolors='', label='Filtered data')
ax3.grid(True)
ax3.set_xlabel('Hs [m]', size=10)
ax3.set_ylabel('Hs_raw [m]', size=10)

ax3.set_title("LPN={}: Len={} _ ELC={}: Cons={}; Thres={}; Age={} _ Date={}-{}".format(LPNF, LPNF_lenght, ELC, ELC_timeConst, ELC_thres, ELC_age, month, year))

#----------------------------------------------------------------------------

fig4, ax4 = plt.subplots()

SM_050_Tp = SM_050_Params['Tp']
SM_050_raw_Tp = SM_050_Params_raw['Tp_r']

k1 = np.isnan(SM_050_Tp)
k2 = np.isnan(SM_050_raw_Tp)
SM_050_Tp = SM_050_Tp[~k1]
SM_050_Tp = SM_050_Tp[~k2]
SM_050_raw_Tp = SM_050_raw_Tp[~k2]
SM_050_raw_Tp = SM_050_raw_Tp[~k1]

SM_050_Tp_sort = SM_050_Tp.sort_values(ascending=False)
SM_050_raw_Tp_sort = SM_050_raw_Tp.sort_values(ascending=False)

maxval = max(SM_050_Tp_sort[-1],SM_050_raw_Tp_sort[-1])
minval = min(SM_050_Tp_sort[0],SM_050_raw_Tp_sort[0])
ax4.plot([minval,maxval],[minval,maxval],'k-')

# Colored by density --------------------

# Calculate the point density
xy = np.vstack([SM_050_Tp,SM_050_raw_Tp])
z = gaussian_kde(xy)(xy)

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
SM_050_Tp, SM_050_raw_Tp, z = SM_050_Tp[idx], SM_050_raw_Tp[idx], z[idx]

ax4.scatter(SM_050_Tp, SM_050_raw_Tp, c=z, s=25, marker='o', alpha=0.35, edgecolors='')
ax4.grid(True)
ax4.set_xlabel('Tp [s]', size=10)
ax4.set_ylabel('Tp_raw [s]', size=10)

ax4.set_title("LPN={}: Len={} _ ELC={}: Cons={}; Thres={}; Age={} _ Date={}-{}".format(LPNF, LPNF_lenght, ELC, ELC_timeConst, ELC_thres, ELC_age, month, year))

#fig5, ax5 = plt.subplots()
#
#ax5.plot(SM_050_Params['ThetaP'], SM_050_Params_raw['ThetaP_r'], 'k.', markersize=2.5)
#ax5.grid(True)
#ax5.set_xlabel('ThetaP [°]', size=10)
#ax5.set_ylabel('ThetaP_raw [°]', size=10)

#ax3[0].legend(loc='upper right', shadow=False, fontsize='small')

#plt.show()


# QQ_plots (NaNs removed BEFORE computing quantiles) 
#---------------------------------------------------

#fig2, ax2 = plt.subplots(1, 3)
#fig2.tight_layout()
#
##-- Hs ---------------
#
#SM_050_Hs = SM_050_Params['Hs'].sort_values(ascending=False)
#SM_050_raw_Hs = SM_050_Params_raw['Hs_r'].sort_values(ascending=False)
#
#k1 = np.isnan(SM_050_Hs)
#k2 = np.isnan(SM_050_raw_Hs)
#SM_050_Hs = SM_050_Hs[~k1]
#SM_050_raw_Hs = SM_050_raw_Hs[~k2]
#
##k1 = np.isnan(SM_050_Hs)
#k2 = np.isnan(SM_050_raw_Hs)
#SM_050_Hs = SM_050_Hs[~k2]
#SM_050_raw_Hs = SM_050_raw_Hs[~k2]

#
#quantile_levels_SM_050_Hs = np.arange(len(SM_050_Hs),dtype=float)/len(SM_050_Hs)
#quantile_levels_SM_050_raw_Hs = np.arange(len(SM_050_raw_Hs),dtype=float)/len(SM_050_raw_Hs)
# 
#quantile_levels = quantile_levels_SM_050_raw_Hs
#
#quantile_SM_050_raw_Hs = SM_050_raw_Hs
#quantile_SM_050_Hs = np.interp(quantile_levels, quantile_levels_SM_050_Hs, SM_050_Hs)
#
#ax2[0].plot(quantile_SM_050_raw_Hs, quantile_SM_050_Hs, 'b.', markersize=.5)
#ax2[0].grid(True)
#
#maxval = max(SM_050_Hs[-1],SM_050_raw_Hs[-1])
#minval = min(SM_050_Hs[0],SM_050_raw_Hs[0])
#ax2[0].plot([minval,maxval],[minval,maxval],'k-')
##ax4.plot([0,5],[0,5],'r-')
#ax2[0].set_xlabel('Hs', size=12)
#ax2[0].set_ylabel('raw_Hs', size=12)
##ax2.set_title("QQ: SM-050_Hs vs. SM_050_raw_Hs")
#
##-- Tp ---------------
#
#SM_050_Tp = SM_050_Params['Tp'].sort_values(ascending=False)
#SM_050_raw_Tp = SM_050_Params_raw['Tp_r'].sort_values(ascending=False)
#
#k1 = np.isnan(SM_050_Tp)
#k2 = np.isnan(SM_050_raw_Tp)
#SM_050_Tp = SM_050_Tp[~k1]
#SM_050_raw_Tp = SM_050_raw_Tp[~k2]
#
#quantile_levels_SM_050_Tp = np.arange(len(SM_050_Tp),dtype=float)/len(SM_050_Tp)
#quantile_levels_SM_050_raw_Tp = np.arange(len(SM_050_raw_Tp),dtype=float)/len(SM_050_raw_Tp)
# 
#quantile_levels = quantile_levels_SM_050_raw_Tp
#
#quantile_SM_050_raw_Tp = SM_050_raw_Tp
#quantile_SM_050_Tp = np.interp(quantile_levels, quantile_levels_SM_050_Tp, SM_050_Tp)
#
#ax2[1].plot(quantile_SM_050_raw_Tp, quantile_SM_050_Tp, 'm.', markersize=.5)
#ax2[1].grid(True)
#
#maxval = max(SM_050_Tp[-1],SM_050_raw_Tp[-1])
#minval = min(SM_050_Tp[0],SM_050_raw_Tp[0])
#ax2[1].plot([minval,maxval],[minval,maxval],'k-')
##ax4.plot([0,5],[0,5],'r-')
#ax2[1].set_xlabel('Tp', size=12)
#ax2[1].set_ylabel('raw_Tp', size=12)
#ax2[1].set_title("QQ_plot _ SM_050 _ LPN={}: Len={} _ ELC={}: Cons={}; Thres={}; Age={} _ Date={}-{}".format(LPNF, LPNF_lenght, ELC, ELC_timeConst, ELC_thres, ELC_age, month, year), size=12)
#
##-- ThetaP ----------------
#
#SM_050_ThetaP = SM_050_Params['ThetaP'].sort_values(ascending=False)
#SM_050_raw_ThetaP = SM_050_Params_raw['ThetaP_r'].sort_values(ascending=False)
#
#k1 = np.isnan(SM_050_ThetaP)
#k2 = np.isnan(SM_050_raw_ThetaP)
#SM_050_ThetaP = SM_050_ThetaP[~k1]
#SM_050_raw_ThetaP = SM_050_raw_ThetaP[~k2]
#
#quantile_levels_SM_050_ThetaP = np.arange(len(SM_050_ThetaP),dtype=float)/len(SM_050_ThetaP)
#quantile_levels_SM_050_raw_ThetaP = np.arange(len(SM_050_raw_ThetaP),dtype=float)/len(SM_050_raw_ThetaP)
# 
#quantile_levels = quantile_levels_SM_050_raw_ThetaP
#
#quantile_SM_050_raw_ThetaP = SM_050_raw_ThetaP
#quantile_SM_050_ThetaP = np.interp(quantile_levels, quantile_levels_SM_050_ThetaP, SM_050_ThetaP)
#
#ax2[2].plot(quantile_SM_050_raw_ThetaP, quantile_SM_050_ThetaP, 'g.', markersize=.5)
#ax2[2].grid(True)
#
#maxval = max(SM_050_ThetaP[-1],SM_050_raw_ThetaP[-1])
#minval = min(SM_050_ThetaP[0],SM_050_raw_ThetaP[0])
#ax2[2].plot([minval,maxval],[minval,maxval],'k-')
##ax4.plot([0,5],[0,5],'r-')
#ax2[2].set_xlabel('ThetaP', size=12)
#ax2[2].set_ylabel('raw_ThetaP', size=12)
#
#plt.show()


