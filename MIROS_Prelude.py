# -*- coding: utf-8 -*-
"""
Created on Fri Aug,3 19:42:56 2018

@author: javimozo

Shell Prelude project

Reads netCDF data files from Datawell Waverider (DWR) buoy and ASCII data files
from MIROS SM-050 radar 
"""

from netCDF4 import Dataset, num2date
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import xarray as xr

# download files
DWR_files = glob.glob('C:/Users/admin/Desktop/javimozo/MIROS/Project/DF037&DWR/Datawell/NC/MON/Wave/*.nc')
SM_050_files = glob.glob('C:/Users/admin/Desktop/javimozo/MIROS/Project/DF037&DWR/DF037/WM/MON/*.DF037')
Wind_files = glob.glob('C:/Users/admin/Desktop/javimozo/MIROS/Project/DF037&DWR/DF037/WA/DAY/*.DF037')
Dir_files = glob.glob('C:/Users/admin/Desktop/javimozo/MIROS/Project/DF037&DWR/DF038/*.DF038')

# create empty DataFrames to initialize data concatenation
DWR_Params = pd.DataFrame({'Hs':[], 'ThetaM':[], 'ThetaP':[], 'Tp':[], 'Tz':[]})
SM_050_Params = pd.DataFrame({'Hs':[], 'Tp':[], 'Tz':[], 'ThetaP':[], 'ThetaM':[]})
Wind_Params = pd.DataFrame({'Wspeed10':[], 'Wspeed2':[], 'Wdir10':[], 'Wdir2':[]})
Dir_Params = pd.DataFrame({'AvgHeading':[], 'HeadingSpr':[]})

# download DWR data from DWR files
for DWR_file in DWR_files:

    DWR_data = Dataset(DWR_file, mode='r')
    #with Dataset(DWR[0], mode='r') as DWR_data:        # doesn't work for param..._DWR w/o [:]
    Hs_DWR = DWR_data.variables['Hs']
    Tp_DWR = DWR_data.variables['Tp']
    Tz_DWR = DWR_data.variables['Tz']
    ThetaM_DWR = DWR_data.variables['ThetaM']
    ThetaP_DWR = DWR_data.variables['ThetaP']
    Time_DWR = DWR_data.variables['Time']

    # create a dictionary with data from every parameter    
    data = {'Hs':Hs_DWR[:], 'Tp':Tp_DWR[:], 'Tz':Tz_DWR[:], 'ThetaM':ThetaM_DWR[:], 'ThetaP':ThetaP_DWR[:]}

    # transform time data from netCDF to datetime instance (TimeStamp)
    t = num2date(Time_DWR[:], Time_DWR.units)

    # list comprehension equivalent to
    # for j in range(len(t)):
    #    t[j] = t[j].replace(microsecond=0)
    # to eliminate dangling milliseconds in some TimeStamps        
    t = [j.replace(microsecond=0) for j in t]

    # link data and time (as index) in DataFrame
    DWR_params = pd.DataFrame(data, index=t)

    #concatenate DataFrame to previous (empty) generated ones
    DWR_Params = pd.concat([DWR_Params, DWR_params])

## 3 ways of filling gaps in final DataFrame
## reindexing (1)
#DWR_Params_reindex = DWR_Params.reindex(pd.date_range(DWR_Params.index[0], DWR_Params.index[-1], freq='30T'))
#
## resampling (2,3)
#DWR_Params_resamp_asfreq = DWR_Params.resample('30T').asfreq()
DWR_Params_asfreq = DWR_Params.asfreq('30T')

# download SM-050 data from SM-050 files
for SM_050_file in SM_050_files:
    
    with open(SM_050_file, 'r') as SM_050_data:
    #SM-050_data = open(SM_050_files[0], 'r')
        SM_050_params = pd.read_table(SM_050_data, skiprows=15, names=['Date','Time','Hs','Tp','Tz','ThetaP','ThetaM','Status'],
            usecols=(0,1,4,7,13,18,19,46), sep='\s+', parse_dates=[['Date', 'Time']], index_col=0 )
    #SM-050_data.close()
    
    ## Try to fin a method to directly extract names from "Parameters code row" in DF037 instead of creating "names" list
    
    # replace wrong data with NaNs according to Status    
    for i in range(len(SM_050_params)):
        if SM_050_params['Status'][i][1] != '1':
            SM_050_params.iloc[i,[0]] = np.nan
        if SM_050_params['Status'][i][4] != '1':
            SM_050_params.iloc[i,[1]] = np.nan
        if SM_050_params['Status'][i][10] != '1':
            SM_050_params.iloc[i,[2]] = np.nan
        if SM_050_params['Status'][i][15] != '1':
            SM_050_params.iloc[i,[3]] = np.nan
        if SM_050_params['Status'][i][16] != '1':
            SM_050_params.iloc[i,[4]] = np.nan
            
#        if SM_050_params['Status'][i][1,4,10,15,16] != '1':    # doesn't seem to work (doesn't discriminate individually - all or none)
#            SM_050_params.iloc[i,[0,1,2,3,4]] = np.nan
            
#    for i in SM_050_params['Index']:                           # give this a try to test .loc
#        if SM_050_params['Status'][i][1] != '1':
#            SM_050_params.loc[i,['Hs']] = np.nan
            
#    for i in range(len(SM_050_params)):
#        if SM_050_params['Status'][i][1] != '1':
#            SM_050_params['Hs'][i] = np.nan   ---->   'chained indexing' (to avoid) 
#                                                       see http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

    # remove Status column altogether
    SM_050_params.drop('Status', axis=1, inplace=True)
    
    #T_stamp = pd.date_range(start='2017-09-29 08:20:00', end='2017-09-30 23:59:00', freq='2T')
    #
    #comp_array = np.zeros([len(T_stamp),5],dtype=np.float) 
    #comp_array[:,:] = np.nan
    #
    #SM_050_comp = pd.DataFrame(comp_array, columns=['Hs','Tp','Tz','ThetaP','ThetaM'], index=T_stamp)

    #concatenate DataFrame to previous (empty) generated ones    
    SM_050_Params = pd.concat([SM_050_Params, SM_050_params])

## 3 ways of filling gaps in final DataFrame
## reindexing (1)
#SM_050_Params_reindex = SM_050_Params.reindex(pd.date_range(SM_050_Params.index[0], SM_050_Params.index[-1], freq='T'))
#
## upsampling (2,3)
#SM_050_Params_resamp_asfreq = SM_050_Params.resample('T').asfreq()
SM_050_Params_asfreq = SM_050_Params.asfreq('T')
#
## then downsampling data to match DWR data time step
#SM_050_Params_reindex_resamp = SM_050_Params_reindex.resample('30T', closed='right').mean()
#SM_050_Params_resamp_asfreq_resamp = SM_050_Params_resamp_asfreq.resample('30T', closed='right').mean()
SM_050_Params_asfreq_resamp = SM_050_Params_asfreq.resample('30T', closed='right').mean()

# download Wind data from Wind files
for Wind_file in Wind_files:

    with open(Wind_file, 'r') as Wind_data:

        Wind_params = pd.read_table(Wind_data, skiprows=16, names=['Date','Time','Wspeed10','Wspeed2','Wdir10','Wdir2','Status'],
            usecols=(0,1,3,4,5,6,11), dtype={'Status': object}, sep='\s+', parse_dates=[['Date', 'Time']], index_col=0 )

    # replace wrong data with NaNs according to Status    
    for i in range(len(Wind_params)):
        if Wind_params['Status'][i][0] != '0':
            Wind_params.iloc[i,[0]] = np.nan
        if Wind_params['Status'][i][1] != '0':
            Wind_params.iloc[i,[1]] = np.nan
        if Wind_params['Status'][i][2] != '0':
            Wind_params.iloc[i,[2]] = np.nan
        if Wind_params['Status'][i][3] != '0':
            Wind_params.iloc[i,[3]] = np.nan

    # remove Status column altogether
    Wind_params.drop('Status', axis=1, inplace=True)

    #concatenate DataFrame to previous (empty) generated ones    
    Wind_Params = pd.concat([Wind_Params, Wind_params])

# upsample data to fill gaps in final DataFrame
Wind_Params_asfreq = Wind_Params.asfreq('T')

# then downsample data to match DWR data time step
Wind_Params_asfreq_resamp = Wind_Params_asfreq.resample('30T', closed='right').mean()
#Wind_Params_resamp = Wind_Params.resample('30T', closed='right').mean()

# download Direction data from Direction files
for Dir_file in Dir_files:

    with open(Dir_file, 'r') as Dir_data:

        Dir_params = pd.read_table(Dir_data, skiprows=26, names=['Date', 'Time', 'AvgHeading', 'HeadingSpr'],
            usecols=(0,1,1373,1374), sep='\s+', parse_dates=[['Date', 'Time']], index_col=0 )

    #concatenate DataFrame to previous (empty) generated ones    
    Dir_Params = pd.concat([Dir_Params, Dir_params])

# upsample data to fill gaps in final DataFrame
Dir_Params_asfreq = Dir_Params.asfreq('T')

# then downsample data
Dir_Params_asfreq_resamp = Dir_Params_asfreq.resample('5T', closed='right').mean()

# comparison btw different methods (reindex, resample, asfreq) for DWR
#fig1, ax1 = plt.subplots(4, sharex=True)#, sharey=True)
## ax1[0].plot is fictitiously continuous at gaps in time series / discontinuous at NaNs 
#ax1[0].plot(DWR_Params['Hs'])
## ax1[1,2,3].plots are identical / discontinuous at gaps in time series and NaNs
#ax1[1].plot(DWR_Params_reindex['Hs'])
#ax1[2].plot(DWR_Params_resamp_asfreq['Hs'])
#ax1[3].plot(DWR_Params_asfreq['Hs'])
#fig1.subplots_adjust(hspace=0)
##ax[1].set_xticklabels(SM_050_Params_reindex['Index'], rotation=45, fontsize=10 )
#plt.xticks(rotation=25)
#ax1[0].set_title('Hs_DWR_Params vs. Hs_DWR_Params_reindex vs. Hs_DWR_Params_resamp_asfreq  vs. Hs_DWR_Params_asfreq')
##plt.savefig('Hs_DWR vs. Hs_MS_050.pdf')
##plt.show()
#
# comparison btw different methods (reindex, resample, asfreq) for SM-050
#fig2, ax2 = plt.subplots(4, sharex=True)
## ax2[0].plot is fictitiously continuous at gaps in time series / discontinuous at NaNs
#ax2[0].plot(SM_050_Params['Hs'])
## ax2[1,2,3].plots are identical / discontinuous at gaps in time series and NaNs
#ax2[1].plot(SM_050_Params_reindex_resamp['Hs'])
#ax2[2].plot(SM_050_Params_resamp_asfreq_resamp['Hs'])
#ax2[3].plot(SM_050_Params_asfreq_resamp['Hs'])
#fig2.subplots_adjust(hspace=0)
##ax[1].set_xticklabels(SM_050_Params_reindex['Index'], rotation=45, fontsize=10 )
#plt.xticks(rotation=25)
#ax2[0].set_title('Hs_SM_050_Params vs. Hs_SM_050_Params_reindex_resamp vs. Hs_SM_050_Params_resamp_asfreq_resamp vs. Hs_SM_050_Params_asfreq_resamp')
##plt.savefig('Hs_DWR vs. Hs_MS_050.pdf')
#plt.show()

#plot data
# [624:12430] ???
fig3, ax3 = plt.subplots(2, sharex=True)
fig3.subplots_adjust(hspace=0)
ax3[1].plot(DWR_Params_asfreq['Hs'], 'r-', label='DWR')
ax3[1].plot(SM_050_Params_asfreq_resamp['Hs'], 'b-', label='SM_050')
ax3[0].plot(Wind_Params_asfreq_resamp['Wspeed10'], 'k-', label='Wind Speed 10m')
ax3[1].set_xlabel('Time', size=14)
ax3[1].set_ylabel('Hs', size=14)
ax3[0].set_ylabel('Wspeed10', size=14)
ax3[1].legend(loc='upper left', shadow=False, fontsize='large')
ax3[0].legend(loc='upper left', shadow=False, fontsize='large')
ax3[0].set_title("DWR_Params_asfreq['Hs'] vs. SM_050_Params_asfreq_resamp['Hs'] vs Wind Speed 10m")
ax3[1].grid(True)
ax3[0].grid(True)
#fig3.tight_layout()
plt.xticks(rotation=25)
plt.show()

fig4, ax4 = plt.subplots(2, sharex=True)
fig4.subplots_adjust(hspace=0)
ax4[0].plot(Dir_Params_asfreq_resamp['AvgHeading'], 'm.', label='Average Heading')
ax4[1].plot(Dir_Params_asfreq_resamp['HeadingSpr'], 'g.', label='Heading Spread')
ax4[1].set_xlabel('Time', size=14)
ax4[0].set_ylabel('Average Heading', size=14)
ax4[1].set_ylabel('Heading Spread', size=14)
ax4[0].legend(loc='upper left', shadow=False, fontsize='large')
ax4[1].legend(loc='upper left', shadow=False, fontsize='large')
ax4[0].set_title("Average Heading vs. Heading Spread")
#ax4[1].set_yticklabels({'N': 0,'NE':45,'E':90,'SE':135,'S':180,'SW':225,'W':270,'NW':315})
ax4[0].grid(True)
ax4[1].grid(True)
#fig3.tight_layout()
plt.xticks(rotation=25)
plt.show()


## parameter value by timestamp
#SM_050_Params_asfreq_resamp['Hs'].loc['2018-04-16 23:00:00']
## row number by timestamp
#SM_050_Params_asfreq_resamp.index.get_loc('2018-04-16 23:00:00')
#SM_050_Params_asfreq_resamp['Hs'].index.get_loc('2018-04-16 23:00:00')


## QQ_plots (NaNs removed AFTER computing quantiles)
#
#fig4, ax4 = plt.subplots()
#
#DWR_Hs = DWR_Params_asfreq['Hs'].sort_values(ascending=False)
#SM_050_Hs = SM_050_Params_asfreq_resamp['Hs'].sort_values(ascending=False)
#
#quantile_levels_DWR = np.arange(len(DWR_Hs),dtype=float)/len(DWR_Hs)
#quantile_levels_SM_050 = np.arange(len(SM_050_Hs),dtype=float)/len(SM_050_Hs)
#
##if len(quantile_levels_DWR) > len(quantile_levels_SM_050):
##    quantile_levels = quantile_levels_SM_050
##elif len(quantile_levels_DWR) < len(quantile_levels_SM_050):
##    quantile_levels = quantile_levels_DWR
# 
#quantile_levels = quantile_levels_SM_050
#
#quantile_SM_050 = SM_050_Hs
#quantile_DWR = np.interp(quantile_levels, quantile_levels_DWR, DWR_Hs)
#
#ax4.plot(quantile_SM_050, quantile_DWR, 'b.')
#ax4.grid(True)
#
#k1 = np.isnan(SM_050_Hs)
#k2 = np.isnan(DWR_Hs)
#SM_050_Hs = SM_050_Hs[~k1]
#DWR_Hs = DWR_Hs[~k2]
#
#maxval = max(SM_050_Hs[-1],DWR_Hs[-1])
#minval = min(SM_050_Hs[0],DWR_Hs[0])
#ax4.plot([minval,maxval],[minval,maxval],'k-')
##ax4.plot([0,5],[0,5],'r-')
#
#plt.show()


# QQ_plots (NaNs removed BEFORE computing quantiles) 

fig5, ax5 = plt.subplots()

DWR_Hs = DWR_Params_asfreq['Hs'].sort_values(ascending=False)
SM_050_Hs = SM_050_Params_asfreq_resamp['Hs'].sort_values(ascending=False)

k1 = np.isnan(SM_050_Hs)
k2 = np.isnan(DWR_Hs)
SM_050_Hs = SM_050_Hs[~k1]
DWR_Hs = DWR_Hs[~k2]

quantile_levels_DWR = np.arange(len(DWR_Hs),dtype=float)/len(DWR_Hs)
quantile_levels_SM_050 = np.arange(len(SM_050_Hs),dtype=float)/len(SM_050_Hs)
 
quantile_levels = quantile_levels_SM_050

quantile_SM_050 = SM_050_Hs
quantile_DWR = np.interp(quantile_levels, quantile_levels_DWR, DWR_Hs)

ax5.plot(quantile_SM_050, quantile_DWR, 'b.', markersize=.5)
ax5.grid(True)

maxval = max(SM_050_Hs[-1],DWR_Hs[-1])
minval = min(SM_050_Hs[0],DWR_Hs[0])
ax5.plot([minval,maxval],[minval,maxval],'k-')
#ax4.plot([0,5],[0,5],'r-')

plt.show()

#------------------------
## Wind_Params[Wind_Params.index.duplicated()]       # checks for duplicated indices
#------------------------
