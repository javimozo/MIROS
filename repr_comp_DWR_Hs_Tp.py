# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 12:52:25 2018

@author: admin
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from netCDF4 import Dataset, num2date
from scipy.stats import gaussian_kde, linregress
import statsmodels.api as sm
from astropy.stats.circstats import circcorrcoef, rayleightest, circmean, circvar


LPNF = 'T'
LPNF_lenght = 21
ELC = 'F'
ELC_timeConst = 0
ELC_thres = 0
ELC_age = 0

start_year = 2017
start_month = 8
start_day = 1
start_hour = 00
start_minute = 00
start_second = 00

end_year = 2018
end_month = 8
end_day = 31
end_hour = 23
end_minute = 59
end_second = 00

start_timedate = datetime(start_year, start_month, start_day, start_hour, start_minute, start_second)
start_time_date = start_timedate.strftime("%Y-%m-%d %H:%M:%S")
end_timedate = datetime(end_year, end_month, end_day, end_hour, end_minute, end_second)
end_time_date = end_timedate.strftime("%Y-%m-%d %H:%M:%S")
dato = start_timedate.strftime("%Y-%m")

file_path = 'C:/Users/admin/Desktop/javimozo/MIROS/Project/DF037&DWR'
#file_path = 'F:/MIROS/Project/DF037&DWR'

file_name = 'Reprocessed/{}.{}_{}.{}.{}.{}/DF037/WM/MON/*.DF037'.format(LPNF, LPNF_lenght, ELC, ELC_timeConst, ELC_thres, ELC_age)
file_name_raw = 'Reprocessed/Raw_F_F/DF037/WM/MON/*.DF037'
file_nc = 'Datawell/NC/MON/Wave/*.nc'

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

DWR_Params_asfreq = DWR_Params.asfreq('30T')
  

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

SM_050_Params_asfreq = SM_050_Params.asfreq('T')

SM_050_Params_asfreq_resamp_30 = SM_050_Params_asfreq.resample('30T', closed='right').mean()


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

SM_050_Params_raw_asfreq = SM_050_Params_raw.asfreq('T')

SM_050_Params_raw_asfreq_resamp_30 = SM_050_Params_raw_asfreq.resample('30T', closed='right').mean()

# Plotting
# -----------------------------------------------------------------
# timeSeries_plot

fig1, ax1 = plt.subplots(3, sharex=True)

fig1.subplots_adjust(hspace=0.1)
#fig1.tight_layout()

ax1[0].plot(SM_050_Params.loc[start_timedate:end_timedate]['Hs'], 'k:', markersize=2.5, label='SM-050 filtered data')
ax1[0].plot(SM_050_Params_raw.loc[start_timedate:end_timedate]['Hs_r'], 'b--', markersize=2.5, label='SM-050 non filtered data')
ax1[0].plot(DWR_Params.loc[start_timedate:end_timedate]['Hs'], 'g-', markersize=2.5, label='DWR data')
ax1[1].plot(SM_050_Params.loc[start_timedate:end_timedate]['Tp'], 'k:', markersize=2.5)
ax1[1].plot(SM_050_Params_raw.loc[start_timedate:end_timedate]['Tp_r'], 'b--', markersize=2.5)
ax1[1].plot(DWR_Params.loc[start_timedate:end_timedate]['Tp'], 'g.-', markersize=2.5)
ax1[2].plot(SM_050_Params.loc[start_timedate:end_timedate]['ThetaP'], 'k:', markersize=2.5)
ax1[2].plot(SM_050_Params_raw.loc[start_timedate:end_timedate]['ThetaP_r'], 'b--', markersize=2.5)
ax1[2].plot(DWR_Params.loc[start_timedate:end_timedate]['ThetaP'], 'g-', markersize=2.5)

#ax1[1].set_xlabel('Time', size=10)

ax1[0].set_ylabel('Hs [m]', size=10)
ax1[1].set_ylabel('Tp [s]', size=10)
ax1[1].yaxis.tick_right()
ax1[1].yaxis.set_label_position('right')
ax1[2].set_ylabel('ThetaP [°]', size=10)

ax1[0].legend(loc='upper right', shadow=False, fontsize='small')
#ax1[1].legend(loc='upper left', shadow=False, fontsize='small')
#ax1[2].legend(loc='upper left', shadow=False, fontsize='small')

ax1[0].set_title("LPN={}: Len={} _ ELC={}: Cons={}; Thres={}; Age={} _ Date={}".format(LPNF, LPNF_lenght, ELC, ELC_timeConst, ELC_thres, ELC_age, dato))

ax1[0].grid(True)
ax1[1].grid(True)
ax1[2].grid(True)
#fig3.tight_layout()
plt.xticks(rotation=25)
#plt.show()

#-------------------------

fig2, ax2 = plt.subplots(1, 2, sharex=True, sharey=True)
fig2.subplots_adjust(wspace=0.1)
#fig2.tight_layout()

# Scatter plot

SM_050_Hs = SM_050_Params.loc[start_timedate:end_timedate]['Hs']
SM_050_raw_Hs = SM_050_Params_raw.loc[start_timedate:end_timedate]['Hs_r']

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

# Colored by density --------------------

# Calculate the point density
xy = np.vstack([SM_050_Hs,SM_050_raw_Hs])
z = gaussian_kde(xy)(xy)

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
SM_050_Hs, SM_050_raw_Hs, z = SM_050_Hs[idx], SM_050_raw_Hs[idx], z[idx]

#plt.set_cmap('jet')
ax2[0].scatter(SM_050_Hs, SM_050_raw_Hs, c=z, s=25, marker='o', alpha=0.35, edgecolors='', cmap=plt.cm.jet)
#cb = plt.colorbar(s)
#cb.set_label('Colour bar')

#results = sm.OLS(SM_050_raw_Hs,sm.add_constant(SM_050_Hs)).fit()
#ax2[0].plot(SM_050_Hs, SM_050_Hs*results.params[1] + results.params[0], 'r:', linewidth=1)#, linestyle='dashed', linewidth=2)

slope, intercept, r_value, p_value, std_err = linregress(SM_050_Hs, SM_050_raw_Hs)
ax2[0].plot(SM_050_Hs, SM_050_Hs*slope + intercept, 'r--', linewidth=0.5)#(), label='R: '+str(r_value))
ax2[0].legend(['R: '+'{:06.4f}'.format(r_value)], loc='lower right', shadow=False, fontsize='small')
ax2[0].plot([minval,maxval],[minval,maxval],'k-')

ax2[0].grid(True)
ax2[0].set_xlabel('Hs [m]', size=10)
ax2[0].set_ylabel('Hs_raw [m]', size=10)

fig2.suptitle("LPN={}: Len={} _ ELC={}: Cons={}; Thres={}; Age={} _ Date={}".format(LPNF, LPNF_lenght, ELC, ELC_timeConst, ELC_thres, ELC_age, dato))


# QQ plot

quantile_levels_SM_050_Hs = np.arange(len(SM_050_Hs_sort),dtype=float)/len(SM_050_Hs_sort)
quantile_levels_SM_050_raw_Hs = np.arange(len(SM_050_raw_Hs_sort),dtype=float)/len(SM_050_raw_Hs_sort)
 
quantile_levels = quantile_levels_SM_050_raw_Hs

quantile_SM_050_raw_Hs = SM_050_raw_Hs_sort
#quantile_SM_050_Hs = np.interp(quantile_levels, quantile_levels_SM_050_Hs, SM_050_Hs_sort)
quantile_SM_050_Hs = SM_050_Hs_sort

ax2[1].plot(quantile_SM_050_Hs, quantile_SM_050_raw_Hs, 'b.', markersize=.5)
ax2[1].grid(True)

ax2[1].plot([minval,maxval],[minval,maxval],'k-')
ax2[1].set_xlabel('Hs [m]', size=10)
#ax2[1].set_ylabel('Hs_raw [m]', size=10)

#----------------------------------------------------------------------------

#fig3, ax3 = plt.subplots(1, 2, sharex=True, sharey=True)
#fig3.subplots_adjust(wspace=0.1)
##fig3.tight_layout()
#
## scatter_plot
#
#SM_050_Hs = SM_050_Params_asfreq_resamp_30.loc[start_timedate:end_timedate]['Hs']
#DWR_Hs = DWR_Params_asfreq.loc[start_timedate:end_timedate]['Hs']
#
#k1 = np.isnan(SM_050_Hs)
#k2 = np.isnan(DWR_Hs)
#SM_050_Hs = SM_050_Hs[~k1]
#SM_050_Hs = SM_050_Hs[~k2]
#DWR_Hs = DWR_Hs[~k2]
#DWR_Hs = DWR_Hs[~k1]
#
#SM_050_Hs_sort = SM_050_Hs.sort_values(ascending=False)
#DWR_Hs_sort = DWR_Hs.sort_values(ascending=False)
#
#maxval = max(SM_050_Hs_sort[-1],DWR_Hs_sort[-1])
#minval = min(SM_050_Hs_sort[0],DWR_Hs_sort[0])
#ax3[0].plot([minval,maxval],[minval,maxval],'k-')
#
## Colored by density --------------------
#
## Calculate the point density
#xy = np.vstack([SM_050_Hs,DWR_Hs])
#z = gaussian_kde(xy)(xy)
#
## Sort the points by density, so that the densest points are plotted last
#idx = z.argsort()
#SM_050_Hs, DWR_Hs, z = SM_050_Hs[idx], DWR_Hs[idx], z[idx]
#
#ax3[0].scatter(SM_050_Hs, DWR_Hs, c=z, s=25, marker='o', alpha=0.35, edgecolors='')
#
#slope, intercept, r_value, p_value, std_err = linregress(SM_050_Hs, DWR_Hs)
#ax3[0].plot(SM_050_Hs, SM_050_Hs*slope + intercept, 'r--', linewidth=0.5, label='R: '+str(r_value))
#ax3[0].legend(loc='lower right', shadow=False, fontsize='small')
#
#ax3[0].grid(True)
#ax3[0].set_xlabel('SM-050_Hs [m]', size=10)
#ax3[0].set_ylabel('DWR_Hs [m]', size=10)
#
#fig3.suptitle("LPN={}: Len={} _ ELC={}: Cons={}; Thres={}; Age={} _ Date={}".format(LPNF, LPNF_lenght, ELC, ELC_timeConst, ELC_thres, ELC_age, dato))
#
## QQ_plot
#
#quantile_levels_SM_050_Hs = np.arange(len(SM_050_Hs_sort),dtype=float)/len(SM_050_Hs_sort)
#quantile_levels_DWR_Hs = np.arange(len(DWR_Hs_sort),dtype=float)/len(DWR_Hs_sort)
# 
#quantile_levels = quantile_levels_DWR_Hs
#
#quantile_DWR_Hs = DWR_Hs_sort
##quantile_SM_050_Hs = np.interp(quantile_levels, quantile_levels_SM_050_Hs, SM_050_Hs_sort)
#quantile_SM_050_Hs = SM_050_Hs_sort
#
#ax3[1].plot(quantile_SM_050_Hs, quantile_DWR_Hs, 'b.', markersize=.5)
#ax3[1].grid(True)
#
#ax3[1].plot([minval,maxval],[minval,maxval],'k-')
#ax3[1].set_xlabel('SM-050_Hs', size=10)
##ax3[1].set_ylabel('DWR_Hs', size=10)

#----------------------------------------------------------------------------

fig4, ax4 = plt.subplots(1, 2, sharex=True, sharey=True)
fig4.subplots_adjust(wspace=0.1)
#fig4.tight_layout()

# scatter_plot

SM_050_Tp = SM_050_Params.loc[start_timedate:end_timedate]['Tp']
SM_050_raw_Tp = SM_050_Params_raw.loc[start_timedate:end_timedate]['Tp_r']

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

# Colored by density --------------------

# Calculate the point density
xy = np.vstack([SM_050_Tp,SM_050_raw_Tp])
z = gaussian_kde(xy)(xy)

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
SM_050_Tp, SM_050_raw_Tp, z = SM_050_Tp[idx], SM_050_raw_Tp[idx], z[idx]

ax4[0].scatter(SM_050_Tp, SM_050_raw_Tp, c=z, s=25, marker='o', alpha=0.35, edgecolors='')

slope, intercept, r_value, p_value, std_err = linregress(SM_050_Tp, SM_050_raw_Tp)
ax4[0].plot(SM_050_Tp, SM_050_Tp*slope + intercept, 'r--', linewidth=0.5)#, label='R: '+str(r_value))
ax4[0].legend(['R: '+'{:06.4f}'.format(r_value)], loc='lower right', shadow=False, fontsize='small')
ax4[0].plot([minval,maxval],[minval,maxval],'k-')

ax4[0].grid(True)
ax4[0].set_xlabel('Tp [s]', size=10)
ax4[0].set_ylabel('Tp_raw [s]', size=10)

fig4.suptitle("LPN={}: Len={} _ ELC={}: Cons={}; Thres={}; Age={} _ Date={}".format(LPNF, LPNF_lenght, ELC, ELC_timeConst, ELC_thres, ELC_age, dato))

# QQ plot

quantile_levels_SM_050_Tp = np.arange(len(SM_050_Tp_sort),dtype=float)/len(SM_050_Tp_sort)
quantile_levels_SM_050_raw_Tp = np.arange(len(SM_050_raw_Tp_sort),dtype=float)/len(SM_050_raw_Tp_sort)
 
quantile_levels = quantile_levels_SM_050_raw_Tp

quantile_SM_050_raw_Tp = SM_050_raw_Tp_sort
#quantile_SM_050_Tp = np.interp(quantile_levels, quantile_levels_SM_050_Tp, SM_050_Tp_sort)
quantile_SM_050_Tp = SM_050_Tp_sort

ax4[1].plot(quantile_SM_050_Tp, quantile_SM_050_raw_Tp, 'b.', markersize=.5)
ax4[1].grid(True)

ax4[1].plot([minval,maxval],[minval,maxval],'k-')
ax4[1].set_xlabel('Tp [s]', size=10)
#ax4[1].set_ylabel('Hs_raw [m]', size=10)

#----------------------------------------------------------------------------

#fig5, ax5 = plt.subplots(1, 2, sharex=True, sharey=True)
#fig5.subplots_adjust(wspace=0.1)
#fig5.tight_layout()
#
## scatter_plot
#
#SM_050_Tp = SM_050_Params_asfreq_resamp_30.loc[start_timedate:end_timedate]['Tp']
#DWR_Tp = DWR_Params_asfreq.loc[start_timedate:end_timedate]['Tp']
#
#k1 = np.isnan(SM_050_Tp)
#k2 = np.isnan(DWR_Tp)
#SM_050_Tp = SM_050_Tp[~k1]
#SM_050_Tp = SM_050_Tp[~k2]
#DWR_Tp = DWR_Tp[~k2]
#DWR_Tp = DWR_Tp[~k1]
#
#SM_050_Tp_sort = SM_050_Tp.sort_values(ascending=False)
#DWR_Tp_sort = DWR_Tp.sort_values(ascending=False)
#
#maxval = max(SM_050_Tp_sort[-1],DWR_Tp_sort[-1])
#minval = min(SM_050_Tp_sort[0],DWR_Tp_sort[0])
#ax5[0].plot([minval,maxval],[minval,maxval],'k-')
#
## Colored by density --------------------
#
## Calculate the point density
#xy = np.vstack([SM_050_Tp,DWR_Tp])
#z = gaussian_kde(xy)(xy)
#
## Sort the points by density, so that the densest points are plotted last
#idx = z.argsort()
#SM_050_Tp, DWR_Tp, z = SM_050_Tp[idx], DWR_Tp[idx], z[idx]
#
#ax5[0].scatter(SM_050_Tp, DWR_Tp, c=z, s=25, marker='o', alpha=0.35, edgecolors='')
#
#slope, intercept, r_value, p_value, std_err = linregress(SM_050_Tp, DWR_Tp)
#ax5[0].plot(SM_050_Tp, SM_050_Tp*slope + intercept, 'k--', linewidth=0.5, label='R: '+str(r_value))
#ax5[0].plot(SM_050_Tp, SM_050_Tp*slope + intercept, 'k--', linewidth=0.5, label='sigma: '+str(std_err))
#ax5[0].legend(loc='lower right', shadow=False, fontsize='small')
#
#ax5[0].grid(True)
#ax5[0].set_xlabel('SM-050_Tp [s]', size=10)
#ax5[0].set_ylabel('DWR_Tp [s]', size=10)
#
#fig5.suptitle("LPN={}: Len={} _ ELC={}: Cons={}; Thres={}; Age={} _ Date={}".format(LPNF, LPNF_lenght, ELC, ELC_timeConst, ELC_thres, ELC_age, dato))
#
## QQ_plot
#
#quantile_levels_SM_050_Tp = np.arange(len(SM_050_Tp_sort),dtype=float)/len(SM_050_Tp_sort)
#quantile_levels_DWR_Tp = np.arange(len(DWR_Tp_sort),dtype=float)/len(DWR_Tp_sort)
# 
#quantile_levels = quantile_levels_DWR_Tp
#
#quantile_DWR_Tp = DWR_Tp_sort
##quantile_SM_050_Tp = np.interp(quantile_levels, quantile_levels_SM_050_Tp, SM_050_Tp_sort)
#quantile_SM_050_Tp = SM_050_Tp_sort
#
#ax5[1].plot(quantile_SM_050_Tp, quantile_DWR_Tp, 'b.', markersize=.5)
#ax5[1].grid(True)
#
#ax5[1].plot([minval,maxval],[minval,maxval],'k-')
#ax5[1].set_xlabel('SM-050_Hs', size=10)
##ax3[1].set_ylabel('DWR_Hs', size=10)

#----------------------------------------------------------

fig6, ax6 = plt.subplots(1, 2, subplot_kw=dict(polar=True))
fig6.subplots_adjust(wspace=0.1)
##fig6.tight_layout()

# Circular histogram

SM_050_ThetaP = SM_050_Params.loc[start_timedate:end_timedate]['ThetaP']
SM_050_raw_ThetaP = SM_050_Params_raw.loc[start_timedate:end_timedate]['ThetaP_r']

k1 = np.isnan(SM_050_ThetaP)
k2 = np.isnan(SM_050_raw_ThetaP)
SM_050_ThetaP = SM_050_ThetaP[~k1]
#SM_050_ThetaP = SM_050_ThetaP[~k2]
SM_050_raw_ThetaP = SM_050_raw_ThetaP[~k2]
#SM_050_raw_ThetaP = SM_050_raw_ThetaP[~k1]

SM_050_ThetaP_round = round(SM_050_ThetaP)
SM_050_ThetaP_value_counts = SM_050_ThetaP_round.value_counts().reset_index().rename(columns={'ThetaP':'count', 'index':'ThetaP'})
#SM_050_ThetaP_value_counts_sort_values = SM_050_ThetaP_value_counts.sort_values(['ThetaP'], axis=0)

SM_050_raw_ThetaP_round = round(SM_050_raw_ThetaP)
SM_050_raw_ThetaP_value_counts = SM_050_raw_ThetaP_round.value_counts().reset_index().rename(columns={'ThetaP_r':'count', 'index':'ThetaP_r'})
#SM_050_raw_ThetaP_value_counts_sort_values = SM_050_raw_ThetaP_value_counts.sort_values(['ThetaP_r'], axis=0)

width1 = (2*np.pi) / len(SM_050_ThetaP_value_counts)
width2 = (2*np.pi) / len(SM_050_raw_ThetaP_value_counts)

SM_050_ThetaP_value_counts.loc[:,'ThetaP'] = np.deg2rad(SM_050_ThetaP_value_counts.loc[:,'ThetaP'])
SM_050_raw_ThetaP_value_counts.loc[:,'ThetaP_r'] = np.deg2rad(SM_050_raw_ThetaP_value_counts.loc[:,'ThetaP_r'])

#ax6.scatter(SM_050_ThetaP, SM_050_raw_ThetaP, c=z, s=25, marker='o', alpha=0.35, edgecolors='')

SM_050_ThetaP = SM_050_ThetaP[~k2]
SM_050_raw_ThetaP = SM_050_raw_ThetaP[~k1]

SM_050_ThetaP = np.deg2rad(SM_050_ThetaP)
SM_050_raw_ThetaP = np.deg2rad(SM_050_raw_ThetaP)

uniformity_SM_050_ThetaP = rayleightest(SM_050_ThetaP)
uniformity_SM_050_raw_ThetaP = rayleightest(SM_050_raw_ThetaP)

circ_r_SM_050_ThetaP_raw_ThetaP = circcorrcoef(SM_050_ThetaP, SM_050_raw_ThetaP)
circ_mean_SM_050_ThetaP = np.rad2deg(circmean(SM_050_ThetaP))
circ_mean_SM_050_raw_ThetaP = np.rad2deg(circmean(SM_050_raw_ThetaP))

if circ_mean_SM_050_ThetaP < 0:
    circ_mean_SM_050_ThetaP = circ_mean_SM_050_ThetaP + 360
if circ_mean_SM_050_raw_ThetaP < 0:
    circ_mean_SM_050_raw_ThetaP = circ_mean_SM_050_raw_ThetaP + 360

circ_std_SM_050_ThetaP = np.rad2deg(np.sqrt(circvar(SM_050_ThetaP)))
circ_std_SM_050_raw_ThetaP = np.rad2deg(np.sqrt(circvar(SM_050_raw_ThetaP)))

ax6[0].bar(SM_050_ThetaP_value_counts.loc[:,'ThetaP'], SM_050_ThetaP_value_counts.loc[:,'count'], width=width1, bottom=0)#, label='ThetaP')
ax6[0].legend(['ThetaP'], loc='lower right', shadow=False, fontsize='small')
ax6[0].set_theta_direction(-1)
#ax6[0].set_theta_offset(np.pi/2.0)
ax6[0].set_theta_zero_location('N', offset=0)
#ax6[0].set_title('mean: '+'{:06.4f}'.format(circ_mean_SM_050_ThetaP)+' _ '+'var: '+'{:06.4f}'.format(circ_var_SM_050_ThetaP))

txt1='mean: '+'{:06.4f}'.format(circ_mean_SM_050_ThetaP)+' _ '+'std: '+'{:06.4f}'.format(circ_std_SM_050_ThetaP)
plt.figtext(0.1, 0.01, txt1, wrap=True, horizontalalignment='center', fontsize=10)

ax6[1].bar(SM_050_raw_ThetaP_value_counts.loc[:,'ThetaP_r'], SM_050_raw_ThetaP_value_counts.loc[:,'count'], width=width2, bottom=0, label='ThetaP_raw')
ax6[1].legend(loc='lower right', shadow=False, fontsize='small')
ax6[1].set_theta_direction(-1)
#ax6[1].set_theta_offset(np.pi/2.0)
ax6[1].set_theta_zero_location('N', offset=0)

txt2='mean: '+'{:06.4f}'.format(circ_mean_SM_050_raw_ThetaP)+' _ '+'std: '+'{:06.4f}'.format(circ_std_SM_050_raw_ThetaP)
plt.figtext(0.9, 0.01, txt2, wrap=True, horizontalalignment='center', fontsize=10)

txt3='circ_r: '+'{:06.4f}'.format(circ_r_SM_050_ThetaP_raw_ThetaP)
plt.figtext(0.5, 0.01, txt3, wrap=True, horizontalalignment='center', fontsize=10)

fig6.suptitle("LPN={}: Len={} _ ELC={}: Cons={}; Thres={}; Age={} _ Date={}".format(LPNF, LPNF_lenght, ELC, ELC_timeConst, ELC_thres, ELC_age, dato))

#fig61, ax61 = plt.subplots()
##ax61.hist(SM_050_ThetaP_value_counts_sort_values.loc[:,'ThetaP'], SM_050_ThetaP_value_counts_sort_values.loc[:,'count']))
#SM_050_ThetaP_value_counts['count'].plot.hist(ax=ax61)

#ax6.legend(['circ_R: '+'{:06.4f}'.format(circ_r_SM_050_ThetaP_raw_ThetaP)], loc='lower left', shadow=False, fontsize='small')

# QQ plot

fig61, ax61 = plt.subplots()

SM_050_ThetaP = SM_050_Params.loc[start_timedate:end_timedate]['ThetaP']
SM_050_raw_ThetaP = SM_050_Params_raw.loc[start_timedate:end_timedate]['ThetaP_r']

k1 = np.isnan(SM_050_ThetaP)
k2 = np.isnan(SM_050_raw_ThetaP)
SM_050_ThetaP = SM_050_ThetaP[~k1]
#SM_050_ThetaP = SM_050_ThetaP[~k2]
SM_050_raw_ThetaP = SM_050_raw_ThetaP[~k2]
#SM_050_raw_ThetaP = SM_050_raw_ThetaP[~k1]

SM_050_ThetaP_sort = SM_050_ThetaP.sort_values(ascending=False)
SM_050_raw_ThetaP_sort = SM_050_raw_ThetaP.sort_values(ascending=False)

maxval = max(SM_050_ThetaP_sort[-1],SM_050_raw_ThetaP_sort[-1])
minval = min(SM_050_ThetaP_sort[0],SM_050_raw_ThetaP_sort[0])
ax61.plot([minval,maxval],[minval,maxval],'k-')

quantile_levels_SM_050_ThetaP = np.arange(len(SM_050_ThetaP_sort),dtype=float)/len(SM_050_ThetaP_sort)
quantile_levels_SM_050_raw_ThetaP = np.arange(len(SM_050_raw_ThetaP_sort),dtype=float)/len(SM_050_raw_ThetaP_sort)
 
quantile_levels = quantile_levels_SM_050_raw_ThetaP

quantile_SM_050_ThetaP = np.interp(quantile_levels, quantile_levels_SM_050_ThetaP, SM_050_ThetaP_sort)
#quantile_SM_050_ThetaP = SM_050_ThetaP_sort
quantile_SM_050_raw_ThetaP = SM_050_raw_ThetaP_sort

ax61.plot(quantile_SM_050_ThetaP, quantile_SM_050_raw_ThetaP, 'b.', markersize=.5)
ax61.grid(True)

ax61.plot([minval,maxval],[minval,maxval],'k-')
ax61.set_xlabel('ThetaP [°]', size=10)
ax61.set_ylabel('ThetaP_raw [°]', size=10)

fig61.suptitle("LPN={}: Len={} _ ELC={}: Cons={}; Thres={}; Age={} _ Date={}".format(LPNF, LPNF_lenght, ELC, ELC_timeConst, ELC_thres, ELC_age, dato))

#----------------------------------------------------------------------------

#fig7, ax7 = plt.subplots(1, 2, subplot_kw=dict(polar=True))
#fig7.subplots_adjust(wspace=0.1)
##fig4.tight_layout()
#
## scatter_plot
#
#SM_050_ThetaP = SM_050_Params_asfreq_resamp_30.loc[start_timedate:end_timedate]['ThetaP']
#DWR_ThetaP = DWR_Params_asfreq.loc[start_timedate:end_timedate]['ThetaP']
#
#k1 = np.isnan(SM_050_ThetaP)
#k2 = np.isnan(DWR_ThetaP)
#SM_050_ThetaP = SM_050_ThetaP[~k1]
##SM_050_ThetaP = SM_050_ThetaP[~k2]
#DWR_ThetaP = DWR_ThetaP[~k2]
##DWR_ThetaP = DWR_ThetaP[~k1]
#
#SM_050_ThetaP_round = round(SM_050_ThetaP)
#SM_050_ThetaP_value_counts = SM_050_ThetaP_round.value_counts().reset_index().rename(columns={'ThetaP':'count', 'index':'ThetaP'})
#
#DWR_ThetaP_round = round(DWR_ThetaP)
#DWR_ThetaP_value_counts = DWR_ThetaP_round.value_counts().reset_index().rename(columns={'ThetaP':'count', 'index':'ThetaP'})
#
#width3 = (2*np.pi) / len(SM_050_ThetaP_value_counts)
#width4 = (2*np.pi) / len(DWR_ThetaP_value_counts)
#
#SM_050_ThetaP_value_counts.loc[:,'ThetaP'] = np.deg2rad(SM_050_ThetaP_value_counts.loc[:,'ThetaP'])
#DWR_ThetaP_value_counts.loc[:,'ThetaP'] = np.deg2rad(DWR_ThetaP_value_counts.loc[:,'ThetaP'])
#
##ax7.scatter(SM_050_ThetaP, DWR_ThetaP, c=z, s=25, marker='o', alpha=0.35, edgecolors='')
#ax7[0].bar(SM_050_ThetaP_value_counts.loc[:,'ThetaP'], SM_050_ThetaP_value_counts.loc[:,'count'], width=width3, bottom=0)
#
#ax7[0].set_theta_direction(-1)
##ax7[0].set_theta_offset(np.pi/2.0)
#ax7[0].set_theta_zero_location('N', offset=0)
#
#ax7[1].bar(DWR_ThetaP_value_counts.loc[:,'ThetaP'], DWR_ThetaP_value_counts.loc[:,'count'], width=width4, bottom=0)
#
#ax7[1].set_theta_direction(-1)
##ax7[1].set_theta_offset(np.pi/2.0)
#ax7[1].set_theta_zero_location('N', offset=0)
#
#fig7.suptitle("LPN={}: Len={} _ ELC={}: Cons={}; Thres={}; Age={} _ Date={}".format(LPNF, LPNF_lenght, ELC, ELC_timeConst, ELC_thres, ELC_age, dato))
#
#uniformity_SM_050_ThetaP = rayleightest(SM_050_ThetaP)
#uniformity_DWR_ThetaP = rayleightest(DWR_ThetaP)
#
#SM_050_ThetaP = SM_050_ThetaP[~k2]
#DWR_ThetaP = DWR_ThetaP[~k1]
#
#uniformity_SM_050_ThetaP = rayleightest(SM_050_ThetaP)
#uniformity_DWR_ThetaP = rayleightest(DWR_ThetaP)
#
#SM_050_ThetaP = SM_050_ThetaP[~k2]
#DWR_ThetaP = DWR_ThetaP[~k1]
#
#circ_r_SM_050_ThetaP_DWR_ThetaP = circcorrcoef(SM_050_ThetaP, DWR_ThetaP)
#circ_mean_SM_050_ThetaP = circmean(SM_050_ThetaP)
#circ_var_SM_050_ThetaP = circvar(SM_050_ThetaP)
#circ_mean_DWR_ThetaP = circmean(DWR_ThetaP)
#circ_var_DWR_ThetaP = circvar(DWR_ThetaP)
#
#fig71, ax71 = plt.subplots()
#
#SM_050_ThetaP_sort = SM_050_ThetaP.sort_values(ascending=False)
#DWR_ThetaP_sort = DWR_ThetaP.sort_values(ascending=False)
#
#maxval = max(SM_050_ThetaP_sort[-1],SM_050_raw_ThetaP_sort[-1])
#minval = min(SM_050_ThetaP_sort[0],DWR_ThetaP_sort[0])
#ax71.plot([minval,maxval],[minval,maxval],'k-')
#
#quantile_levels_SM_050_ThetaP = np.arange(len(SM_050_ThetaP_sort),dtype=float)/len(SM_050_ThetaP_sort)
#quantile_levels_DWR_ThetaP = np.arange(len(DWR_ThetaP_sort),dtype=float)/len(DWR_ThetaP_sort)
# 
#quantile_levels = quantile_levels_DWR_ThetaP
#
##quantile_SM_050_Tp = np.interp(quantile_levels, quantile_levels_SM_050_ThetaP, SM_050_ThetaP_sort)
#quantile_SM_050_ThetaP = SM_050_ThetaP_sort
#quantile_DWR_ThetaP = DWR_ThetaP_sort
#
#ax71.plot(quantile_DWR_ThetaP, quantile_SM_050_ThetaP, 'b.', markersize=.5)
#ax71.grid(True)
#
#ax71.plot([minval,maxval],[minval,maxval],'k-')
#ax71.set_xlabel('SM-050_ThetaP [s]', size=10)
#ax71.set_ylabel('DWR_ThetaP [s]', size=10)

