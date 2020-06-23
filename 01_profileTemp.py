#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
wd = os.path.dirname(os.getcwd())
pd.options.display.max_columns = 50
pd.options.display.max_rows = 20


# In[3]:


# 30717 rows, 56 columns
# pd.read_csv(wd+'/dataset/Lakes/nla2012_wide_profile_08232016.csv', encoding = 'unicode_escape')
profile = pd.read_csv(wd+'/dataset/Lakes/nla2012_wide_profile_08232016.csv', encoding = 'ISO-8859-1')
profile.columns


# In[4]:


# Keep only useful columns
profile = profile[['UID', 'SITE_ID', 'DATE_COL', 'VISIT_NO', 'SAMPLE_TYPE', 'CONDUCTIVITY',
         'DEPTH', 'DO_BARO_PRESSURE', 'DO_CALIBRATION_UNITS', 'DO_DISPLAYED_UNITS', 'DO_DISPLAYED_VALUE', 'DO_ELEVATION', 
         'DO2_2M', 'INDEX_LAT_DD', 'INDEX_LON_DD', 'INDEX_SITE_DEPTH', 'OXYGEN', 'PH', 'TEMPERATURE']]


# In[44]:


profileDepth = profile[~profile.DEPTH.isna()]
numMeas = profileDepth[['UID', 'SITE_ID']].groupby('UID').count()
print(numMeas.SITE_ID.unique())
siteNumMeas2plus = numMeas[numMeas.SITE_ID > 2]
# filter profileDepth by number of measurements at each site
profileDepth = profileDepth.loc[profileDepth.UID.isin(siteNumMeas2plus.index),]


# In[57]:


fig, ax = plt.subplots()
ax.scatter(profileDepth[profileDepth.UID == 1000010].DEPTH, profileDepth[profileDepth.UID == 1000010].TEMPERATURE)


# In[156]:


def decreasingTemp(id):
    profileDepthSorted = profileDepth.loc[profileDepth.UID == id, ['DEPTH', 'TEMPERATURE']].sort_values('DEPTH')
    profileDepthSorted = profileDepthSorted[1:]
    decreasing = (np.diff(profileDepthSorted['TEMPERATURE']) < 0).sum()
    return(decreasing)


# In[153]:


# siteNumMeas2plus = siteNumMeas2plus.reset_index()
siteNumMeas2plus.loc[:,['decTemp']] = np.nan


# In[198]:


for index, series in siteNumMeas2plus.iterrows():
    siteNumMeas2plus.loc[row, ['decTemp']] = decreasingTemp(siteNumMeas2plus.loc[row, 'UID'])


# In[200]:


siteNumMeas2plus = siteNumMeas2plus[:-4]


# In[226]:


siteNumMeas2plus.loc[:,'nondecTemp'] = np.nan
siteNumMeas2plus.loc[:,['nondecTemp']] = siteNumMeas2plus['SITE_ID'] - siteNumMeas2plus['decTemp']
siteNumMeas2plus.loc[:,'percDecTemp'] = np.nan
siteNumMeas2plus.loc[:,['percDecTemp']] = siteNumMeas2plus.decTemp / siteNumMeas2plus.SITE_ID * 100


# In[232]:


siteNumMeas2plusDecTemp = siteNumMeas2plus[siteNumMeas2plus.percDecTemp >= 50]


# In[233]:


siteNumMeas2plusDecTemp.shape


# In[234]:


siteNumMeas2plusDecTemp.to_csv(wd+'/dataset/Lakes/siteNumMeas2plusDecTemp.csv')

