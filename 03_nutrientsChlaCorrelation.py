#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from IPython.display import display
pd.options.display.max_columns = 50
pd.options.display.max_rows = 20

wd = os.path.dirname(os.getcwd())


# In[2]:


nlaWaterChem = pd.read_csv(wd+'/dataset/Lakes/nlaWaterChem.csv')
nlaWaterChem = nlaWaterChem.drop('Unnamed: 0', axis = 1)


# In[3]:


def corrChl(df):
    dfNew = df[['AREA_HA', 'ELEVATION', 'AMMONIA_N_RESULT', 'ANC_RESULT', 'CALCIUM_RESULT', 'CHLORIDE_RESULT', 
              'COLOR_RESULT', 'COND_RESULT', 'DOC_RESULT', 'MAGNESIUM_RESULT', 'NITRATE_N_RESULT',
              'NITRATE_NITRITE_N_RESULT', 'NITRITE_N_RESULT', 'NTL_RESULT', 'PH_RESULT', 'POTASSIUM_RESULT', 
              'PTL_RESULT', 'SILICA_RESULT', 'SODIUM_RESULT', 'SULFATE_RESULT', 'TSS_RESULT',
              'TURB_RESULT', 'CHLL_ugL', 'CHLX_ugL', 'DO2_2M', 'CONDUCTIVITY', 'OXYGEN', 'PH', 'TEMPERATURE']]
    corrRes = pd.DataFrame(columns=dfNew.columns)
    for col in dfNew:
        if col != 'CHLX_ugL':
            newWaterChem = dfNew[['CHLX_ugL', col]]
            newWaterChem = newWaterChem.dropna()
#             print(col, newWaterChem.shape)
            pearsonReg = stats.pearsonr(newWaterChem['CHLX_ugL'], newWaterChem[col])
            corrRes.loc[1, col], corrRes.loc[2, col] = pearsonReg
    return(corrRes)


# In[4]:


corrChl(nlaWaterChem)


# In[5]:


corrChl(nlaWaterChem[nlaWaterChem.LAKE_ORIGIN == 'NATURAL'])


# In[6]:


corrChl(nlaWaterChem[nlaWaterChem.LAKE_ORIGIN == 'MAN_MADE'])


# In[7]:


corrChl(nlaWaterChem[nlaWaterChem.LAKE_ORIGIN12 == 'RESERVOIR'])

