#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
wd = os.path.dirname(os.getcwd())
pd.options.display.max_columns = 50
pd.options.display.max_rows = 20


# In[2]:


# 1138 rows, 50 colomns
keyVariables = pd.read_csv(wd+'/dataset/Lakes/nla12_keyvariables_data.csv')
keyVariables.columns


# In[4]:


keyVariables.head()


# In[5]:


# MAN_MADE: 642; NATURAL: 496
# .groupby('CHLX_UNITS').nunique()
keyVariables =keyVariables[['UID', 'SITE_ID', 'VISIT_NO', 'INDEX_NLA', 'AREA_HA', 'COMID2012', 'ELEVATION',
              'LAKE_ORIGIN', 'CHLX_RESULT', 'INDEX_SITE_DEPTH', 'INDEX_LAT_DD', 'INDEX_LON_DD',
              'COLOR_RESULT','CHLORIDE_RESULT', 'NTL_RESULT', 'PTL_RESULT', 'TROPHIC_STATE']]
keyVariables.to_csv(wd+'/dataset/Lakes/keyVariables12.csv')


# In[6]:


# 2764 rows, 111 columns
siteInfo = pd.read_csv(wd+'/dataset/Lakes/nla2012_wide_siteinfo_08232016.csv')
siteInfo.columns


# In[7]:


# No index sites for INDXSAMP_CHEM or INDXSAMP_CHLA?
# .groupby('INDXSAMP_CHLA').nunique()
siteInfo = siteInfo[['SITE_ID', 'VISIT_NO', 'UID', 'DATE_COL','AREA_HA', 'CAT_UNIQUE',
          'CH0712_CAT', 'COMID2007', 'COMID2012', 'COMIDS2007', 'DES_FTYPE', 'ELEVATION', 'LAKE_ORIGIN', 'LAKE_ORIGIN12',
         'LAT_DD83', 'LON_DD83', 'PANEL', 'PERIM_KM', 'SITEID_07', 'SIZE_CLASS',
        'URBAN', 'INDEX_NLA', 'INDXSAMP_CHEM', 'INDXSAMP_CHLA', 'SAMPLED_PROFILE']]
# siteInfo.to_csv(wd+'/dataset/Lakes/siteInfo12.csv')


# In[8]:


# # There are 2664 unique site IDS
# len(siteInfo.SITE_ID.unique())

# # VISIT_NO: 1, 1286; 2, 100
# siteInfo.groupby('VISIT_NO').nunique()

# # A total of 1386 sites were visited?
# siteInfo.VISIT_NO.notna().sum()

# # A total of 1386 sites have UID?
# siteInfo.UID.notna().sum()

# # Where VISIT_NO is true, UID is ture
# siteInfo.loc[siteInfo.VISIT_NO.notna(),['UID']].notna().sum()
# # the Sum is 0, suggesting they are exactly the same
# (siteInfo.VISIT_NO.isna() == siteInfo.UID.notna()).sum()

# We should rm sites not visited; 1386 sites left
siteInfo = siteInfo.loc[siteInfo.UID.notna(),]

# # There are 1286 sites, among which 100 were visited twice
# len(siteInfo.SITE_ID.unique())

# # These UID are unique for each site visit
# len(siteInfo.UID.unique())

# # Save site types
# siteInfo[['SITE_ID', 'UID', 'LAKE_ORIGIN', 'LAKE_ORIGIN12']].groupby('LAKE_ORIGIN').nunique().to_csv(wd+'/output/table/siteTypes07.csv')
# siteInfo[['SITE_ID', 'UID', 'LAKE_ORIGIN', 'LAKE_ORIGIN12']].groupby('LAKE_ORIGIN12').nunique().to_csv(wd+'/output/table/siteTypes07.csv')

# Save the siteInfo
siteInfo.to_csv(wd+'/dataset/Lakes/siteInfo12.csv')


# In[9]:


# 1230 rows, with 233 columns
waterChem = pd.read_csv(wd+'/dataset/Lakes/nla2012_waterchem_wide.csv')
waterChem.columns


# In[10]:


# ALUMINUM_UNITS: mgL # AMMONIA_N_UNITS: mgNL # ANC_UNITS: ueqL # CALCIUM_UNITS: mgL # CHLORIDE_UNITS: mgL # COLOR_UNITS: APHA Pt-Co # COND_UNITS: uScm25C
# DOC_UNITS: mgL # MAGNESIUM_UNITS: mgL # NITRATE_N_UNITS: mgNL # NITRATE_NITRITE_N_UNITS: 1222 with mgNL, 7 with mgL # NITRITE_N_UNITS: mgNL # NTL_UNITS: mgL
# PH_UNITS: StdUnits # POTASSIUM_UNITS: mgL # PTL_UNITS: ugL # SILICA_UNITS: mgL # SODIUM_UNITS: mgL # SULFATE_UNITS: mgL # TOC_UNITS: mgL
# TSS_UNITS: mgL, 116 samples # TURB_UNITS: NTU, 1189 samples

# waterChem[['UID','SAM_CODE', 'COND_UNITS', 'DOC_UNITS', 'MAGNESIUM_UNITS', 'NITRATE_N_UNITS', 'NITRATE_NITRITE_N_UNITS', 'NITRITE_N_UNITS', 'NTL_UNITS', 'PH_UNITS',
#  'POTASSIUM_UNITS', 'PTL_UNITS', 'SILICA_UNITS', 'SODIUM_UNITS', 'SULFATE_UNITS', 'TOC_UNITS', 'TSS_UNITS', 'TURB_UNITS',]].groupby('TURB_UNITS').nunique()

# waterChem[['UID','SAM_CODE','ALUMINUM_LAB_FLAG']].groupby('ALUMINUM_LAB_FLAG').nunique()
waterChem = waterChem[['UID','SAM_CODE', 'CHEM_SAMPLE_ID', 'NUTS_SAMPLE_ID', 'ALUMINUM_QA_FLAG', 'ALUMINUM_RESULT', 'AMMONIA_N_QA_FLAG','AMMONIA_N_RESULT','ANC_QA_FLAG', 'ANC_RESULT',
          'CALCIUM_QA_FLAG', 'CALCIUM_RESULT','CHLORIDE_QA_FLAG', 'CHLORIDE_RESULT','COLOR_QA_FLAG', 'COLOR_RESULT','COND_QA_FLAG', 'COND_RESULT',
          'DOC_QA_FLAG', 'DOC_RESULT', 'MAGNESIUM_QA_FLAG', 'MAGNESIUM_RESULT', 'NITRATE_N_QA_FLAG', 'NITRATE_N_RESULT', 'NITRATE_NITRITE_N_QA_FLAG',
           'NITRATE_NITRITE_N_RESULT', 'NITRITE_N_QA_FLAG', 'NITRITE_N_RESULT', 'NTL_QA_FLAG', 'NTL_RESULT', 'PH_QA_FLAG', 'PH_RESULT', 'POTASSIUM_QA_FLAG',
           'POTASSIUM_RESULT', 'PTL_QA_FLAG', 'PTL_RESULT', 'SILICA_QA_FLAG', 'SILICA_RESULT', 'SODIUM_QA_FLAG', 'SODIUM_RESULT', 'SULFATE_QA_FLAG', 'SULFATE_RESULT', 'TSS_QA_FLAG', 
           'TSS_RESULT', 'TURB_QA_FLAG', 'TURB_RESULT']]

# # No measurement flags
# waterChem = \
# waterChem[['UID','SAM_CODE', 'CHEM_SAMPLE_ID', 'NUTS_SAMPLE_ID', 'ALUMINUM_RESULT', 'AMMONIA_N_RESULT', 'ANC_RESULT',
#           'CALCIUM_RESULT', 'CHLORIDE_RESULT', 'COLOR_RESULT', 'COND_RESULT',
#            'DOC_RESULT', 'MAGNESIUM_RESULT', 'NITRATE_N_RESULT', 
#            'NITRATE_NITRITE_N_RESULT', 'NITRITE_N_RESULT', 'NTL_RESULT', 'PH_RESULT',
#            'POTASSIUM_RESULT', 'PTL_RESULT', 'SILICA_RESULT', 'SODIUM_RESULT', 'SULFATE_RESULT',
#            'TSS_RESULT', 'TURB_RESULT']]


# In[12]:


# # The UID are unique for each observation
# len(waterChem.UID.unique()) == waterChem.shape[0]

# # All waterChem UID are in siteInfo UID so I can join the two based on UID
# len(waterChem.UID.isin(siteInfo.UID))

# # Check out sites that are not with waterChem
# siteInfo[['UID']].loc[~siteInfo.UID.isin(waterChem.UID)]

# Merge waterChem to siteInfo and save it as nlaWaterChem.csv
nlaWaterChem = siteInfo.merge(waterChem, on = "UID", how = 'left')
nlaWaterChem.to_csv(wd+'/dataset/Lakes/nlaWaterChem.csv')


# In[13]:


# 1230 rows, 34 columns
chlA = pd.read_csv(wd+'/dataset/Lakes/nla2012_chla_wide.csv')
chlA.columns


# In[14]:


# there is only 1 unit for chl a
# chlA[['UID', 'CHLL_RESULT','CHLL_UNITS','CHLX_RESULT','CHLX_UNITS']].groupby('CHLL_UNITS').nunique()
chlA = chlA[['UID', 'CHLL_RESULT','CHLX_RESULT']]
chlA.columns = ['UID', 'CHLL_ugL', 'CHLX_ugL']

# save the dataset
chlA.to_csv(wd+'/dataset/Lakes/chlA12.csv')


# # Compare and merge nlaWaterChem with chlA

# In[15]:


# # All chlA UID are in nlaWaterChem
# chlA.UID.isin(nlaWaterChem.UID).sum() == chlA.shape[0]

# # UID for chlA and waterChem are the same
# set(chlA.UID) == set(waterChem.UID)

# Merge chlA to nlaWaterChem
nlaWaterChem = nlaWaterChem.merge(chlA, how = 'left', on = 'UID')

# Save nlaWaterChem
nlaWaterChem.to_csv(wd+'/dataset/Lakes/nlaWaterChem.csv')


# # Checking UID for profile data

# In[2]:


# 30717 rows, 56 columns
# pd.read_csv(wd+'/dataset/Lakes/nla2012_wide_profile_08232016.csv', encoding = 'unicode_escape')
profile = pd.read_csv(wd+'/dataset/Lakes/nla2012_wide_profile_08232016.csv', encoding = 'ISO-8859-1')
profile.columns


# In[3]:


# Keep only useful columns
profile = profile[['UID', 'SITE_ID', 'DATE_COL', 'VISIT_NO', 'SAMPLE_TYPE', 'CONDUCTIVITY',
         'DEPTH', 'DO_BARO_PRESSURE', 'DO_CALIBRATION_UNITS', 'DO_DISPLAYED_UNITS', 'DO_DISPLAYED_VALUE', 'DO_ELEVATION', 
         'DO2_2M', 'INDEX_LAT_DD', 'INDEX_LON_DD', 'INDEX_SITE_DEPTH', 'OXYGEN', 'PH', 'TEMPERATURE']]

# profile.to_csv(wd+'/dataset/Lakes/profile07_12.csv')


# In[18]:


# # A total of 1131 sites
# len(set(profile.SITE_ID))

# # All sites are in siteInfo or nlaWaterChem
# set(profile.SITE_ID) - set(nlaWaterChem.SITE_ID)

# # A total of 156 sites not in profile
# len(set(siteInfo.SITE_ID) - set(profile.SITE_ID))

# # A total of 2523 UID
# len(set(profile.UID))

# # All chlA UID are in profile
# len(set(chlA.UID) - set(profile.UID))
# (set(chlA.UID) & set(profile.UID)) == set(chlA.UID)

# # # All chlA UID are in profile
# # len(set(chlA.UID) - set(profile.UID))
# (set(nlaWaterChem.UID) & set(profile.UID)) == set(nlaWaterChem.UID)

# # But profile contains 1293 UID that are not contained in chlA; These UID are likely from 2007 sampling campaigns
# len(set(profile.UID) - set(chlA.UID))
# # set(profile.UID) - set(chlA.UID)

# # There are sites from the 2007 survey
# for uid in set(profile.UID) - set(nlaWaterChem.UID):
#     print (uid)

# rm sites from the 2007 survey
profile = profile.loc[profile.UID.isin(nlaWaterChem.UID),]


# In[19]:


hisObj = plt.hist(profile[~profile.TEMPERATURE.isna()].DEPTH)
plt.xlabel(r'depth (m)')
print(r'Total number of observation is {}'.format(hisObj[0].sum().astype(int)))


# In[20]:


# Join DO2_2M
# A total of 1189 UID with DO2_2M
profile.DO2_2M.notna().sum()
# Merge DO2_2M to nlaWaterChem
nlaWaterChem = nlaWaterChem.merge(profile.loc[profile.DO2_2M.notna(),                                               ['UID', 'DO2_2M', 'INDEX_LAT_DD', 'INDEX_LON_DD', 'INDEX_SITE_DEPTH']],                                   how = 'left', on = 'UID')


# In[37]:


# Aggegrate OXYGEN, CONDUCTIVITY, PH, TEMPERATURE
profileAgg = profile.loc[profile.DEPTH == 0,].groupby('UID').agg(['mean'])
profileAgg.columns = profileAgg.columns.get_level_values(0)


# In[38]:


# profileAgg['UID'] = np.nan
# profileAgg['UID'] = profileAgg.index
profileAgg = profileAgg.reset_index()


# In[40]:


# Join OXYGEN, CONDUCTIVITY, PH, TEMPERATURE   
nlaWaterChem = nlaWaterChem.merge(profileAgg[['UID', 'CONDUCTIVITY', 'OXYGEN', 'PH', 'TEMPERATURE']],                                   how = 'left', on = 'UID')


# In[41]:


# Save nlaWaterChem
nlaWaterChem.to_csv(wd+'/dataset/Lakes/nlaWaterChem.csv')

