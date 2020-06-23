#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
pd.options.display.max_columns = 50
pd.options.display.max_rows = 20
from scipy import stats
import matplotlib as mpl
wd = os.path.dirname(os.getcwd())


# In[2]:


# set title
mpl.rc('figure', titlesize = 45, titleweight = 'bold')
# set axes
mpl.rc('axes', titlesize = 40, titleweight = 'bold', titlepad = 20,
       facecolor = 'white', edgecolor = 'black',
       linewidth = 2, 
       labelweight = 'normal', labelsize = 40, labelcolor = 'black')
# set lines
mpl.rc('lines', linewidth = 1.5, color = 'red')
# set font
font = {'family': 'Arial',
       'size': 35}
mpl.rc('font', **font)
# set x tickes
mpl.rc('xtick.major', width = 2, size = 15)
mpl.rc('xtick', labelsize = 35)
# set y ticks
mpl.rc('ytick.major', width = 2, size = 15)
mpl.rc('ytick', labelsize = 35)

# set subplot space
mpl.rc('figure.subplot', bottom = 0.15, top = 0.90, 
       left = 0.1, right = 0.9, hspace = 0.3, wspace = 0.25)


# ## Read in the nlaWaterChem dataset

# In[3]:


nlaWaterChem = pd.read_csv(wd+'/dataset/Lakes/nlaWaterChem.csv')
# nlaWaterChem.columns


# ## Read in predicted Nutrients from Shen et al.

# In[4]:


import geopandas as gpd


# In[5]:


nutrientsShen = gpd.read_file(wd+'/output/gis/vector/nlaLakeSitesSnappedEdited.shp')


# In[6]:


# Calculate TN and TP
nutrientsShen['TN'] = np.nan
nutrientsShen['TP'] = np.nan

nutrientsShen.loc[nutrientsShen.TNautumn == -9999, 'TNautumn'] = np.nan
nutrientsShen.loc[nutrientsShen.TNspring == -9999, 'TNspring'] = np.nan
nutrientsShen.loc[nutrientsShen.TNsummer == -9999, 'TNsummer'] = np.nan
nutrientsShen.loc[nutrientsShen.TNwinter == -9999, 'TNwinter'] = np.nan

nutrientsShen.loc[nutrientsShen.TPautumn == -9999, 'TPautumn'] = np.nan
nutrientsShen.loc[nutrientsShen.TPspring == -9999, 'TPspring'] = np.nan
nutrientsShen.loc[nutrientsShen.TPsummer == -9999, 'TPsummer'] = np.nan
nutrientsShen.loc[nutrientsShen.TPwinter == -9999, 'TPwinter'] = np.nan

nutrientsShen.loc[:,['TN']] = (nutrientsShen.TNautumn + nutrientsShen.TNspring + nutrientsShen.TNsummer + nutrientsShen.TNwinter) / 4
nutrientsShen.loc[:,['TP']] = (nutrientsShen.TPautumn + nutrientsShen.TPspring + nutrientsShen.TPsummer + nutrientsShen.TPwinter) / 4


# In[7]:


# Change column names
cols = nutrientsShen.columns.tolist()
cols[1] = 'UID'
nutrientsShen.columns = cols


# In[8]:


nutrientsShen = nutrientsShen.drop(['pointid', 'geometry'], axis = 1)
nutrientsShen = nutrientsShen.merge(nlaWaterChem[['UID', 'SITE_ID']], on = 'UID')


# ## Combine nlaWaterChem with nutrientsShen

# In[9]:


nlaWaterChem = nlaWaterChem.merge(nutrientsShen.drop(['UID'], axis = 1), on = 'SITE_ID', how = 'left')
nlaWaterChem = nlaWaterChem.drop(['Unnamed: 0'], axis = 1)


# ### Plot Figures for Reseroirs

# In[10]:


nlaWaterChemRes = nlaWaterChem[nlaWaterChem.LAKE_ORIGIN12 == 'RESERVOIR']


# In[11]:


fig = plt.figure(figsize=(24, 9))

# subplot9: totalP versus chlx
ax = fig.add_subplot(1,2,1)
ax.scatter(nlaWaterChemRes['TP'], nlaWaterChemRes['CHLX_ugL'], s = 160, c = 'blue')
nlaWaterChemClean = nlaWaterChemRes[~nlaWaterChemRes.TP.isna() & ~nlaWaterChemRes.CHLX_ugL.isna()]
lm = stats.linregress(nlaWaterChemClean.TP, nlaWaterChemClean.CHLX_ugL)
a, b, r, p, e = np.asarray(lm)
x = np.array([nlaWaterChemClean.TP.min(), nlaWaterChemClean.TP.max()])
plt.plot(x, a*x+b, linewidth = 6, color = 'blue')
plt.text(0.1, 500, 
         '$y$ = {:1.0f} $x$ + {:1.3f},\n$r^2$ = {:1.2f}, $p$ = {:1.4f}'.format(a,b, r**2, p))
plt.title('total P versus Chl a', loc = 'center')
# plt.xticks(ticks = np.linspace(0, 2100, 4))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'total P')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')

# subplot10: totalN versus chlx
ax = fig.add_subplot(1,2,2)
ax.scatter(nlaWaterChemRes['TN'], nlaWaterChemRes['CHLX_ugL'], s = 160, c = 'blue')
nlaWaterChemClean = nlaWaterChemRes[~nlaWaterChemRes.TN.isna() & ~nlaWaterChemRes.CHLX_ugL.isna()]
lm = stats.linregress(nlaWaterChemClean.TN, nlaWaterChemClean.CHLX_ugL)
a, b, r, p, e = np.asarray(lm)
x = np.array([nlaWaterChemClean.TN.min(),nlaWaterChemClean.TN.max()])
plt.plot(x, a*x+b, linewidth = 6, color = 'blue')
plt.text(2, 500, 
         '$y$ = {:1.0f} $x$ + {:1.0f},\n$r^2$ = {:1.2f}, $p$ = {:1.4f}'.format(a,b, r**2, p))
plt.title('total N versus Chl a', loc = 'center')
# ax.set_xlim([0, 10])
# plt.xticks(ticks = np.linspace(0, 10, 6))
# plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'total N')
plt.ylabel(r'$\mathrm{Chl \/ (}\mu g \/ L^{-1})$')

plt.show()

fig.savefig(wd+'/output/figure/nutrientsChla/reservoirShenNP.png')


# ### Plot Figures for Lakes

# In[12]:


nlaWaterChemRes = nlaWaterChem[nlaWaterChem.LAKE_ORIGIN12 == 'NATURAL']


# In[13]:


fig = plt.figure(figsize=(24, 9))

# subplot9: totalP versus chlx
ax = fig.add_subplot(1,2,1)
ax.scatter(nlaWaterChemRes['TP'], nlaWaterChemRes['CHLX_ugL'], s = 160, c = 'blue')
nlaWaterChemClean = nlaWaterChemRes[~nlaWaterChemRes.TP.isna() & ~nlaWaterChemRes.CHLX_ugL.isna()]
lm = stats.linregress(nlaWaterChemClean.TP, nlaWaterChemClean.CHLX_ugL)
a, b, r, p, e = np.asarray(lm)
x = np.array([nlaWaterChemClean.TP.min(), nlaWaterChemClean.TP.max()])
plt.plot(x, a*x+b, linewidth = 6, color = 'blue')
plt.text(0.1, 500, 
         '$y$ = {:1.0f} $x$ + {:1.3f},\n$r^2$ = {:1.2f}, $p$ = {:1.4f}'.format(a,b, r**2, p))
plt.title('total P versus Chl a', loc = 'center')
# plt.xticks(ticks = np.linspace(0, 2100, 4))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'total P')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')

# subplot10: totalN versus chlx
ax = fig.add_subplot(1,2,2)
ax.scatter(nlaWaterChemRes['TN'], nlaWaterChemRes['CHLX_ugL'], s = 160, c = 'blue')
nlaWaterChemClean = nlaWaterChemRes[~nlaWaterChemRes.TN.isna() & ~nlaWaterChemRes.CHLX_ugL.isna()]
lm = stats.linregress(nlaWaterChemClean.TN, nlaWaterChemClean.CHLX_ugL)
a, b, r, p, e = np.asarray(lm)
x = np.array([nlaWaterChemClean.TN.min(),nlaWaterChemClean.TN.max()])
plt.plot(x, a*x+b, linewidth = 6, color = 'blue')
plt.text(2, 500, 
         '$y$ = {:1.0f} $x$ + {:1.0f},\n$r^2$ = {:1.2f}, $p$ = {:1.4f}'.format(a,b, r**2, p))
plt.title('total N versus Chl a', loc = 'center')
# ax.set_xlim([0, 10])
# plt.xticks(ticks = np.linspace(0, 10, 6))
# plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'total N')
plt.ylabel(r'$\mathrm{Chl \/ (}\mu g \/ L^{-1})$')

plt.show()

fig.savefig(wd+'/output/figure/nutrientsChla/lakeShenNP.png')


# ## Keep Only Sites with Decreasing Temp

# In[14]:


siteDecTemp = pd.read_csv(wd+'/dataset/Lakes/siteNumMeas2plusDecTemp.csv')
nlaWaterChemDecTemp = nlaWaterChem[nlaWaterChem.UID.isin(siteDecTemp.UID)]


# ### Plot Figures for Reservoirs

# In[15]:


nlaWaterChemRes = nlaWaterChemDecTemp[nlaWaterChemDecTemp.LAKE_ORIGIN12 == 'RESERVOIR']


# In[16]:


fig = plt.figure(figsize=(24, 9))

# subplot9: totalP versus chlx
ax = fig.add_subplot(1,2,1)
ax.scatter(nlaWaterChemRes['TP'], nlaWaterChemRes['CHLX_ugL'], s = 160, c = 'blue')
nlaWaterChemClean = nlaWaterChemRes[~nlaWaterChemRes.TP.isna() & ~nlaWaterChemRes.CHLX_ugL.isna()]
lm = stats.linregress(nlaWaterChemClean.TP, nlaWaterChemClean.CHLX_ugL)
a, b, r, p, e = np.asarray(lm)
x = np.array([nlaWaterChemClean.TP.min(), nlaWaterChemClean.TP.max()])
plt.plot(x, a*x+b, linewidth = 6, color = 'blue')
plt.text(0.1, 500, 
         '$y$ = {:1.0f} $x$ + {:1.3f},\n$r^2$ = {:1.2f}, $p$ = {:1.4f}'.format(a,b, r**2, p))
plt.title('total P versus Chl a', loc = 'center')
# plt.xticks(ticks = np.linspace(0, 2100, 4))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'total P')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')

# subplot10: totalN versus chlx
ax = fig.add_subplot(1,2,2)
ax.scatter(nlaWaterChemRes['TN'], nlaWaterChemRes['CHLX_ugL'], s = 160, c = 'blue')
nlaWaterChemClean = nlaWaterChemRes[~nlaWaterChemRes.TN.isna() & ~nlaWaterChemRes.CHLX_ugL.isna()]
lm = stats.linregress(nlaWaterChemClean.TN, nlaWaterChemClean.CHLX_ugL)
a, b, r, p, e = np.asarray(lm)
x = np.array([nlaWaterChemClean.TN.min(),nlaWaterChemClean.TN.max()])
plt.plot(x, a*x+b, linewidth = 6, color = 'blue')
plt.text(2, 500, 
         '$y$ = {:1.0f} $x$ + {:1.0f},\n$r^2$ = {:1.2f}, $p$ = {:1.4f}'.format(a,b, r**2, p))
plt.title('total N versus Chl a', loc = 'center')
# ax.set_xlim([0, 10])
# plt.xticks(ticks = np.linspace(0, 10, 6))
# plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'total N')
plt.ylabel(r'$\mathrm{Chl \/ (}\mu g \/ L^{-1})$')

plt.show()

fig.savefig(wd+'/output/figure/nutrientsChla/reservoirDecreaseTempShenNP.png')


# ### Plot Figures for Lakes

# In[17]:


nlaWaterChemRes = nlaWaterChemDecTemp[nlaWaterChemDecTemp.LAKE_ORIGIN12 == 'NATURAL']


# In[18]:


fig = plt.figure(figsize=(24, 9))

# subplot9: totalP versus chlx
ax = fig.add_subplot(1,2,1)
ax.scatter(nlaWaterChemRes['TP'], nlaWaterChemRes['CHLX_ugL'], s = 160, c = 'blue')
nlaWaterChemClean = nlaWaterChemRes[~nlaWaterChemRes.TP.isna() & ~nlaWaterChemRes.CHLX_ugL.isna()]
lm = stats.linregress(nlaWaterChemClean.TP, nlaWaterChemClean.CHLX_ugL)
a, b, r, p, e = np.asarray(lm)
x = np.array([nlaWaterChemClean.TP.min(), nlaWaterChemClean.TP.max()])
plt.plot(x, a*x+b, linewidth = 6, color = 'blue')
plt.text(0.1, 150, 
         '$y$ = {:1.0f} $x$ + {:1.3f},\n$r^2$ = {:1.2f}, $p$ = {:1.4f}'.format(a,b, r**2, p))
plt.title('total P versus Chl a', loc = 'center')
# plt.xticks(ticks = np.linspace(0, 2100, 4))
plt.yticks(np.arange(0, 201, 50, dtype=int))
plt.xlabel(r'total P')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')

# subplot10: totalN versus chlx
ax = fig.add_subplot(1,2,2)
ax.scatter(nlaWaterChemRes['TN'], nlaWaterChemRes['CHLX_ugL'], s = 160, c = 'blue')
nlaWaterChemClean = nlaWaterChemRes[~nlaWaterChemRes.TN.isna() & ~nlaWaterChemRes.CHLX_ugL.isna()]
lm = stats.linregress(nlaWaterChemClean.TN, nlaWaterChemClean.CHLX_ugL)
a, b, r, p, e = np.asarray(lm)
x = np.array([nlaWaterChemClean.TN.min(),nlaWaterChemClean.TN.max()])
plt.plot(x, a*x+b, linewidth = 6, color = 'blue')
plt.text(2, 150, 
         '$y$ = {:1.0f} $x$ + {:1.0f},\n$r^2$ = {:1.2f}, $p$ = {:1.4f}'.format(a,b, r**2, p))
plt.title('total N versus Chl a', loc = 'center')
# ax.set_xlim([0, 10])
# plt.xticks(ticks = np.linspace(0, 10, 6))
# plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'total N')
plt.ylabel(r'$\mathrm{Chl \/ (}\mu g \/ L^{-1})$')

plt.show()

fig.savefig(wd+'/output/figure/nutrientsChla/lakeDecreaseTempShenNP.png')

