#!/usr/bin/env python
# coding: utf-8

# ## Import modules and set path

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
       left = 0.1, right = 0.90, hspace = 0.3, wspace = 0.25)


# ## Read in the nlaWaterChem dataset

# In[3]:


nlaWaterChem = pd.read_csv(wd+'/dataset/Lakes/nlaWaterChem.csv')
# nlaWaterChem.columns


# In[4]:


# nlaWaterChem = \
# nlaWaterChem[['UID', 'LAKE_ORIGIN', 'LAKE_ORIGIN12', 'LAT_DD83', 'LON_DD83', 'SIZE_CLASS', 'URBAN',
#               'DOC_RESULT', 'PH_RESULT', 'ANC_RESULT', 'COND_RESULT', 'TURB_RESULT',
#               'DO2_2M', 'PTL_RESULT', 'NTL_RESULT', 'CHLL_ugL', 'CHLX_ugL',
#               'AMMONIA_N_RESULT', 'NITRATE_N_RESULT', 'NITRITE_N_RESULT', 'NITRATE_NITRITE_N_RESULT',
#               'SODIUM_RESULT', 'POTASSIUM_RESULT', 'CALCIUM_RESULT',  'MAGNESIUM_RESULT', 'ALUMINUM_RESULT',
#               'CHLORIDE_RESULT',  'SULFATE_RESULT', 'SILICA_RESULT',]]


# In[5]:


def thousandkCon(start, end, number):
    labNum = np.linspace(start, end, number) / 1000
#     labNum = list(map(int, labNum.tolist()))
    labStr = []
    for num in labNum:
        num = '{:.1f}'.format(num)
        labStr.append(str(num)+'k')
    return(labStr)


# ## Plot Figures for Reseroirs

# In[6]:


nlaWaterChemRes = nlaWaterChem[nlaWaterChem.LAKE_ORIGIN12 == 'RESERVOIR']


# In[7]:


fig = plt.figure(figsize=(24, 9))

# subplot9: totalP versus chlx
ax = fig.add_subplot(1,2,1)
ax.scatter(nlaWaterChemRes['PTL_RESULT'], nlaWaterChemRes['CHLX_ugL'], s = 160, c = 'blue')
nlaWaterChemClean = nlaWaterChemRes[~nlaWaterChemRes.PTL_RESULT.isna() & ~nlaWaterChemRes.CHLX_ugL.isna()]
lm = stats.linregress(nlaWaterChemClean.PTL_RESULT, nlaWaterChemClean.CHLX_ugL)
a, b, r, p, e = np.asarray(lm)
x = np.array([nlaWaterChemClean.PTL_RESULT.min(), 2100])
plt.plot(x, a*x+b, linewidth = 6, color = 'blue')
plt.text(1000, 500, 
         '$y$ = {:1.1f} $x$ + {:1.0f},\n$r^2$ = {:1.2f}, $p$ = {:1.4f}'.format(a,b, r**2, p))
plt.title('total P versus Chl a', loc = 'center')
plt.xticks(ticks = np.linspace(0, 2100, 4))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'total P')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')

# subplot10: totalN versus chlx
ax = fig.add_subplot(1,2,2)
ax.scatter(nlaWaterChemRes['NTL_RESULT'], nlaWaterChemRes['CHLX_ugL'], s = 160, c = 'blue')
nlaWaterChemClean = nlaWaterChemRes[~nlaWaterChemRes.NTL_RESULT.isna() & ~nlaWaterChemRes.CHLX_ugL.isna()]
lm = stats.linregress(nlaWaterChemClean.NTL_RESULT, nlaWaterChemClean.CHLX_ugL)
a, b, r, p, e = np.asarray(lm)
x = np.array([nlaWaterChemClean.NTL_RESULT.min(),nlaWaterChemClean.NTL_RESULT.max()])
plt.plot(x, a*x+b, linewidth = 6, color = 'blue')
plt.text(4, 500, 
         '$y$ = {:1.0f} $x$ + {:1.0f},\n$r^2$ = {:1.2f}, $p$ = {:1.4f}'.format(a,b, r**2, p))
plt.title('total N versus Chl a', loc = 'center')
# ax.set_xlim([0, 10])
# plt.xticks(ticks = np.linspace(0, 10, 6))
# plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'total N')
plt.ylabel(r'$\mathrm{Chl \/ (}\mu g \/ L^{-1})$')

plt.show()

fig.savefig(wd+'/output/figure/nutrientsChla/reservoir.png')


# In[8]:


# subplot10: totalN versus chlx
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(1,1,1)
ax.scatter(nlaWaterChemRes['PTL_RESULT'], nlaWaterChemRes['NTL_RESULT'], s = 160, c = 'blue')
nlaWaterChemClean = nlaWaterChemRes[~nlaWaterChemRes.NTL_RESULT.isna() & ~nlaWaterChemRes.CHLX_ugL.isna()]
lm = stats.linregress(nlaWaterChemClean.PTL_RESULT, nlaWaterChemClean.NTL_RESULT)
a, b, r, p, e = np.asarray(lm)
x = np.array([nlaWaterChemClean.PTL_RESULT.min(),nlaWaterChemClean.PTL_RESULT.max()])
plt.plot(x, a*x+b, linewidth = 6, color = 'blue')
plt.text(600, 4, 
         '$y$ = {:1.3f} $x$ + {:1.2f},\n$r^2$ = {:1.2f}, $p$ = {:1.4f}'.format(a,b, r**2, p))
plt.title('total P versus total N', loc = 'center')
# ax.set_xlim([0, 10])
# plt.xticks(ticks = np.linspace(0, 10, 6))
# plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'total P')
plt.ylabel(r'total N')

plt.show()


# ## Keep Only Sites with Decreasing Temp

# In[9]:


siteDecTemp = pd.read_csv(wd+'/dataset/Lakes/siteNumMeas2plusDecTemp.csv')
nlaWaterChemDecTemp = nlaWaterChem[nlaWaterChem.UID.isin(siteDecTemp.UID)]


# In[10]:


nlaWaterChemRes = nlaWaterChemDecTemp[nlaWaterChemDecTemp.LAKE_ORIGIN12 == 'RESERVOIR']


# In[11]:


fig = plt.figure(figsize=(24, 9))

# subplot9: totalP versus chlx
ax = fig.add_subplot(1,2,1)
ax.scatter(nlaWaterChemRes['PTL_RESULT'], nlaWaterChemRes['CHLX_ugL'], s = 160, c = 'blue')
nlaWaterChemClean = nlaWaterChemRes[~nlaWaterChemRes.PTL_RESULT.isna() & ~nlaWaterChemRes.CHLX_ugL.isna()]
lm = stats.linregress(nlaWaterChemClean.PTL_RESULT, nlaWaterChemClean.CHLX_ugL)
a, b, r, p, e = np.asarray(lm)
x = np.array([nlaWaterChemClean.PTL_RESULT.min(), 2100])
plt.plot(x, a*x+b, linewidth = 6, color = 'blue')
plt.text(1000, 500, 
         '$y$ = {:1.1f} $x$ + {:1.0f},\n$r^2$ = {:1.2f}, $p$ = {:1.4f}'.format(a,b, r**2, p))
plt.title('total P versus Chl a', loc = 'center')
plt.xticks(ticks = np.linspace(0, 2100, 4))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'total P')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')

# subplot10: totalN versus chlx
ax = fig.add_subplot(1,2,2)
ax.scatter(nlaWaterChemRes['NTL_RESULT'], nlaWaterChemRes['CHLX_ugL'], s = 160, c = 'blue')
nlaWaterChemClean = nlaWaterChemRes[~nlaWaterChemRes.NTL_RESULT.isna() & ~nlaWaterChemRes.CHLX_ugL.isna()]
lm = stats.linregress(nlaWaterChemClean.NTL_RESULT, nlaWaterChemClean.CHLX_ugL)
a, b, r, p, e = np.asarray(lm)
x = np.array([nlaWaterChemClean.NTL_RESULT.min(),nlaWaterChemClean.NTL_RESULT.max()])
plt.plot(x, a*x+b, linewidth = 6, color = 'blue')
plt.text(4, 500, 
         '$y$ = {:1.0f} $x$ + {:1.0f},\n$r^2$ = {:1.2f}, $p$ = {:1.4f}'.format(a,b, r**2, p))
plt.title('total N versus Chl a', loc = 'center')
# ax.set_xlim([0, 10])
# plt.xticks(ticks = np.linspace(0, 10, 6))
# plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'total N')
plt.ylabel(r'$\mathrm{Chl \/ (}\mu g \/ L^{-1})$')

plt.show()

fig.savefig(wd+'/output/figure/nutrientsChla/reservoirDecreaseTemp.png')


# ## keep only sites where Temp > 20

# In[12]:


nlaWaterChemDecTemp20degPlus = nlaWaterChemDecTemp[nlaWaterChemDecTemp.TEMPERATURE >= 20]


# In[13]:


nlaWaterChemRes = nlaWaterChemDecTemp20degPlus[nlaWaterChemDecTemp20degPlus.LAKE_ORIGIN12 == 'RESERVOIR']


# In[14]:


fig = plt.figure(figsize=(24, 9))

# subplot9: totalP versus chlx
ax = fig.add_subplot(1,2,1)
ax.scatter(nlaWaterChemRes['PTL_RESULT'], nlaWaterChemRes['CHLX_ugL'], s = 160, c = 'blue')
nlaWaterChemClean = nlaWaterChemRes[~nlaWaterChemRes.PTL_RESULT.isna() & ~nlaWaterChemRes.CHLX_ugL.isna()]
lm = stats.linregress(nlaWaterChemClean.PTL_RESULT, nlaWaterChemClean.CHLX_ugL)
a, b, r, p, e = np.asarray(lm)
x = np.array([nlaWaterChemClean.PTL_RESULT.min(), 2100])
plt.plot(x, a*x+b, linewidth = 6, color = 'blue')
plt.text(1000, 500, 
         '$y$ = {:1.1f} $x$ + {:1.0f},\n$r^2$ = {:1.2f}, $p$ = {:1.4f}'.format(a,b, r**2, p))
plt.title('total P versus Chl a', loc = 'center')
plt.xticks(ticks = np.linspace(0, 2100, 4))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'total P')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')

# subplot10: totalN versus chlx
ax = fig.add_subplot(1,2,2)
ax.scatter(nlaWaterChemRes['NTL_RESULT'], nlaWaterChemRes['CHLX_ugL'], s = 160, c = 'blue')
nlaWaterChemClean = nlaWaterChemRes[~nlaWaterChemRes.NTL_RESULT.isna() & ~nlaWaterChemRes.CHLX_ugL.isna()]
lm = stats.linregress(nlaWaterChemClean.NTL_RESULT, nlaWaterChemClean.CHLX_ugL)
a, b, r, p, e = np.asarray(lm)
x = np.array([nlaWaterChemClean.NTL_RESULT.min(),nlaWaterChemClean.NTL_RESULT.max()])
plt.plot(x, a*x+b, linewidth = 6, color = 'blue')
plt.text(2, 300, 
         '$y$ = {:1.0f} $x$ + {:1.0f},\n$r^2$ = {:1.2f}, $p$ = {:1.4f}'.format(a,b, r**2, p))
plt.title('total N versus Chl a', loc = 'center')
# ax.set_xlim([0, 10])
# plt.xticks(ticks = np.linspace(0, 10, 6))
# plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'total N')
plt.ylabel(r'$\mathrm{Chl \/ (}\mu g \/ L^{-1})$')

plt.show()

fig.savefig(wd+'/output/figure/nutrientsChla/reservoirDecreaseTemp20degPlus.png')


# ## keep only reservoirs bigger than 5 ha

# In[15]:


nlaWaterChemDecTempBig = nlaWaterChemDecTemp[nlaWaterChemDecTemp.AREA_HA >= 5]
nlaWaterChemDecTempBig.shape


# In[16]:


nlaWaterChemRes = nlaWaterChemDecTempBig[nlaWaterChemDecTempBig.LAKE_ORIGIN12 == 'RESERVOIR']


# In[17]:


fig = plt.figure(figsize=(24, 9))

# subplot9: totalP versus chlx
ax = fig.add_subplot(1,2,1)
ax.scatter(nlaWaterChemRes['PTL_RESULT'], nlaWaterChemRes['CHLX_ugL'], s = 160, c = 'blue')
nlaWaterChemClean = nlaWaterChemRes[~nlaWaterChemRes.PTL_RESULT.isna() & ~nlaWaterChemRes.CHLX_ugL.isna()]
lm = stats.linregress(nlaWaterChemClean.PTL_RESULT, nlaWaterChemClean.CHLX_ugL)
a, b, r, p, e = np.asarray(lm)
x = np.array([nlaWaterChemClean.PTL_RESULT.min(), 2100])
plt.plot(x, a*x+b, linewidth = 6, color = 'blue')
plt.text(1000, 500, 
         '$y$ = {:1.1f} $x$ + {:1.0f},\n$r^2$ = {:1.2f}, $p$ = {:1.4f}'.format(a,b, r**2, p))
plt.title('total P versus Chl a', loc = 'center')
plt.xticks(ticks = np.linspace(0, 2100, 4))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'total P')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')

# subplot10: totalN versus chlx
ax = fig.add_subplot(1,2,2)
ax.scatter(nlaWaterChemRes['NTL_RESULT'], nlaWaterChemRes['CHLX_ugL'], s = 160, c = 'blue')
nlaWaterChemClean = nlaWaterChemRes[~nlaWaterChemRes.NTL_RESULT.isna() & ~nlaWaterChemRes.CHLX_ugL.isna()]
lm = stats.linregress(nlaWaterChemClean.NTL_RESULT, nlaWaterChemClean.CHLX_ugL)
a, b, r, p, e = np.asarray(lm)
x = np.array([nlaWaterChemClean.NTL_RESULT.min(),nlaWaterChemClean.NTL_RESULT.max()])
plt.plot(x, a*x+b, linewidth = 6, color = 'blue')
plt.text(2, 300, 
         '$y$ = {:1.0f} $x$ + {:1.0f},\n$r^2$ = {:1.2f}, $p$ = {:1.4f}'.format(a,b, r**2, p))
plt.title('total N versus Chl a', loc = 'center')
# ax.set_xlim([0, 10])
# plt.xticks(ticks = np.linspace(0, 10, 6))
# plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'total N')
plt.ylabel(r'$\mathrm{Chl \/ (}\mu g \/ L^{-1})$')

plt.show()

fig.savefig(wd+'/output/figure/nutrientsChla/reservoirDecreaseTempBigThan5HA.png')

