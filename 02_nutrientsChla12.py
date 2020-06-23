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


# In[14]:


# mpl.rcParams


# In[53]:


# set title
mpl.rc('figure', titlesize = 45, titleweight = 'bold')
# set axes
mpl.rc('axes', titlesize = 40, titleweight = 'bold', titlepad = 30,
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
plt.subplots_adjust(bottom = 0.05, top = 0.92, left = 0.05, right = 0.95, wspace = 0.25, hspace = 0.3)
mpl.rc('figure.subplot', bottom = 0.05, top = 0.95, 
       left = 0.05, right = 0.95, hspace = 0.3, wspace = 0.25)


# ## Read in the nlaWaterChem dataset

# In[4]:


nlaWaterChem = pd.read_csv(wd+'/dataset/Lakes/nlaWaterChem.csv')
nlaWaterChem.columns


# In[ ]:


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


# ## Plotting all lakes and reservoirs

# In[ ]:


fig = plt.figure(figsize=(42,36))

# subplot1: area versus chlx
ax = fig.add_subplot(4,5,1)
ax.scatter(nlaWaterChem['AREA_HA'], nlaWaterChem['CHLX_ugL'], s = 80, c = 'green')
plt.title('area versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
plt.xticks(ticks = np.linspace(0, 180000, 6), labels = thousandkCon(0, 180000, 6), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'area (ha)', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)


# subplot2: elevation versus chlx
ax = fig.add_subplot(4,5,2)
ax.scatter(nlaWaterChem['ELEVATION'], nlaWaterChem['CHLX_ugL'], s = 80, c = 'green')
plt.title('elevation versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
plt.xticks(ticks = np.linspace(0, 3600, 7), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'elevation (m)', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)



# subplot3: color versus chlx
ax = fig.add_subplot(4,5,3)
ax.scatter(nlaWaterChem['COLOR_RESULT'], nlaWaterChem['CHLX_ugL'], s = 80, c = 'green')
plt.title('color versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
plt.xticks(ticks = np.linspace(0, 800, 5), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'color', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)


# subplot4: turbidity versus chlx
ax = fig.add_subplot(4,5,4)
ax.scatter(nlaWaterChem['TURB_RESULT'], nlaWaterChem['CHLX_ugL'], s = 80, c = 'green')
plt.title('turbidity versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
plt.xticks(ticks = np.linspace(0, 500, 6), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'turbidity', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# subplot5: tss versus chlx
ax = fig.add_subplot(4,5,5)
ax.scatter(nlaWaterChem['TSS_RESULT'], nlaWaterChem['CHLX_ugL'], s = 80, c = 'green')
plt.title('tss versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
# plt.xticks(ticks = np.linspace(0, 100, 6), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'tss', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# subplot6: doc versus chlx
ax = fig.add_subplot(4,5,6)
ax.scatter(nlaWaterChem['DOC_RESULT'], nlaWaterChem['CHLX_ugL'], s = 80, c = 'green')
plt.title('doc versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
ax.set_xlim([0, 100])
plt.xticks(ticks = np.linspace(0, 100, 6), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'doc', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)


# subplot7: chll versus chlx
ax = fig.add_subplot(4,5,7)
ax.scatter(nlaWaterChem['CHLL_ugL'], nlaWaterChem['CHLX_ugL'], s = 80, c = 'green')
plt.title('Chll a versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
plt.xticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'chll', fontsize = 20, fontweight = 'bold')
plt.ylabel(r'chlx', fontsize = 20, fontweight = 'bold')

lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])]  # min and max of both axes
# now plot both limits against eachother
ax.plot(lims, lims, 'k-', alpha=0.5, zorder=0)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# subplot8: silica versus chlx
ax = fig.add_subplot(4,5,8)
ax.scatter(nlaWaterChem['SILICA_RESULT'], nlaWaterChem['CHLX_ugL'], s = 80, c = 'green')
plt.title('silica versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
ax.set_xlim([0, 200])
plt.xticks(ticks = np.linspace(0, 200, 5), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'silica', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# subplot9: totalP versus chlx
ax = fig.add_subplot(4,5,9)
ax.scatter(nlaWaterChem['PTL_RESULT'], nlaWaterChem['CHLX_ugL'], s = 80, c = 'green')
plt.title('totalP versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
plt.xticks(ticks = np.linspace(0, 3500, 6), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'totalP', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# subplot10: totalN versus chlx
ax = fig.add_subplot(4,5,10)
ax.scatter(nlaWaterChem['NTL_RESULT'], nlaWaterChem['CHLX_ugL'], s = 80, c = 'green')
plt.title('totalN versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
ax.set_xlim([0, 20])
plt.xticks(ticks = np.linspace(0, 20, 5), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'totalN', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# subplot11: NO3+NO2 versus chlx
ax = fig.add_subplot(4,5,11)
ax.scatter(nlaWaterChem['NITRATE_NITRITE_N_RESULT'], nlaWaterChem['CHLX_ugL'], s = 80, c = 'green')
plt.title('NO3+NO2 versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
ax.set_xlim([0, 10])
plt.xticks(ticks = np.linspace(0, 10, 6), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'NO3+NO2', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# subplot12: nitrite versus chlx
ax = fig.add_subplot(4,5,12)
ax.scatter(nlaWaterChem['NITRITE_N_RESULT'], nlaWaterChem['CHLX_ugL'], s = 80, c = 'green')
plt.title('nitrite versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
ax.set_xlim([0, 0.1])
plt.xticks(ticks = np.linspace(0, 0.1, 6), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'nitrite', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# subplot13: nitrate versus chlx
ax = fig.add_subplot(4,5,13)
ax.scatter(nlaWaterChem['NITRATE_N_RESULT'], nlaWaterChem['CHLX_ugL'], s = 80, c = 'green')
plt.title('nitrate versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
ax.set_xlim([0, 10])
plt.xticks(ticks = np.linspace(0, 10, 6), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'nitrate', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# subplot14: amonia versus chlx
ax = fig.add_subplot(4,5,14)
ax.scatter(nlaWaterChem['AMMONIA_N_RESULT'], nlaWaterChem['CHLX_ugL'], s = 80, c = 'green')
plt.title('amonia versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
plt.xticks(ticks = np.linspace(0, 3, 6), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'amonia', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# subplot15: oxygen versus chlx
ax = fig.add_subplot(4,5,15)
ax.scatter(nlaWaterChem['DO2_2M'], nlaWaterChem['CHLX_ugL'], s = 80, c = 'green')
plt.title('DO versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
plt.xticks(ticks = np.linspace(0, 25, 6), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'DO', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# subplot16: oxygen versus chlx
ax = fig.add_subplot(4,5,16)
ax.scatter(nlaWaterChem['OXYGEN'], nlaWaterChem['CHLX_ugL'], s = 80, c = 'green')
plt.title('DO at depth0 versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
plt.xticks(ticks = np.linspace(0, 25, 6), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'DO', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# subplot17: pH versus chlx
ax = fig.add_subplot(4,5,17)
ax.scatter(nlaWaterChem['PH'], nlaWaterChem['CHLX_ugL'], s = 80, c = 'green')
plt.title('pH versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
plt.xticks(ticks = np.linspace(0, 12, 4), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'pH', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# subplot18: conductivity versus chlx
ax = fig.add_subplot(4,5,18)
ax.scatter(nlaWaterChem['CONDUCTIVITY'], nlaWaterChem['CHLX_ugL'], s = 80, c = 'green')
plt.title('CONDUCTIVITY versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
plt.xticks(ticks = np.linspace(0, 70000, 8), size = 14)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'CONDUCTIVITY', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# subplot19: TEMPERATURE versus chlx
ax = fig.add_subplot(4,5,19)
ax.scatter(nlaWaterChem['TEMPERATURE'], nlaWaterChem['CHLX_ugL'], s = 80, c = 'green')
plt.title('TEMPERATURE versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
ax.set_xlim([0,40])
plt.xticks(ticks = np.linspace(0, 40, 6), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'TEMPERATURE', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

plt.subplots_adjust(bottom = 0.05, top = 0.95, left = 0.05, right = 0.95, wspace = 0.25, hspace = 0.3)


plt.show()

fig.savefig(wd+'/output/figure/nutrientsChla/allLakeReservoir.png')


# ## Plotting Figures for nutural lakes

# In[ ]:


fig = plt.figure(figsize=(42,36))

# subplot1: area versus chlx
ax = fig.add_subplot(3,5,1)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'NATURAL', 'AREA_HA'], 
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'NATURAL', 'CHLX_ugL'], s = 80, c = 'green')
plt.title('area versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
plt.xticks(ticks = np.linspace(0, 180000, 6), labels = thousandkCon(0, 180000, 6), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'area (ha)', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)


# subplot2: elevation versus chlx
ax = fig.add_subplot(3,5,2)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'NATURAL', 'ELEVATION'],
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'NATURAL', 'CHLX_ugL'], s = 80, c = 'green')
plt.title('elevation versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
plt.xticks(ticks = np.linspace(0, 3600, 7), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'elevation (m)', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)


# subplot3: color versus chlx
ax = fig.add_subplot(3,5,3)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'NATURAL', 'COLOR_RESULT'], 
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'NATURAL', 'CHLX_ugL'], s = 80, c = 'green')
plt.title('color versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
plt.xticks(ticks = np.linspace(0, 800, 5), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'color', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# subplot4: turbidity versus chlx
ax = fig.add_subplot(3,5,4)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'NATURAL', 'TURB_RESULT'], 
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'NATURAL', 'CHLX_ugL'], s = 80, c = 'green')
plt.title('turbidity versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
plt.xticks(ticks = np.linspace(0, 500, 6), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'turbidity', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# subplot5: tss versus chlx
ax = fig.add_subplot(3,5,5)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'NATURAL', 'TSS_RESULT'],
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'NATURAL', 'CHLX_ugL'], s = 80, c = 'green')
plt.title('tss versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
# plt.xticks(ticks = np.linspace(0, 100, 6), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'tss', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# subplot6: doc versus chlx
ax = fig.add_subplot(3,5,6)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'NATURAL', 'DOC_RESULT'],
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'NATURAL', 'CHLX_ugL'], s = 80, c = 'green')
plt.title('doc versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
ax.set_xlim([0, 100])
plt.xticks(ticks = np.linspace(0, 100, 6), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'doc', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)


# subplot7: chll versus chlx
ax = fig.add_subplot(3,5,7)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'NATURAL', 'CHLL_ugL'],
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'NATURAL', 'CHLX_ugL'], s = 80, c = 'green')
plt.title('Chll a versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
plt.xticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'chll', fontsize = 20, fontweight = 'bold')
plt.ylabel(r'chlx', fontsize = 20, fontweight = 'bold')

lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])]  # min and max of both axes
# now plot both limits against eachother
ax.plot(lims, lims, 'k-', alpha=0.5, zorder=0)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# subplot8: silica versus chlx
ax = fig.add_subplot(3,5,8)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'NATURAL', 'SILICA_RESULT'],
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'NATURAL', 'CHLX_ugL'], s = 80, c = 'green')
plt.title('silica versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
ax.set_xlim([0, 200])
plt.xticks(ticks = np.linspace(0, 200, 5), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'silica', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# subplot9: totalP versus chlx
ax = fig.add_subplot(3,5,9)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'NATURAL', 'PTL_RESULT'],
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'NATURAL', 'CHLX_ugL'], s = 80, c = 'green')
plt.title('totalP versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
plt.xticks(ticks = np.linspace(0, 3500, 6), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'totalP', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# subplot10: totalN versus chlx
ax = fig.add_subplot(3,5,10)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'NATURAL', 'NTL_RESULT'], 
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'NATURAL', 'CHLX_ugL'], s = 80, c = 'green')
plt.title('totalN versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
ax.set_xlim([0, 20])
plt.xticks(ticks = np.linspace(0, 20, 5), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'totalN', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# subplot11: NO3+NO2 versus chlx
ax = fig.add_subplot(3,5,11)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'NATURAL', 'NITRATE_NITRITE_N_RESULT'],
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'NATURAL', 'CHLX_ugL'], s = 80, c = 'green')
plt.title('NO3+NO2 versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
ax.set_xlim([0, 10])
plt.xticks(ticks = np.linspace(0, 10, 6), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'NO3+NO2', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# subplot12: nitrite versus chlx
ax = fig.add_subplot(3,5,12)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'NATURAL', 'NITRITE_N_RESULT'],
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'NATURAL', 'CHLX_ugL'], s = 80, c = 'green')
plt.title('nitrite versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
ax.set_xlim([0, 0.1])
plt.xticks(ticks = np.linspace(0, 0.1, 6), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'nitrite', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# subplot13: nitrate versus chlx
ax = fig.add_subplot(3,5,13)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'NATURAL', 'NITRATE_N_RESULT'], 
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'NATURAL', 'CHLX_ugL'], s = 80, c = 'green')
plt.title('nitrate versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
ax.set_xlim([0, 10])
plt.xticks(ticks = np.linspace(0, 10, 6), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'nitrate', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# subplot14: amonia versus chlx
ax = fig.add_subplot(3,5,14)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'NATURAL', 'AMMONIA_N_RESULT'],
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'NATURAL', 'CHLX_ugL'], s = 80, c = 'green')
plt.title('amonia versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
plt.xticks(ticks = np.linspace(0, 3, 6), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'amonia', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# subplot15: oxygen versus chlx
ax = fig.add_subplot(3,5,15)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'NATURAL', 'DO2_2M'],
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'NATURAL', 'CHLX_ugL'], s = 80, c = 'green')
plt.title('DO versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
plt.xticks(ticks = np.linspace(0, 25, 6), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'DO', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

plt.subplots_adjust(bottom = 0.05, top = 0.95, left = 0.05, right = 0.95, wspace = 0.25, hspace = 0.3)


plt.show()

fig.savefig(wd+'/output/figure/nutrientsChla/naturalLakes.png')


# ## Plot Figures for MAN-MADE Lakes

# In[ ]:


fig = plt.figure(figsize=(42,36))

# subplot1: area versus chlx
ax = fig.add_subplot(3,5,1)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'MAN_MADE', 'AREA_HA'], 
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'MAN_MADE', 'CHLX_ugL'], s = 80, c = 'green')
plt.title('area versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
plt.xticks(ticks = np.linspace(0, 180000, 6), labels = thousandkCon(0, 180000, 6), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'area (ha)', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)


# subplot2: elevation versus chlx
ax = fig.add_subplot(3,5,2)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'MAN_MADE', 'ELEVATION'],
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'MAN_MADE', 'CHLX_ugL'], s = 80, c = 'green')
plt.title('elevation versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
plt.xticks(ticks = np.linspace(0, 3600, 7), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'elevation (m)', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)


# subplot3: color versus chlx
ax = fig.add_subplot(3,5,3)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'MAN_MADE', 'COLOR_RESULT'], 
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'MAN_MADE', 'CHLX_ugL'], s = 80, c = 'green')
plt.title('color versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
plt.xticks(ticks = np.linspace(0, 800, 5), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'color', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# subplot4: turbidity versus chlx
ax = fig.add_subplot(3,5,4)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'MAN_MADE', 'TURB_RESULT'], 
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'MAN_MADE', 'CHLX_ugL'], s = 80, c = 'green')
plt.title('turbidity versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
plt.xticks(ticks = np.linspace(0, 500, 6), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'turbidity', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# subplot5: tss versus chlx
ax = fig.add_subplot(3,5,5)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'MAN_MADE', 'TSS_RESULT'],
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'MAN_MADE', 'CHLX_ugL'], s = 80, c = 'green')
plt.title('tss versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
# plt.xticks(ticks = np.linspace(0, 100, 6), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'tss', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# subplot6: doc versus chlx
ax = fig.add_subplot(3,5,6)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'MAN_MADE', 'DOC_RESULT'],
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'MAN_MADE', 'CHLX_ugL'], s = 80, c = 'green')
plt.title('doc versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
ax.set_xlim([0, 100])
plt.xticks(ticks = np.linspace(0, 100, 6), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'doc', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)


# subplot7: chll versus chlx
ax = fig.add_subplot(3,5,7)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'MAN_MADE', 'CHLL_ugL'],
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'MAN_MADE', 'CHLX_ugL'], s = 80, c = 'green')
plt.title('Chll a versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
plt.xticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'chll', fontsize = 20, fontweight = 'bold')
plt.ylabel(r'chlx', fontsize = 20, fontweight = 'bold')

lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])]  # min and max of both axes
# now plot both limits against eachother
ax.plot(lims, lims, 'k-', alpha=0.5, zorder=0)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# subplot8: silica versus chlx
ax = fig.add_subplot(3,5,8)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'MAN_MADE', 'SILICA_RESULT'],
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'MAN_MADE', 'CHLX_ugL'], s = 80, c = 'green')
plt.title('silica versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
ax.set_xlim([0, 200])
plt.xticks(ticks = np.linspace(0, 200, 5), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'silica', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# subplot9: totalP versus chlx
ax = fig.add_subplot(3,5,9)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'MAN_MADE', 'PTL_RESULT'],
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'MAN_MADE', 'CHLX_ugL'], s = 80, c = 'green')
plt.title('totalP versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
plt.xticks(ticks = np.linspace(0, 3500, 6), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'totalP', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# subplot10: totalN versus chlx
ax = fig.add_subplot(3,5,10)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'MAN_MADE', 'NTL_RESULT'], 
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'MAN_MADE', 'CHLX_ugL'], s = 80, c = 'green')
plt.title('totalN versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
ax.set_xlim([0, 20])
plt.xticks(ticks = np.linspace(0, 20, 5), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'totalN', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# subplot11: NO3+NO2 versus chlx
ax = fig.add_subplot(3,5,11)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'MAN_MADE', 'NITRATE_NITRITE_N_RESULT'],
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'MAN_MADE', 'CHLX_ugL'], s = 80, c = 'green')
plt.title('NO3+NO2 versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
ax.set_xlim([0, 10])
plt.xticks(ticks = np.linspace(0, 10, 6), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'NO3+NO2', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# subplot12: nitrite versus chlx
ax = fig.add_subplot(3,5,12)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'MAN_MADE', 'NITRITE_N_RESULT'],
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'MAN_MADE', 'CHLX_ugL'], s = 80, c = 'green')
plt.title('nitrite versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
ax.set_xlim([0, 0.1])
plt.xticks(ticks = np.linspace(0, 0.1, 6), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'nitrite', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# subplot13: nitrate versus chlx
ax = fig.add_subplot(3,5,13)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'MAN_MADE', 'NITRATE_N_RESULT'], 
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'MAN_MADE', 'CHLX_ugL'], s = 80, c = 'green')
plt.title('nitrate versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
ax.set_xlim([0, 10])
plt.xticks(ticks = np.linspace(0, 10, 6), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'nitrate', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# subplot14: amonia versus chlx
ax = fig.add_subplot(3,5,14)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'MAN_MADE', 'AMMONIA_N_RESULT'],
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'MAN_MADE', 'CHLX_ugL'], s = 80, c = 'green')
plt.title('amonia versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
plt.xticks(ticks = np.linspace(0, 3, 6), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'amonia', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

# subplot15: oxygen versus chlx
ax = fig.add_subplot(3,5,15)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'MAN_MADE', 'DO2_2M'],
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN == 'MAN_MADE', 'CHLX_ugL'], s = 80, c = 'green')
plt.title('DO versus Chlx a', size = 20, loc = 'left', fontweight = 'bold')
plt.xticks(ticks = np.linspace(0, 25, 6), size = 20)
plt.yticks(np.arange(0, 801, 200, dtype=int), size = 20)
plt.xlabel(r'DO', fontsize = 20)
plt.ylabel(r'$\mathrm{chlx \/ (}\mu g \/ L^{-1})$', fontsize = 20)
ax.tick_params(length = 12, width = 1)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

plt.subplots_adjust(bottom = 0.05, top = 0.95, left = 0.05, right = 0.95, wspace = 0.25, hspace = 0.3)


plt.show()

fig.savefig(wd+'/output/figure/nutrientsChla/manMade.png')


# ## Plot Figures for Reservoirs

# In[54]:


fig = plt.figure(figsize=(60,30))

# subplot1: area versus chlx
ax = fig.add_subplot(3,5,1)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN12 == 'RESERVOIR', 'AREA_HA'], 
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title('Area versus Chl a', loc = 'center')
plt.xticks(ticks = np.linspace(0, 180000, 6), labels = thousandkCon(0, 180000, 6))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'Area (ha)')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')


# subplot2: elevation versus chlx
ax = fig.add_subplot(3,5,2)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN12 == 'RESERVOIR', 'ELEVATION'],
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title('Elevation versus Chl a', loc = 'center')
plt.xticks(ticks = np.linspace(0, 3600, 7))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'Elevation (m)')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')

# subplot3: color versus chlx
ax = fig.add_subplot(3,5,3)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN12 == 'RESERVOIR', 'COLOR_RESULT'], 
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title('Color versus Chl a', loc = 'center')
plt.xticks(ticks = np.linspace(0, 800, 5))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'Color')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')

# subplot4: turbidity versus chlx
ax = fig.add_subplot(3,5,4)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN12 == 'RESERVOIR', 'TURB_RESULT'], 
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title('Turbidity versus Chl a', loc = 'center')
plt.xticks(ticks = np.linspace(0, 500, 6))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'Turbidity')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')


# subplot5: tss versus chlx
ax = fig.add_subplot(3,5,5)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN12 == 'RESERVOIR', 'TSS_RESULT'],
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
nlaWaterChemClean = nlaWaterChem[~nlaWaterChem.TSS_RESULT.isna() & ~nlaWaterChem.CHLX_ugL.isna()]
lm = stats.linregress(nlaWaterChemClean.TSS_RESULT, nlaWaterChemClean.CHLX_ugL)
a, b, r, p, e = np.asarray(lm)
x = np.array([nlaWaterChemClean.TSS_RESULT.min(), nlaWaterChemClean.TSS_RESULT.max()])
plt.plot(x, a*x+b, linewidth = 6, color = 'blue')
plt.text(50, 100, 
         '$y$ = {:1.1f} $x$ + {:1.0f},\n$r^2$ = {:1.2f}, $p$ = {:1.4f}'.format(a,b, r**2, p))
plt.title('TSS versus Chl a', loc = 'center')
# plt.xticks(ticks = np.linspace(0, 100, 6))
plt.yticks(np.arange(0, 201, 50, dtype=int))
plt.xlabel(r'TSS')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')

# subplot6: doc versus chlx
ax = fig.add_subplot(3,5,6)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN12 == 'RESERVOIR', 'DOC_RESULT'],
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
nlaWaterChemClean = nlaWaterChem[~nlaWaterChem.DOC_RESULT.isna() & ~nlaWaterChem.CHLX_ugL.isna()]
lm = stats.linregress(nlaWaterChemClean.DOC_RESULT, nlaWaterChemClean.CHLX_ugL)
a, b, r, p, e = np.asarray(lm)
x = np.array([nlaWaterChemClean.DOC_RESULT.min(),nlaWaterChemClean.DOC_RESULT.max()])
plt.plot(x, a*x+b, linewidth = 6, color = 'blue')
plt.text(20, 200, 
         '$y$ = {:1.1f} $x$ + {:1.0f},\n$r^2$ = {:1.2f}, $p$ = {:1.4f}'.format(a,b, r**2, p))
plt.title('DOC versus Chl a', loc = 'center')
ax.set_xlim([-0.5, 40])
plt.xticks(ticks = np.linspace(0, 40, 5))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'DOC')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')

# subplot7: chll versus chlx
ax = fig.add_subplot(3,5,7)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLL_ugL'],
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
x = [0, 800]
plt.plot(x, x, linewidth = 6, color = 'blue', linestyle = '--')
plt.text(410, 400, r'1:1 line', rotation = 45)
plt.title('', loc = 'center')
plt.xticks(np.arange(0, 801, 200, dtype=int))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'Chl a (Edge)')
plt.ylabel(r'Chl a (Site)')


# subplot8: silica versus chlx
ax = fig.add_subplot(3,5,8)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN12 == 'RESERVOIR', 'SILICA_RESULT'],
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title('Silica versus Chl a', loc = 'center')
ax.set_xlim([0, 200])
plt.xticks(ticks = np.linspace(0, 200, 5))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'Silica')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')


# subplot9: totalP versus chlx
ax = fig.add_subplot(3,5,9)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN12 == 'RESERVOIR', 'PTL_RESULT'],
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
nlaWaterChemClean = nlaWaterChem[~nlaWaterChem.PTL_RESULT.isna() & ~nlaWaterChem.CHLX_ugL.isna()]
lm = stats.linregress(nlaWaterChemClean.PTL_RESULT, nlaWaterChemClean.CHLX_ugL)
a, b, r, p, e = np.asarray(lm)
x = np.array([nlaWaterChemClean.PTL_RESULT.min(), 2100])
plt.plot(x, a*x+b, linewidth = 6, color = 'blue')
plt.text(1000, 200, 
         '$y$ = {:1.1f} $x$ + {:1.0f},\n$r^2$ = {:1.2f}, $p$ = {:1.4f}'.format(a,b, r**2, p))
plt.title('totalP versus Chlx a', loc = 'center')
plt.xticks(ticks = np.linspace(0, 2100, 4))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'total P')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')



# subplot10: totalN versus chlx
ax = fig.add_subplot(3,5,10)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN12 == 'RESERVOIR', 'NTL_RESULT'], 
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
nlaWaterChemClean = nlaWaterChem[~nlaWaterChem.NTL_RESULT.isna() & ~nlaWaterChem.CHLX_ugL.isna()]
lm = stats.linregress(nlaWaterChemClean.NTL_RESULT, nlaWaterChemClean.CHLX_ugL)
a, b, r, p, e = np.asarray(lm)
x = np.array([nlaWaterChemClean.NTL_RESULT.min(),nlaWaterChemClean.NTL_RESULT.max()])
plt.plot(x, a*x+b, linewidth = 6, color = 'blue')
plt.text(5, 200, 
         '$y$ = {:1.0f} $x$ + {:1.0f},\n$r^2$ = {:1.2f}, $p$ = {:1.4f}'.format(a,b, r**2, p))
plt.title('total N versus Chl a', loc = 'center')
ax.set_xlim([0, 10])
plt.xticks(ticks = np.linspace(0, 10, 6))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'total N')
plt.ylabel(r'$\mathrm{Chl \/ (}\mu g \/ L^{-1})$')



# subplot11: NO3+NO2 versus chlx
ax = fig.add_subplot(3,5,11)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN12 == 'RESERVOIR', 'NITRATE_NITRITE_N_RESULT'],
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title('NO3 + NO2 versus Chlx a', loc = 'center')
ax.set_xlim([-0.5, 4])
plt.xticks(ticks = np.linspace(0, 4, 5))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'NO3 + NO2')
plt.ylabel(r'$\mathrm{Chl \/ (}\mu g \/ L^{-1})$')



# subplot12: nitrite versus chlx
ax = fig.add_subplot(3,5,12)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN12 == 'RESERVOIR', 'NITRITE_N_RESULT'],
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title('Nitrite versus Chl a', loc = 'center')
ax.set_xlim([-0.001, 0.02])
plt.xticks(ticks = np.linspace(0, 0.02, 5))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'Nitrite')
plt.ylabel(r'$\mathrm{Chl \/ (}\mu g \/ L^{-1})$')



# subplot13: nitrate versus chlx
ax = fig.add_subplot(3,5,13)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN12 == 'RESERVOIR', 'NITRATE_N_RESULT'], 
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title('Nitrate versus Chl a', loc = 'center')
ax.set_xlim([-0.5, 4])
plt.xticks(ticks = np.linspace(0, 4, 5))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'Nitrate')
plt.ylabel(r'$\mathrm{Chl \/ (}\mu g \/ L^{-1})$')


# subplot14: amonia versus chlx
ax = fig.add_subplot(3,5,14)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN12 == 'RESERVOIR', 'AMMONIA_N_RESULT'],
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title('Amonia versus Chl a', loc = 'center')
plt.xticks(ticks = np.linspace(0, 1.2, 5))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'Amonia')
plt.ylabel(r'$\mathrm{Chl \/ (}\mu g \/ L^{-1})$')

# subplot15: oxygen versus chlx
ax = fig.add_subplot(3,5,15)
ax.scatter(nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN12 == 'RESERVOIR', 'DO2_2M'],
           nlaWaterChem.loc[nlaWaterChem.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title(r'DO versus Chl a', loc = 'center')
plt.xticks(ticks = np.linspace(0, 25, 6))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'DO (mg/L)')
plt.ylabel(r'$\mathrm{Chl \/ (}\mu g \/ L^{-1})$')


plt.show()

fig.savefig(wd+'/output/figure/nutrientsChla/reservoir.png')


# In[55]:


siteDecTemp = pd.read_csv(wd+'/dataset/Lakes/siteNumMeas2plusDecTemp.csv')
nlaWaterChemDecTemp = nlaWaterChem[nlaWaterChem.UID.isin(siteDecTemp.UID)]


# In[62]:


fig = plt.figure(figsize=(60,30))

# subplot1: area versus chlx
ax = fig.add_subplot(3,5,1)
ax.scatter(nlaWaterChemDecTemp.loc[nlaWaterChemDecTemp.LAKE_ORIGIN12 == 'RESERVOIR', 'AREA_HA'], 
           nlaWaterChemDecTemp.loc[nlaWaterChemDecTemp.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title('Area versus Chl a', loc = 'center')
plt.xticks(ticks = np.linspace(0, 180000, 6), labels = thousandkCon(0, 180000, 6))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'Area (ha)')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')


# subplot2: elevation versus chlx
ax = fig.add_subplot(3,5,2)
ax.scatter(nlaWaterChemDecTemp.loc[nlaWaterChemDecTemp.LAKE_ORIGIN12 == 'RESERVOIR', 'ELEVATION'],
           nlaWaterChemDecTemp.loc[nlaWaterChemDecTemp.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title('Elevation versus Chl a', loc = 'center')
plt.xticks(ticks = np.linspace(0, 3600, 7))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'Elevation (m)')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')

# subplot3: color versus chlx
ax = fig.add_subplot(3,5,3)
ax.scatter(nlaWaterChemDecTemp.loc[nlaWaterChemDecTemp.LAKE_ORIGIN12 == 'RESERVOIR', 'COLOR_RESULT'], 
           nlaWaterChemDecTemp.loc[nlaWaterChemDecTemp.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title('Color versus Chl a', loc = 'center')
plt.xticks(ticks = np.linspace(0, 800, 5))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'Color')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')

# subplot4: turbidity versus chlx
ax = fig.add_subplot(3,5,4)
ax.scatter(nlaWaterChemDecTemp.loc[nlaWaterChemDecTemp.LAKE_ORIGIN12 == 'RESERVOIR', 'TURB_RESULT'], 
           nlaWaterChemDecTemp.loc[nlaWaterChemDecTemp.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title('Turbidity versus Chl a', loc = 'center')
plt.xticks(ticks = np.linspace(0, 500, 6))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'Turbidity')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')


# subplot5: tss versus chlx
ax = fig.add_subplot(3,5,5)
ax.scatter(nlaWaterChemDecTemp.loc[nlaWaterChemDecTemp.LAKE_ORIGIN12 == 'RESERVOIR', 'TSS_RESULT'],
           nlaWaterChemDecTemp.loc[nlaWaterChemDecTemp.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
nlaWaterChemClean = nlaWaterChemDecTemp[~nlaWaterChemDecTemp.TSS_RESULT.isna() & ~nlaWaterChemDecTemp.CHLX_ugL.isna()]
lm = stats.linregress(nlaWaterChemClean.TSS_RESULT, nlaWaterChemClean.CHLX_ugL)
a, b, r, p, e = np.asarray(lm)
x = np.array([nlaWaterChemClean.TSS_RESULT.min(), nlaWaterChemClean.TSS_RESULT.max()])
plt.plot(x, a*x+b, linewidth = 6, color = 'blue')
plt.text(50, 100, 
         '$y$ = {:1.1f} $x$ + {:1.0f},\n$r^2$ = {:1.2f}, $p$ = {:1.4f}'.format(a,b, r**2, p))
plt.title('TSS versus Chl a', loc = 'center')
# plt.xticks(ticks = np.linspace(0, 100, 6))
plt.yticks(np.arange(0, 201, 50, dtype=int))
plt.xlabel(r'TSS')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')

# subplot6: doc versus chlx
ax = fig.add_subplot(3,5,6)
ax.scatter(nlaWaterChemDecTemp.loc[nlaWaterChemDecTemp.LAKE_ORIGIN12 == 'RESERVOIR', 'DOC_RESULT'],
           nlaWaterChemDecTemp.loc[nlaWaterChemDecTemp.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
nlaWaterChemClean = nlaWaterChemDecTemp[~nlaWaterChemDecTemp.DOC_RESULT.isna() & ~nlaWaterChemDecTemp.CHLX_ugL.isna()]
lm = stats.linregress(nlaWaterChemClean.DOC_RESULT, nlaWaterChemClean.CHLX_ugL)
a, b, r, p, e = np.asarray(lm)
x = np.array([nlaWaterChemClean.DOC_RESULT.min(),nlaWaterChemClean.DOC_RESULT.max()])
plt.plot(x, a*x+b, linewidth = 6, color = 'blue')
plt.text(20, 200, 
         '$y$ = {:1.1f} $x$ + {:1.0f},\n$r^2$ = {:1.2f}, $p$ = {:1.4f}'.format(a,b, r**2, p))
plt.title('DOC versus Chl a', loc = 'center')
ax.set_xlim([-0.5, 40])
plt.xticks(ticks = np.linspace(0, 40, 5))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'DOC')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')

# subplot7: chll versus chlx
ax = fig.add_subplot(3,5,7)
ax.scatter(nlaWaterChemDecTemp.loc[nlaWaterChemDecTemp.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLL_ugL'],
           nlaWaterChemDecTemp.loc[nlaWaterChemDecTemp.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
x = [0, 800]
plt.plot(x, x, linewidth = 6, color = 'blue', linestyle = '--')
plt.text(410, 400, r'1:1 line', rotation = 45)
plt.title('', loc = 'center')
plt.xticks(np.arange(0, 801, 200, dtype=int))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'Chl a (Edge)')
plt.ylabel(r'Chl a (Site)')


# subplot8: silica versus chlx
ax = fig.add_subplot(3,5,8)
ax.scatter(nlaWaterChemDecTemp.loc[nlaWaterChemDecTemp.LAKE_ORIGIN12 == 'RESERVOIR', 'SILICA_RESULT'],
           nlaWaterChemDecTemp.loc[nlaWaterChemDecTemp.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title('Silica versus Chl a', loc = 'center')
ax.set_xlim([0, 200])
plt.xticks(ticks = np.linspace(0, 200, 5))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'Silica')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')


# subplot9: totalP versus chlx
ax = fig.add_subplot(3,5,9)
ax.scatter(nlaWaterChemDecTemp.loc[nlaWaterChemDecTemp.LAKE_ORIGIN12 == 'RESERVOIR', 'PTL_RESULT'],
           nlaWaterChemDecTemp.loc[nlaWaterChemDecTemp.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
nlaWaterChemClean = nlaWaterChemDecTemp[~nlaWaterChemDecTemp.PTL_RESULT.isna() & ~nlaWaterChemDecTemp.CHLX_ugL.isna()]
lm = stats.linregress(nlaWaterChemClean.PTL_RESULT, nlaWaterChemClean.CHLX_ugL)
a, b, r, p, e = np.asarray(lm)
x = np.array([nlaWaterChemClean.PTL_RESULT.min(), 2100])
plt.plot(x, a*x+b, linewidth = 6, color = 'blue')
plt.text(1000, 200, 
         '$y$ = {:1.1f} $x$ + {:1.0f},\n$r^2$ = {:1.2f}, $p$ = {:1.4f}'.format(a,b, r**2, p))
plt.title('totalP versus Chlx a', loc = 'center')
plt.xticks(ticks = np.linspace(0, 2100, 4))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'total P')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')



# subplot10: totalN versus chlx
ax = fig.add_subplot(3,5,10)
ax.scatter(nlaWaterChemDecTemp.loc[nlaWaterChemDecTemp.LAKE_ORIGIN12 == 'RESERVOIR', 'NTL_RESULT'], 
           nlaWaterChemDecTemp.loc[nlaWaterChemDecTemp.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
nlaWaterChemClean = nlaWaterChemDecTemp[~nlaWaterChemDecTemp.NTL_RESULT.isna() & ~nlaWaterChemDecTemp.CHLX_ugL.isna()]
lm = stats.linregress(nlaWaterChemClean.NTL_RESULT, nlaWaterChemClean.CHLX_ugL)
a, b, r, p, e = np.asarray(lm)
x = np.array([nlaWaterChemClean.NTL_RESULT.min(),nlaWaterChemClean.NTL_RESULT.max()])
plt.plot(x, a*x+b, linewidth = 6, color = 'blue')
plt.text(5, 200, 
         '$y$ = {:1.0f} $x$ + {:1.0f},\n$r^2$ = {:1.2f}, $p$ = {:1.4f}'.format(a,b, r**2, p))
plt.title('total N versus Chl a', loc = 'center')
ax.set_xlim([0, 10])
plt.xticks(ticks = np.linspace(0, 10, 6))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'total N')
plt.ylabel(r'$\mathrm{Chl \/ (}\mu g \/ L^{-1})$')



# subplot11: NO3+NO2 versus chlx
ax = fig.add_subplot(3,5,11)
ax.scatter(nlaWaterChemDecTemp.loc[nlaWaterChemDecTemp.LAKE_ORIGIN12 == 'RESERVOIR', 'NITRATE_NITRITE_N_RESULT'],
           nlaWaterChemDecTemp.loc[nlaWaterChemDecTemp.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title('NO3 + NO2 versus Chlx a', loc = 'center')
ax.set_xlim([-0.5, 4])
plt.xticks(ticks = np.linspace(0, 4, 5))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'NO3 + NO2')
plt.ylabel(r'$\mathrm{Chl \/ (}\mu g \/ L^{-1})$')



# subplot12: nitrite versus chlx
ax = fig.add_subplot(3,5,12)
ax.scatter(nlaWaterChemDecTemp.loc[nlaWaterChemDecTemp.LAKE_ORIGIN12 == 'RESERVOIR', 'NITRITE_N_RESULT'],
           nlaWaterChemDecTemp.loc[nlaWaterChemDecTemp.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title('Nitrite versus Chl a', loc = 'center')
ax.set_xlim([-0.001, 0.02])
plt.xticks(ticks = np.linspace(0, 0.02, 5))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'Nitrite')
plt.ylabel(r'$\mathrm{Chl \/ (}\mu g \/ L^{-1})$')



# subplot13: nitrate versus chlx
ax = fig.add_subplot(3,5,13)
ax.scatter(nlaWaterChemDecTemp.loc[nlaWaterChemDecTemp.LAKE_ORIGIN12 == 'RESERVOIR', 'NITRATE_N_RESULT'], 
           nlaWaterChemDecTemp.loc[nlaWaterChemDecTemp.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title('Nitrate versus Chl a', loc = 'center')
ax.set_xlim([-0.5, 4])
plt.xticks(ticks = np.linspace(0, 4, 5))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'Nitrate')
plt.ylabel(r'$\mathrm{Chl \/ (}\mu g \/ L^{-1})$')


# subplot14: amonia versus chlx
ax = fig.add_subplot(3,5,14)
ax.scatter(nlaWaterChemDecTemp.loc[nlaWaterChemDecTemp.LAKE_ORIGIN12 == 'RESERVOIR', 'AMMONIA_N_RESULT'],
           nlaWaterChemDecTemp.loc[nlaWaterChemDecTemp.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title('Amonia versus Chl a', loc = 'center')
plt.xticks(ticks = np.linspace(0, 1.2, 5))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'Amonia')
plt.ylabel(r'$\mathrm{Chl \/ (}\mu g \/ L^{-1})$')

# subplot15: oxygen versus chlx
ax = fig.add_subplot(3,5,15)
ax.scatter(nlaWaterChemDecTemp.loc[nlaWaterChemDecTemp.LAKE_ORIGIN12 == 'RESERVOIR', 'DO2_2M'],
           nlaWaterChemDecTemp.loc[nlaWaterChemDecTemp.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title(r'DO versus Chl a', loc = 'center')
plt.xticks(ticks = np.linspace(0, 25, 6))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'DO (mg/L)')
plt.ylabel(r'$\mathrm{Chl \/ (}\mu g \/ L^{-1})$')


plt.show()

fig.savefig(wd+'/output/figure/nutrientsChla/reservoirDecreaseTemp.png')


# In[65]:


nlaWaterChemDecTemp20degPlus = nlaWaterChemDecTemp[nlaWaterChemDecTemp.TEMPERATURE >= 20]


# In[66]:


fig = plt.figure(figsize=(60,30))

# subplot1: area versus chlx
ax = fig.add_subplot(3,5,1)
ax.scatter(nlaWaterChemDecTemp20degPlus.loc[nlaWaterChemDecTemp20degPlus.LAKE_ORIGIN12 == 'RESERVOIR', 'AREA_HA'], 
           nlaWaterChemDecTemp20degPlus.loc[nlaWaterChemDecTemp20degPlus.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title('Area versus Chl a', loc = 'center')
plt.xticks(ticks = np.linspace(0, 180000, 6), labels = thousandkCon(0, 180000, 6))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'Area (ha)')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')


# subplot2: elevation versus chlx
ax = fig.add_subplot(3,5,2)
ax.scatter(nlaWaterChemDecTemp.loc[nlaWaterChemDecTemp.LAKE_ORIGIN12 == 'RESERVOIR', 'ELEVATION'],
           nlaWaterChemDecTemp.loc[nlaWaterChemDecTemp.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title('Elevation versus Chl a', loc = 'center')
plt.xticks(ticks = np.linspace(0, 3600, 7))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'Elevation (m)')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')

# subplot3: color versus chlx
ax = fig.add_subplot(3,5,3)
ax.scatter(nlaWaterChemDecTemp20degPlus.loc[nlaWaterChemDecTemp20degPlus.LAKE_ORIGIN12 == 'RESERVOIR', 'COLOR_RESULT'], 
           nlaWaterChemDecTemp20degPlus.loc[nlaWaterChemDecTemp20degPlus.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title('Color versus Chl a', loc = 'center')
plt.xticks(ticks = np.linspace(0, 800, 5))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'Color')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')

# subplot4: turbidity versus chlx
ax = fig.add_subplot(3,5,4)
ax.scatter(nlaWaterChemDecTemp20degPlus.loc[nlaWaterChemDecTemp20degPlus.LAKE_ORIGIN12 == 'RESERVOIR', 'TURB_RESULT'], 
           nlaWaterChemDecTemp20degPlus.loc[nlaWaterChemDecTemp20degPlus.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title('Turbidity versus Chl a', loc = 'center')
plt.xticks(ticks = np.linspace(0, 500, 6))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'Turbidity')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')


# subplot5: tss versus chlx
ax = fig.add_subplot(3,5,5)
ax.scatter(nlaWaterChemDecTemp20degPlus.loc[nlaWaterChemDecTemp20degPlus.LAKE_ORIGIN12 == 'RESERVOIR', 'TSS_RESULT'],
           nlaWaterChemDecTemp20degPlus.loc[nlaWaterChemDecTemp20degPlus.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
nlaWaterChemClean = nlaWaterChemDecTemp20degPlus[~nlaWaterChemDecTemp20degPlus.TSS_RESULT.isna() & ~nlaWaterChemDecTemp20degPlus.CHLX_ugL.isna()]
lm = stats.linregress(nlaWaterChemClean.TSS_RESULT, nlaWaterChemClean.CHLX_ugL)
a, b, r, p, e = np.asarray(lm)
x = np.array([nlaWaterChemClean.TSS_RESULT.min(), nlaWaterChemClean.TSS_RESULT.max()])
plt.plot(x, a*x+b, linewidth = 6, color = 'blue')
plt.text(50, 100, 
         '$y$ = {:1.1f} $x$ + {:1.0f},\n$r^2$ = {:1.2f}, $p$ = {:1.4f}'.format(a,b, r**2, p))
plt.title('TSS versus Chl a', loc = 'center')
# plt.xticks(ticks = np.linspace(0, 100, 6))
plt.yticks(np.arange(0, 201, 50, dtype=int))
plt.xlabel(r'TSS')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')

# subplot6: doc versus chlx
ax = fig.add_subplot(3,5,6)
ax.scatter(nlaWaterChemDecTemp20degPlus.loc[nlaWaterChemDecTemp20degPlus.LAKE_ORIGIN12 == 'RESERVOIR', 'DOC_RESULT'],
           nlaWaterChemDecTemp20degPlus.loc[nlaWaterChemDecTemp20degPlus.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
nlaWaterChemClean = nlaWaterChemDecTemp20degPlus[~nlaWaterChemDecTemp20degPlus.DOC_RESULT.isna() & ~nlaWaterChemDecTemp20degPlus.CHLX_ugL.isna()]
lm = stats.linregress(nlaWaterChemClean.DOC_RESULT, nlaWaterChemClean.CHLX_ugL)
a, b, r, p, e = np.asarray(lm)
x = np.array([nlaWaterChemClean.DOC_RESULT.min(),nlaWaterChemClean.DOC_RESULT.max()])
plt.plot(x, a*x+b, linewidth = 6, color = 'blue')
plt.text(20, 200, 
         '$y$ = {:1.1f} $x$ + {:1.0f},\n$r^2$ = {:1.2f}, $p$ = {:1.4f}'.format(a,b, r**2, p))
plt.title('DOC versus Chl a', loc = 'center')
ax.set_xlim([-0.5, 40])
plt.xticks(ticks = np.linspace(0, 40, 5))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'DOC')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')

# subplot7: chll versus chlx
ax = fig.add_subplot(3,5,7)
ax.scatter(nlaWaterChemDecTemp20degPlus.loc[nlaWaterChemDecTemp20degPlus.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLL_ugL'],
           nlaWaterChemDecTemp20degPlus.loc[nlaWaterChemDecTemp20degPlus.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
x = [0, 800]
plt.plot(x, x, linewidth = 6, color = 'blue', linestyle = '--')
plt.text(410, 400, r'1:1 line', rotation = 45)
plt.title('', loc = 'center')
plt.xticks(np.arange(0, 801, 200, dtype=int))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'Chl a (Edge)')
plt.ylabel(r'Chl a (Site)')


# subplot8: silica versus chlx
ax = fig.add_subplot(3,5,8)
ax.scatter(nlaWaterChemDecTemp20degPlus.loc[nlaWaterChemDecTemp20degPlus.LAKE_ORIGIN12 == 'RESERVOIR', 'SILICA_RESULT'],
           nlaWaterChemDecTemp20degPlus.loc[nlaWaterChemDecTemp20degPlus.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title('Silica versus Chl a', loc = 'center')
ax.set_xlim([0, 200])
plt.xticks(ticks = np.linspace(0, 200, 5))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'Silica')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')


# subplot9: totalP versus chlx
ax = fig.add_subplot(3,5,9)
ax.scatter(nlaWaterChemDecTemp20degPlus.loc[nlaWaterChemDecTemp20degPlus.LAKE_ORIGIN12 == 'RESERVOIR', 'PTL_RESULT'],
           nlaWaterChemDecTemp20degPlus.loc[nlaWaterChemDecTemp20degPlus.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
nlaWaterChemClean = nlaWaterChemDecTemp20degPlus[~nlaWaterChemDecTemp20degPlus.PTL_RESULT.isna() & ~nlaWaterChemDecTemp20degPlus.CHLX_ugL.isna()]
lm = stats.linregress(nlaWaterChemClean.PTL_RESULT, nlaWaterChemClean.CHLX_ugL)
a, b, r, p, e = np.asarray(lm)
x = np.array([nlaWaterChemClean.PTL_RESULT.min(), 2100])
plt.plot(x, a*x+b, linewidth = 6, color = 'blue')
plt.text(1000, 200, 
         '$y$ = {:1.1f} $x$ + {:1.0f},\n$r^2$ = {:1.2f}, $p$ = {:1.4f}'.format(a,b, r**2, p))
plt.title('totalP versus Chlx a', loc = 'center')
plt.xticks(ticks = np.linspace(0, 2100, 4))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'total P')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')



# subplot10: totalN versus chlx
ax = fig.add_subplot(3,5,10)
ax.scatter(nlaWaterChemDecTemp20degPlus.loc[nlaWaterChemDecTemp20degPlus.LAKE_ORIGIN12 == 'RESERVOIR', 'NTL_RESULT'], 
           nlaWaterChemDecTemp20degPlus.loc[nlaWaterChemDecTemp20degPlus.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
nlaWaterChemClean = nlaWaterChemDecTemp20degPlus[~nlaWaterChemDecTemp20degPlus.NTL_RESULT.isna() & ~nlaWaterChemDecTemp20degPlus.CHLX_ugL.isna()]
lm = stats.linregress(nlaWaterChemClean.NTL_RESULT, nlaWaterChemClean.CHLX_ugL)
a, b, r, p, e = np.asarray(lm)
x = np.array([nlaWaterChemClean.NTL_RESULT.min(),nlaWaterChemClean.NTL_RESULT.max()])
plt.plot(x, a*x+b, linewidth = 6, color = 'blue')
plt.text(5, 200, 
         '$y$ = {:1.0f} $x$ + {:1.0f},\n$r^2$ = {:1.2f}, $p$ = {:1.4f}'.format(a,b, r**2, p))
plt.title('total N versus Chl a', loc = 'center')
ax.set_xlim([0, 10])
plt.xticks(ticks = np.linspace(0, 10, 6))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'total N')
plt.ylabel(r'$\mathrm{Chl \/ (}\mu g \/ L^{-1})$')



# subplot11: NO3+NO2 versus chlx
ax = fig.add_subplot(3,5,11)
ax.scatter(nlaWaterChemDecTemp20degPlus.loc[nlaWaterChemDecTemp20degPlus.LAKE_ORIGIN12 == 'RESERVOIR', 'NITRATE_NITRITE_N_RESULT'],
           nlaWaterChemDecTemp20degPlus.loc[nlaWaterChemDecTemp20degPlus.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title('NO3 + NO2 versus Chlx a', loc = 'center')
ax.set_xlim([-0.5, 4])
plt.xticks(ticks = np.linspace(0, 4, 5))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'NO3 + NO2')
plt.ylabel(r'$\mathrm{Chl \/ (}\mu g \/ L^{-1})$')



# subplot12: nitrite versus chlx
ax = fig.add_subplot(3,5,12)
ax.scatter(nlaWaterChemDecTemp20degPlus.loc[nlaWaterChemDecTemp20degPlus.LAKE_ORIGIN12 == 'RESERVOIR', 'NITRITE_N_RESULT'],
           nlaWaterChemDecTemp20degPlus.loc[nlaWaterChemDecTemp20degPlus.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title('Nitrite versus Chl a', loc = 'center')
ax.set_xlim([-0.001, 0.02])
plt.xticks(ticks = np.linspace(0, 0.02, 5))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'Nitrite')
plt.ylabel(r'$\mathrm{Chl \/ (}\mu g \/ L^{-1})$')



# subplot13: nitrate versus chlx
ax = fig.add_subplot(3,5,13)
ax.scatter(nlaWaterChemDecTemp20degPlus.loc[nlaWaterChemDecTemp20degPlus.LAKE_ORIGIN12 == 'RESERVOIR', 'NITRATE_N_RESULT'], 
           nlaWaterChemDecTemp20degPlus.loc[nlaWaterChemDecTemp20degPlus.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title('Nitrate versus Chl a', loc = 'center')
ax.set_xlim([-0.5, 4])
plt.xticks(ticks = np.linspace(0, 4, 5))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'Nitrate')
plt.ylabel(r'$\mathrm{Chl \/ (}\mu g \/ L^{-1})$')


# subplot14: amonia versus chlx
ax = fig.add_subplot(3,5,14)
ax.scatter(nlaWaterChemDecTemp20degPlus.loc[nlaWaterChemDecTemp20degPlus.LAKE_ORIGIN12 == 'RESERVOIR', 'AMMONIA_N_RESULT'],
           nlaWaterChemDecTemp20degPlus.loc[nlaWaterChemDecTemp20degPlus.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title('Amonia versus Chl a', loc = 'center')
plt.xticks(ticks = np.linspace(0, 1.2, 5))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'Amonia')
plt.ylabel(r'$\mathrm{Chl \/ (}\mu g \/ L^{-1})$')

# subplot15: oxygen versus chlx
ax = fig.add_subplot(3,5,15)
ax.scatter(nlaWaterChemDecTemp20degPlus.loc[nlaWaterChemDecTemp20degPlus.LAKE_ORIGIN12 == 'RESERVOIR', 'DO2_2M'],
           nlaWaterChemDecTemp20degPlus.loc[nlaWaterChemDecTemp20degPlus.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title(r'DO versus Chl a', loc = 'center')
plt.xticks(ticks = np.linspace(0, 25, 6))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'DO (mg/L)')
plt.ylabel(r'$\mathrm{Chl \/ (}\mu g \/ L^{-1})$')


plt.show()

fig.savefig(wd+'/output/figure/nutrientsChla/reservoirDecreaseTemp20degPlus.png')


# In[78]:


nlaWaterChemDecTempBig = nlaWaterChemDecTemp[nlaWaterChemDecTemp.AREA_HA >= 5]
nlaWaterChemDecTempBig.shape


# In[81]:


nlaWaterChemDecTemp.AREA_HA.min()


# In[82]:


fig = plt.figure(figsize=(60,30))

# subplot1: area versus chlx
ax = fig.add_subplot(3,5,1)
ax.scatter(nlaWaterChemDecTempBig.loc[nlaWaterChemDecTempBig.LAKE_ORIGIN12 == 'RESERVOIR', 'AREA_HA'], 
           nlaWaterChemDecTempBig.loc[nlaWaterChemDecTempBig.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title('Area versus Chl a', loc = 'center')
plt.xticks(ticks = np.linspace(0, 180000, 6), labels = thousandkCon(0, 180000, 6))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'Area (ha)')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')


# subplot2: elevation versus chlx
ax = fig.add_subplot(3,5,2)
ax.scatter(nlaWaterChemDecTempBig.loc[nlaWaterChemDecTempBig.LAKE_ORIGIN12 == 'RESERVOIR', 'ELEVATION'],
           nlaWaterChemDecTempBig.loc[nlaWaterChemDecTempBig.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title('Elevation versus Chl a', loc = 'center')
plt.xticks(ticks = np.linspace(0, 3600, 7))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'Elevation (m)')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')

# subplot3: color versus chlx
ax = fig.add_subplot(3,5,3)
ax.scatter(nlaWaterChemDecTempBig.loc[nlaWaterChemDecTempBig.LAKE_ORIGIN12 == 'RESERVOIR', 'COLOR_RESULT'], 
           nlaWaterChemDecTempBig.loc[nlaWaterChemDecTempBig.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title('Color versus Chl a', loc = 'center')
plt.xticks(ticks = np.linspace(0, 800, 5))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'Color')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')

# subplot4: turbidity versus chlx
ax = fig.add_subplot(3,5,4)
ax.scatter(nlaWaterChemDecTempBig.loc[nlaWaterChemDecTempBig.LAKE_ORIGIN12 == 'RESERVOIR', 'TURB_RESULT'], 
           nlaWaterChemDecTempBig.loc[nlaWaterChemDecTempBig.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title('Turbidity versus Chl a', loc = 'center')
plt.xticks(ticks = np.linspace(0, 500, 6))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'Turbidity')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')


# subplot5: tss versus chlx
ax = fig.add_subplot(3,5,5)
ax.scatter(nlaWaterChemDecTempBig.loc[nlaWaterChemDecTempBig.LAKE_ORIGIN12 == 'RESERVOIR', 'TSS_RESULT'],
           nlaWaterChemDecTempBig.loc[nlaWaterChemDecTempBig.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
nlaWaterChemClean = nlaWaterChemDecTempBig[~nlaWaterChemDecTempBig.TSS_RESULT.isna() & ~nlaWaterChemDecTempBig.CHLX_ugL.isna()]
lm = stats.linregress(nlaWaterChemClean.TSS_RESULT, nlaWaterChemClean.CHLX_ugL)
a, b, r, p, e = np.asarray(lm)
x = np.array([nlaWaterChemClean.TSS_RESULT.min(), nlaWaterChemClean.TSS_RESULT.max()])
plt.plot(x, a*x+b, linewidth = 6, color = 'blue')
plt.text(50, 100, 
         '$y$ = {:1.1f} $x$ + {:1.0f},\n$r^2$ = {:1.2f}, $p$ = {:1.4f}'.format(a,b, r**2, p))
plt.title('TSS versus Chl a', loc = 'center')
# plt.xticks(ticks = np.linspace(0, 100, 6))
plt.yticks(np.arange(0, 201, 50, dtype=int))
plt.xlabel(r'TSS')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')

# subplot6: doc versus chlx
ax = fig.add_subplot(3,5,6)
ax.scatter(nlaWaterChemDecTempBig.loc[nlaWaterChemDecTempBig.LAKE_ORIGIN12 == 'RESERVOIR', 'DOC_RESULT'],
           nlaWaterChemDecTempBig.loc[nlaWaterChemDecTempBig.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
nlaWaterChemClean = nlaWaterChemDecTempBig[~nlaWaterChemDecTempBig.DOC_RESULT.isna() & ~nlaWaterChemDecTempBig.CHLX_ugL.isna()]
lm = stats.linregress(nlaWaterChemClean.DOC_RESULT, nlaWaterChemClean.CHLX_ugL)
a, b, r, p, e = np.asarray(lm)
x = np.array([nlaWaterChemClean.DOC_RESULT.min(),nlaWaterChemClean.DOC_RESULT.max()])
plt.plot(x, a*x+b, linewidth = 6, color = 'blue')
plt.text(20, 200, 
         '$y$ = {:1.1f} $x$ + {:1.0f},\n$r^2$ = {:1.2f}, $p$ = {:1.4f}'.format(a,b, r**2, p))
plt.title('DOC versus Chl a', loc = 'center')
ax.set_xlim([-0.5, 40])
plt.xticks(ticks = np.linspace(0, 40, 5))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'DOC')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')

# subplot7: chll versus chlx
ax = fig.add_subplot(3,5,7)
ax.scatter(nlaWaterChemDecTempBig.loc[nlaWaterChemDecTempBig.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLL_ugL'],
           nlaWaterChemDecTempBig.loc[nlaWaterChemDecTempBig.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
x = [0, 800]
plt.plot(x, x, linewidth = 6, color = 'blue', linestyle = '--')
plt.text(410, 400, r'1:1 line', rotation = 45)
plt.title('', loc = 'center')
plt.xticks(np.arange(0, 801, 200, dtype=int))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'Chl a (Edge)')
plt.ylabel(r'Chl a (Site)')


# subplot8: silica versus chlx
ax = fig.add_subplot(3,5,8)
ax.scatter(nlaWaterChemDecTempBig.loc[nlaWaterChemDecTempBig.LAKE_ORIGIN12 == 'RESERVOIR', 'SILICA_RESULT'],
           nlaWaterChemDecTempBig.loc[nlaWaterChemDecTempBig.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title('Silica versus Chl a', loc = 'center')
ax.set_xlim([0, 200])
plt.xticks(ticks = np.linspace(0, 200, 5))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'Silica')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')


# subplot9: totalP versus chlx
ax = fig.add_subplot(3,5,9)
ax.scatter(nlaWaterChemDecTempBig.loc[nlaWaterChemDecTempBig.LAKE_ORIGIN12 == 'RESERVOIR', 'PTL_RESULT'],
           nlaWaterChemDecTempBig.loc[nlaWaterChemDecTempBig.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
nlaWaterChemClean = nlaWaterChemDecTempBig[~nlaWaterChemDecTempBig.PTL_RESULT.isna() & ~nlaWaterChemDecTempBig.CHLX_ugL.isna()]
lm = stats.linregress(nlaWaterChemClean.PTL_RESULT, nlaWaterChemClean.CHLX_ugL)
a, b, r, p, e = np.asarray(lm)
x = np.array([nlaWaterChemClean.PTL_RESULT.min(), 2100])
plt.plot(x, a*x+b, linewidth = 6, color = 'blue')
plt.text(1000, 200, 
         '$y$ = {:1.1f} $x$ + {:1.0f},\n$r^2$ = {:1.2f}, $p$ = {:1.4f}'.format(a,b, r**2, p))
plt.title('totalP versus Chlx a', loc = 'center')
plt.xticks(ticks = np.linspace(0, 2100, 4))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'total P')
plt.ylabel(r'$\mathrm{chl \/ (}\mu g \/ L^{-1})$')



# subplot10: totalN versus chlx
ax = fig.add_subplot(3,5,10)
ax.scatter(nlaWaterChemDecTempBig.loc[nlaWaterChemDecTempBig.LAKE_ORIGIN12 == 'RESERVOIR', 'NTL_RESULT'], 
           nlaWaterChemDecTempBig.loc[nlaWaterChemDecTempBig.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
nlaWaterChemClean = nlaWaterChemDecTempBig[~nlaWaterChemDecTempBig.NTL_RESULT.isna() & ~nlaWaterChemDecTempBig.CHLX_ugL.isna()]
lm = stats.linregress(nlaWaterChemClean.NTL_RESULT, nlaWaterChemClean.CHLX_ugL)
a, b, r, p, e = np.asarray(lm)
x = np.array([nlaWaterChemClean.NTL_RESULT.min(),nlaWaterChemClean.NTL_RESULT.max()])
plt.plot(x, a*x+b, linewidth = 6, color = 'blue')
plt.text(5, 200, 
         '$y$ = {:1.0f} $x$ + {:1.0f},\n$r^2$ = {:1.2f}, $p$ = {:1.4f}'.format(a,b, r**2, p))
plt.title('total N versus Chl a', loc = 'center')
ax.set_xlim([0, 10])
plt.xticks(ticks = np.linspace(0, 10, 6))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'total N')
plt.ylabel(r'$\mathrm{Chl \/ (}\mu g \/ L^{-1})$')



# subplot11: NO3+NO2 versus chlx
ax = fig.add_subplot(3,5,11)
ax.scatter(nlaWaterChemDecTempBig.loc[nlaWaterChemDecTempBig.LAKE_ORIGIN12 == 'RESERVOIR', 'NITRATE_NITRITE_N_RESULT'],
           nlaWaterChemDecTempBig.loc[nlaWaterChemDecTempBig.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title('NO3 + NO2 versus Chlx a', loc = 'center')
ax.set_xlim([-0.5, 4])
plt.xticks(ticks = np.linspace(0, 4, 5))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'NO3 + NO2')
plt.ylabel(r'$\mathrm{Chl \/ (}\mu g \/ L^{-1})$')



# subplot12: nitrite versus chlx
ax = fig.add_subplot(3,5,12)
ax.scatter(nlaWaterChemDecTempBig.loc[nlaWaterChemDecTempBig.LAKE_ORIGIN12 == 'RESERVOIR', 'NITRITE_N_RESULT'],
           nlaWaterChemDecTempBig.loc[nlaWaterChemDecTempBig.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title('Nitrite versus Chl a', loc = 'center')
ax.set_xlim([-0.001, 0.02])
plt.xticks(ticks = np.linspace(0, 0.02, 5))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'Nitrite')
plt.ylabel(r'$\mathrm{Chl \/ (}\mu g \/ L^{-1})$')



# subplot13: nitrate versus chlx
ax = fig.add_subplot(3,5,13)
ax.scatter(nlaWaterChemDecTempBig.loc[nlaWaterChemDecTempBig.LAKE_ORIGIN12 == 'RESERVOIR', 'NITRATE_N_RESULT'], 
           nlaWaterChemDecTempBig.loc[nlaWaterChemDecTempBig.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title('Nitrate versus Chl a', loc = 'center')
ax.set_xlim([-0.5, 4])
plt.xticks(ticks = np.linspace(0, 4, 5))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'Nitrate')
plt.ylabel(r'$\mathrm{Chl \/ (}\mu g \/ L^{-1})$')


# subplot14: amonia versus chlx
ax = fig.add_subplot(3,5,14)
ax.scatter(nlaWaterChemDecTempBig.loc[nlaWaterChemDecTempBig.LAKE_ORIGIN12 == 'RESERVOIR', 'AMMONIA_N_RESULT'],
           nlaWaterChemDecTempBig.loc[nlaWaterChemDecTempBig.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title('Amonia versus Chl a', loc = 'center')
plt.xticks(ticks = np.linspace(0, 1.2, 5))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'Amonia')
plt.ylabel(r'$\mathrm{Chl \/ (}\mu g \/ L^{-1})$')

# subplot15: oxygen versus chlx
ax = fig.add_subplot(3,5,15)
ax.scatter(nlaWaterChemDecTempBig.loc[nlaWaterChemDecTempBig.LAKE_ORIGIN12 == 'RESERVOIR', 'DO2_2M'],
           nlaWaterChemDecTempBig.loc[nlaWaterChemDecTempBig.LAKE_ORIGIN12 == 'RESERVOIR', 'CHLX_ugL'], s = 160, c = 'blue')
plt.title(r'DO versus Chl a', loc = 'center')
plt.xticks(ticks = np.linspace(0, 25, 6))
plt.yticks(np.arange(0, 801, 200, dtype=int))
plt.xlabel(r'DO (mg/L)')
plt.ylabel(r'$\mathrm{Chl \/ (}\mu g \/ L^{-1})$')


plt.show()

fig.savefig(wd+'/output/figure/nutrientsChla/reservoirDecreaseTempBig.png')

