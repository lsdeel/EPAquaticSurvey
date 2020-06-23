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


# In[6]:


nlaWaterChem = pd.read_csv(wd+'/dataset/Lakes/nlaWaterChem.csv')
nlaWaterChem.columns


# In[12]:


for row, col in nlaWaterChem[['LAKE_ORIGIN', 'LAKE_ORIGIN12']].iterrows():
    print(col)


# In[ ]:




