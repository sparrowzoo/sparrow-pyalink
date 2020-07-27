#!/usr/bin/env python
# coding: utf-8

# In[46]:


import pandas as pd
from sparrow.tools.path import get_workspace_path
workspace=get_workspace_path("source/sparrow/feature")
print(workspace)


# In[45]:


listen_count=pd.read_csv(workspace+"dataset/csv/user_listen_count.csv",header=None,decimal=',')
listen_count[2]=1
print(listen_count)


