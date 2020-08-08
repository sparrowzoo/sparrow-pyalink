#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os

def get_workspace_path(path):
    cwd = os.getcwd()
    end_index = len(cwd)-len(path)
    workspace=cwd[0:end_index]
    return workspace



print get_workspace_path("path.py")