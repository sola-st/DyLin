#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # **FOREWORD**

# This kernel is a feeder for other public/ private work. It aims to install the wheel files for the below packages for my latest work in the competition- <br>
# 
# |Package Label|Version      |Folder location <br> Wheel files|
# |-------------| :-:         | ----------                     |
# |Autogluon-Tabular| 1.1.1   | AG111    |
# |LightAutoML      | 0.3.8.1 | LAMA0381 |
# |LightGBM         | 4.5.0   | LGBM450  | 

# # **INSTALLATIONS**

# In[ ]:


import os

try:
    os.mkdir("AG111")
except:
    pass
#get_ipython().system('pip -q download autogluon.tabular -d "/kaggle/working/AG111"')

try:
    os.mkdir("LAMA0381")
except:
    pass
#get_ipython().system('pip -q download lightautoml -d "/kaggle/working/LAMA0381"')


# In[ ]:


#get_ipython().run_cell_magic('writefile', 'requirements.txt', 'lightgbm==4.5.0\nnumpy==1.26.4\nscipy==1.11.4')


# In[ ]:


try:
    os.mkdir("/kaggle/working/LGBM450");
except:
    pass;
#get_ipython().system('pip -q download -r requirements.txt -d "/kaggle/working/LGBM450"')

