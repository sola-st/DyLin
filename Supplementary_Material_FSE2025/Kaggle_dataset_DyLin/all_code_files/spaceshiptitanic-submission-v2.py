#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # **FOREWORD**

# This is a short kernel aimed to submit multiple prediction files from my yester kernel in this regard- <br>
# 1. https://www.kaggle.com/code/ravi20076/spaceshiptitanic-autogluon-v1 <br>
# 2. https://www.kaggle.com/code/ravi20076/spaceshiptitanic-models-v2 <br> 
# 3. https://www.kaggle.com/code/ravi20076/spaceshiptitanic-lightautoml-v1 <br>
# 
# I create the [below dataset](https://www.kaggle.com/datasets/ravi20076/spaceshiptitanicsubfilesv1) with these prediction files and use them to submit to the competition. <br>
# I usually engender this process to keep track of my submissions and prevent cluttered files in my final submission work. This is incredibly useful in code competitions as well. <br>
# 
# Wishing you all the best for the competition!

# # **SUBMISSION**

# In[ ]:


import pandas as pd, numpy as np

sub_fl    = pd.read_csv(f"/kaggle/input/spaceship-titanic/sample_submission.csv")
Mdl_Preds = pd.read_csv(f"/kaggle/input/spaceshiptitanicsubfilesv1/Submission_AG_LAMA.csv")

display(Mdl_Preds.head(10).style.set_caption(f"Model predictions"))


# In[ ]:


#get_ipython().run_cell_magic('time', '', '\nsub_fl["Transported"] = \\\nMdl_Preds["Blend_AGV1_1"]   | Mdl_Preds["Blend_AGV1_2"] | \\\nMdl_Preds["Blend_AGV1_3"]   | Mdl_Preds["Blend_AGV1_6"] | Mdl_Preds["Blend_AGV1_7"] | \\\nMdl_Preds["Blend_LAMAV1_1"] | Mdl_Preds["Blend_LAMAV1_2"]\n\nsub_fl.to_csv(f"submission.csv", index = False)\nprint()\ndisplay(sub_fl.head(10).style.set_caption(f"Submission file"))\nprint()\n\n!ls\n!head submission.csv')

