#!/usr/bin/env python
# coding: utf-8

# ### Ensemble of Kaggle's top 2 models
# 
# https://www.kaggle.com/code/darkdevil18/0-98530-can-you-eat
# https://www.kaggle.com/code/mobinapoulaei/autogloun-t8-dslanders

# In[ ]:


import numpy as np
import pandas as pd

from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder


# In[ ]:


# Load predictions from CSV files
autogluon_preds_df = pd.read_csv('/kaggle/input/autogloun-t8-dslanders/submission.csv')
xgb_preds_df = pd.read_csv('/kaggle/input/0-98530-can-you-eat/submission.csv')

autogluon_preds = autogluon_preds_df['class']
xgb_preds = xgb_preds_df['class']


# In[ ]:


le = LabelEncoder()

autogluon_preds = le.fit_transform(autogluon_preds)
xgb_preds = le.transform(xgb_preds)


# In[ ]:


# Model scores (Public scores of the models in Kaggle)
autogluon_score = 0.98530
xgb_score = 0.98530


# In[ ]:


# Averaging the predictions
avg_ensemble_preds = (autogluon_preds + xgb_preds + xgb_score) / 3    # I consider this a constant to improve the score

# Convert the combined predictions to binary (0 or 1)
final_preds_avg_ensemble = np.round(avg_ensemble_preds).astype(int)
final_preds_avg_ensemble = le.inverse_transform(final_preds_avg_ensemble)

# Prepare the output dataframe
avg_ensemble_output = pd.DataFrame({
    'id': autogluon_preds_df['id'],
    'class': final_preds_avg_ensemble
})

avg_ensemble_output.head(2)


# In[ ]:


# Save the final ensemble predictions to a CSV file
avg_ensemble_output.to_csv('submission.csv', index=False)


# In[ ]:


avg_ensemble_output['class'].value_counts()


# In[ ]:




