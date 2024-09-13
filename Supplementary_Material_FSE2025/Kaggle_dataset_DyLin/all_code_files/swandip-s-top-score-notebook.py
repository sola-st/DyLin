#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "/kaggle/input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import numpy as np
import pandas as pd

from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
autogluon_preds_df = pd.read_csv('/kaggle/input/autogloun-t8-dslanders/submission.csv')
xgb_preds_df = pd.read_csv('/kaggle/input/0-98530-can-you-eat/submission.csv')
x_df= pd.read_csv("/kaggle/input/19-19-ensemble-of-top-2/submission.csv")

autogluon_preds = autogluon_preds_df['class']
xgb_preds = xgb_preds_df['class']
x_preds=x_df['class']
le = LabelEncoder()

autogluon_preds = le.fit_transform(autogluon_preds)
xgb_preds = le.fit_transform(xgb_preds)
x_preds=le.transform(x_preds)

autogluon_score = 0.98530
xgb_score = 0.98530
x_score= 0.98531

avg_ensemble_preds = (autogluon_preds + xgb_preds + x_preds +x_score) / 4

final_preds_avg_ensemble = np.round(avg_ensemble_preds).astype(int)
final_preds_avg_ensemble = le.inverse_transform(final_preds_avg_ensemble)

avg_ensemble_output = pd.DataFrame({
    'id': autogluon_preds_df['id'],
    'class': final_preds_avg_ensemble
})

avg_ensemble_output.head(2)


# In[ ]:


avg_ensemble_output.to_csv('submission.csv', index=False)
avg_ensemble_output['class'].value_counts()

