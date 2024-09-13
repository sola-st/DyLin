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


#get_ipython().system('pip install autogluon')


# In[ ]:


import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

# Load your data
train_data = TabularDataset('/kaggle/input/spaceship-titanic/train.csv')
test_data = TabularDataset('/kaggle/input/spaceship-titanic/test.csv')

# Specify the target column
label = 'Transported'  # Replace with your actual target column name

# Initialize the predictor
predictor = TabularPredictor(
    label=label,
    eval_metric='accuracy',  # Set accuracy as the evaluation metric
    problem_type='binary'
).fit(
    train_data=train_data,
    presets='best_quality',  # You can change this to 'high_quality_fast_inference_only_refit' or 'optimize_for_deployment' if needed
    time_limit=3600/2  # Set a time limit for training (in seconds)
)

# Make predictions on the test data
predictions = predictor.predict(test_data)

# Evaluate the predictions
# performance = predictor.evaluate(test_data)
# print(performance)

# Optionally, save the model
# predictor.save('autogluon_model')


# In[ ]:


sub = pd.read_csv("/kaggle/input/spaceship-titanic/sample_submission.csv")
sub['Transported'] = predictions
sub.to_csv("submission_spaceship_titanic.csv", index=False)


# In[ ]:


sub.head()


# In[ ]:




