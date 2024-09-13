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


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

train = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')
test = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')


# In[ ]:


features = ['HomePlanet', 'CryoSleep', 'Age', 'VIP', 'RoomService', 'FoodCourt',
           'ShoppingMall', 'Spa', 'VRDeck']

train_dummied = pd.get_dummies(train[features], columns=['HomePlanet', 'CryoSleep', 'VIP'])
test_dummied = pd.get_dummies(test[features], columns=['HomePlanet', 'CryoSleep', 'VIP'])

my_imputer = SimpleImputer()
train_imputed = pd.DataFrame(my_imputer.fit_transform(train_dummied))
train_imputed.columns = train_dummied.columns
test_imputed = pd.DataFrame(my_imputer.fit_transform(test_dummied))
test_imputed.columns = test_dummied.columns

y = train.Transported.astype(int)

train_X, val_X, train_y, val_y = train_test_split(train_imputed, y, random_state=1)


# In[ ]:


rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



# In[ ]:


test_preds = rf_model.predict(test_imputed)
transported = []
for i in range(len(test_preds)):
    if test_preds[i] > 0.5:
        transported.append('True')
    else:
        transported.append('False')


# In[ ]:


output = pd.DataFrame({'PassengerId': test.PassengerId,
                       'Transported': transported})
output.to_csv('submission.csv', index=False)

