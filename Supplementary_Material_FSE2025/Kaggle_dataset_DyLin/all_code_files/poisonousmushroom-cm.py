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


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df_train=pd.read_csv('/kaggle/input/playground-series-s4e8/train.csv')
df_train


# In[ ]:


df_train.isna().sum()


# In[ ]:


df_train.info()


# In[ ]:


train_num_cols=df_train.select_dtypes(exclude='object')
train_num_cols


# In[ ]:


train_obj_cols=df_train.select_dtypes(include='object')
train_obj_cols


# In[ ]:


df_train['class'].value_counts()


# In this dataset, numerical features don't have null values. Only object type features have null value.

# In[ ]:


x=(df_train.iloc[:,1:].isna().sum()/len(df_train))*100


# Drop stem-root,veil-type,veil-color,spore-print-color since they have above 85% of null data.  
# Drop id column also.
# 

# In[ ]:


df_train.drop(columns=['id','stem-root','veil-type','veil-color','spore-print-color'],inplace=True)


# In[ ]:


obj_train_cols=df_train.columns
obj_train_cols


# In[ ]:


from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.decomposition import PCA


# In[ ]:


null_per=(df_train.iloc[:,1:].isna().sum()/len(df_train))*100
null_per


# In[ ]:


for i,cols in enumerate(obj_train_cols[1:]):
    df_train[cols].fillna(df_train[cols].mode()[0],inplace=True)


# In[ ]:


df_train.head(10)


# In[ ]:


le=LabelEncoder()
for i,cols in enumerate(obj_train_cols):
    df_train[cols]=le.fit_transform(df_train[cols])


# In[ ]:


df_train.head()


# In[ ]:


x=df_train.iloc[:,1:]
y=df_train.iloc[:,0]


# In[ ]:


std=StandardScaler()
X_std=std.fit_transform(x)


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(X_std,y,train_size=0.70,random_state=42)


# In[ ]:


from sklearn.pipeline import make_pipeline


# In[ ]:


DT=DecisionTreeClassifier()
DT.fit(x_train,y_train)
pred=DT.predict(x_test)
acc=accuracy_score(y_test,pred)


# In[ ]:


import xgboost
from xgboost import XGBClassifier


# In[ ]:


xgbc = XGBClassifier()
xgbc.fit(x_train,y_train)
pred=xgbc.predict(x_test)
acc=accuracy_score(y_test,pred)


# In[ ]:


xgb_model = XGBClassifier(                                      
    colsample_bytree=0.6,      
    max_depth=14,             
    min_child_weight=7,                
    random_state=42,                 
    n_estimators=200, 
    learning_rate = 0.1,
    gamma = 0,
    subsample = 0.8,
    reg_alpha = 0.005,
    reg_lambda = 1,            
    )
xgb_model.fit(x_train,y_train)
y_pred = xgb_model.predict(x_test)


# In[ ]:


acc=accuracy_score(y_test,y_pred)


# In[ ]:


df_test=pd.read_csv('/kaggle/input/playground-series-s4e8/test.csv')
df_test


# In[ ]:


df_test.isna().sum()


# In[ ]:


test_num_per=(df_test.isna().sum()/len(df_test))*100
test_num_per


# In[ ]:


df_test_id = df_test['id']
df_test_id


# In[ ]:


df_test.drop(columns=['id','stem-root','veil-type','veil-color','spore-print-color'],inplace=True)


# In[ ]:


obj_test_cols=df_test.columns
obj_test_cols


# In[ ]:


for i,cols in enumerate(obj_test_cols):
    df_test[cols].fillna(df_test[cols].mode()[0],inplace=True)


# In[ ]:


le_test=LabelEncoder()
for i,cols in enumerate(obj_test_cols):
    df_test[cols]=le_test.fit_transform(df_test[cols])


# In[ ]:


df_test


# In[ ]:


std=StandardScaler()
df_test_std=std.fit_transform(df_test)


# In[ ]:


pred_test=xgb_model.predict(df_test_std)
# y_pred = xgb_model.predict(x_test)


# In[ ]:


import numpy as np

# Assuming pred_test is a NumPy array
unique_values, counts = np.unique(pred_test, return_counts=True)

# Combine the unique values with their counts
value_counts = dict(zip(unique_values, counts))



# In[ ]:


df_submit = pd.DataFrame({
    'id': df_test_id,
    'class': pred_test })

df_submit['class']=df_submit['class'].map({0:'e',1:'p'})


# In[ ]:


df_submit.to_csv('submission.csv', index = False)


# In[ ]:


dfs=pd.read_csv('/kaggle/working/submission.csv')


# In[ ]:


dfs['class'].value_counts()

