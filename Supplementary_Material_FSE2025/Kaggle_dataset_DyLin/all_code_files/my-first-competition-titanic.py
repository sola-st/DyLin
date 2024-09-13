#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier

import warnings

# Filter all warnings
warnings.filterwarnings('ignore')


# In[ ]:


data_file_path :str = '/kaggle/input/titanic/train.csv'
test_file_path :str = '/kaggle/input/titanic/test.csv'
df_data = pd.read_csv(data_file_path)
df_test = pd.read_csv(test_file_path)


# In[ ]:


df_data.head()


# In[ ]:


df_test.head()


# Clearing the data

# In[ ]:


df_data.isna().sum()


# In[ ]:


df_test.isna().sum()


# In[ ]:


test_ids = df_test['PassengerId']


# In[ ]:


#initialize func because i don't wanna repeat same operations(we have a 2 df, train and test)
def clear_data(data, cols_to_drop):
    data = data.drop(cols_to_drop,axis=1)
    return data
    
def fill_data(data, cols_to_fill):
    for col in cols_to_fill:
        if data[col].dtype == 'object':
            data[col].fillna(data[col].mode()[0],inplace=True)
        else:
            data[col].fillna(data[col].median(),inplace=True)
    return data

def convert_categorical_columns(df):
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    return df

def normalize_columns(df, columns):
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


# In[ ]:


cols_to_delete = ['Name','Ticket','Cabin']
cols_to_fill = ['SibSp','Parch','Fare','Age','Embarked']
cols_to_normalize = ['Age','Fare']
df_data = clear_data(df_data, cols_to_delete)
df_test = clear_data(df_test, cols_to_delete)

df_data = fill_data(df_data,cols_to_fill)
df_test = fill_data(df_test,cols_to_fill)

df_data = convert_categorical_columns(df_data)
df_test = convert_categorical_columns(df_test)

df_data = normalize_columns(df_data, cols_to_normalize)
df_test = normalize_columns(df_test, cols_to_normalize)


# In[ ]:


df_data.head()


# In[ ]:


df_test.head()


# In[ ]:


y = df_data['Survived']
X = df_data.drop('Survived',axis=1)

X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=42)


# In[ ]:


clf = XGBClassifier(random_state=42, n_estimators=100, max_depth=7, learning_rate=0.1).fit(X_train, y_train)


# In[ ]:


predictions = clf.predict(X_val)
accuracy = accuracy_score(y_val, predictions)


# In[ ]:


submission_predicts = clf.predict(df_test)


# In[ ]:


df = pd.DataFrame({'PassengerId':test_ids.values, 'Survived':submission_predicts})
df.to_csv('submission.csv',index=False)


# In[ ]:




