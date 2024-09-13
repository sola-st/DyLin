#!/usr/bin/env python
# coding: utf-8

# # Importing Lib 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')


# # Reading Data

# In[ ]:


train = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')
test = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# # EDA

# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


# Droping not needed Col

train.drop(['PassengerId', 'Name'], axis=1, inplace=True)
test.drop(['PassengerId', 'Name'], axis=1, inplace=True)


# In[ ]:


# Filling NULL for Numerical Values

columns=['HomePlanet','CryoSleep','VIP','Destination']
for col in columns:
    train[col].fillna(train[col].mode(), inplace=True)
    test[col].fillna(train[col].mode(), inplace=True)


# In[ ]:


train['Age'].fillna(train['Age'].median(), inplace=True)
test['Age'].fillna(train['Age'].median(), inplace=True)


# In[ ]:


a=['RoomService','Spa','FoodCourt','ShoppingMall','VRDeck']
for i in a:
    train[i].fillna(train[i].mean(), inplace=True)
    test[i].fillna(train[i].mean(), inplace=True)


# ## Encoding Col

# In[ ]:


train=pd.get_dummies(train, columns=['HomePlanet'],dtype='int64')
test=pd.get_dummies(test, columns=['HomePlanet'],dtype='int64')
train=pd.get_dummies(train, columns=['Destination'],dtype='int64')
test=pd.get_dummies(test, columns=['Destination'],dtype='int64')


# In[ ]:


b=['CryoSleep','VIP']
le=LabelEncoder()
for i in b:
    train[i]=le.fit_transform(train[i])
    test[i]=le.fit_transform(test[i])
train['Transported']=le.fit_transform(train['Transported'])


# In[ ]:


train['CryoSleep']=train['CryoSleep'].astype('int64')
test['CryoSleep']=test['CryoSleep'].astype('int64')


# In[ ]:


train[['cabin_code', 'cabin_number', 'cabin_location']] = train['Cabin'].str.split('/', expand=True)
test[['cabin_code', 'cabin_number', 'cabin_location']] = test['Cabin'].str.split('/', expand=True)


# In[ ]:


train['cabin_code'].fillna('U', inplace=True)
train['cabin_number'] = train['cabin_number'].astype(float)
test['cabin_code'].fillna('U', inplace=True)
test['cabin_number'] = test['cabin_number'].astype(float)
train['cabin_number'].fillna(train['cabin_number'].median(), inplace=True)
test['cabin_number'].fillna(test['cabin_number'].median(), inplace=True)


# In[ ]:


train.drop('Cabin', axis=1, inplace=True)
test.drop('Cabin', axis=1, inplace=True)
train.drop('cabin_location', axis=1, inplace=True)
test.drop('cabin_location', axis=1, inplace=True)


# In[ ]:


train['cabin_code']=le.fit_transform(train['cabin_code'])
test['cabin_code']=le.fit_transform(test['cabin_code'])


# In[ ]:


col = ['RoomService','Spa','FoodCourt','ShoppingMall','VRDeck']
scaler=StandardScaler()
train['Age']=scaler.fit_transform(train['Age'].values.reshape(-1,1))
test['Age']=scaler.fit_transform(test['Age'].values.reshape(-1,1))
for i in col:
    train[i]=scaler.fit_transform(train[i].values.reshape(-1,1))
    test[i]=scaler.fit_transform(test[i].values.reshape(-1,1))


# ## Find Outlier

# In[ ]:


for col in train.columns:
  sns.boxplot(train[col])
  plt.title(col)
  plt.show()


# In[ ]:


for col in test.columns:
  sns.boxplot(train[col])
  plt.title(col)
  plt.show()


# ## Visilization

# In[ ]:


fig = px.pie(train, names='Transported', title='Transported')
fig.show()


# In[ ]:


fig = px.pie(train, names='CryoSleep', title='CryoSleep')
fig.show()


# In[ ]:


fig = px.pie(train, names='VIP', title='VIP')
fig.show()


# In[ ]:


fig = px.pie(train, names='HomePlanet_Earth', title='HomePlanet_Earth')
fig.show()


# ## Prediction

# In[ ]:


X = train.drop('Transported', axis=1)
y = train['Transported']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ### RandomForestClassifier

# In[ ]:


RF = RandomForestClassifier(max_depth= 20, min_samples_leaf=  4, min_samples_split=5, n_estimators= 200)
RF.fit(X_train, y_train)
y_pred = RF.predict(X_test)


# In[ ]:


y_pred_test = RF.predict(test)
rf_test = RF.predict_proba(test)


# ### XGBOOST

# In[ ]:


XG = XGBClassifier()
XG.fit(X_train, y_train)
y_pred = XG.predict(X_test)


# In[ ]:


y_pred_test_xg = XG.predict(test)
xg_test = XG.predict_proba(test)


# ### Logistic

# In[ ]:


LO = LogisticRegression()
LO.fit(X_train, y_train)
y_pred = LO.predict(X_test)


# # Submission

# In[ ]:


y_pred_test = pd.Series(y_pred_test)
y_pred_test = y_pred_test.map({0:False, 1:True})
y_pred_test = y_pred_test.to_list()


# In[ ]:


test2 = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')
T1 = test2['PassengerId']


# In[ ]:


Submission = pd.DataFrame({'PassengerId':T1, 'Transported':y_pred_test})
Submission.to_csv('Submission.csv', index=False)

