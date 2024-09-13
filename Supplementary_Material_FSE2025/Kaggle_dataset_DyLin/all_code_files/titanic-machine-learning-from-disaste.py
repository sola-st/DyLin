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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder


# In[ ]:


df=pd.read_csv('/kaggle/input/titanic/train.csv')
data_test=pd.read_csv('/kaggle/input/titanic/test.csv')
df1=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
df=df.drop('Name',axis='columns')
data_test=data_test.drop('Name',axis='columns')
df=df.drop('Ticket',axis='columns')
data_test=data_test.drop('Ticket',axis='columns')
X_train=df.drop('Survived',axis='columns')
y_train=df['Survived']


# In[ ]:


li0=X_train.loc[:,(X_train.dtypes=='object')&(X_train.isnull().sum()>0)].columns.values
for i in li0:
    X_train=X_train.drop(i,axis='columns')
    data_test=data_test.drop(i,axis='columns')
li1=data_test.loc[:,(data_test.dtypes=='object')&(data_test.isnull().sum()>0)].columns.values
modelEncoder=LabelEncoder()
for i in li1:
    X_train=X_train.drop(i,axis='columns')
    data_test=data_test.drop(i,axis='columns')
for i in list(X_train.loc[:,(X_train.dtypes=='object')].columns):
    X_train[i]=modelEncoder.fit_transform(X_train[i])
    data_test[i]=modelEncoder.transform(data_test[i])
Impute=SimpleImputer(missing_values=np.nan,strategy="mean")
X_train=Impute.fit_transform(X_train)
data_test=Impute.fit_transform(data_test)
X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.1)


# In[ ]:


act=['identity', 'logistic', 'relu', 'tanh']
sol=['sgd','adam','lbfgs']
scorse={}
for wi in act:
    for k in sol :
        model=MLPClassifier(activation=wi,random_state=33,solver=k)
        model.fit(X_train,y_train)
        scorse[model.score(X_val,y_val)]=str(k)+','+wi
krange0=range(1,len(scorse)+1)
r=max(scorse.keys())
wi=scorse[r].split(',')[1]
k=scorse[r].split(',')[0]
plt.plot(krange0,scorse.keys())
plt.title(str(r)+':'+scorse[r])
plt.show()


# In[ ]:


model=MLPClassifier(activation=wi,random_state=33,solver=k)


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


model.score(X_val,y_val)


# In[ ]:


y_pr=model.predict(data_test)
df1['Survived']=y_pr
df1.to_csv('submission.csv',index=False)

