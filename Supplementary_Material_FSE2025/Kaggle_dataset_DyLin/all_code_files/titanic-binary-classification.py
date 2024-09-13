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


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


titanic=pd.read_csv('/kaggle/input/titanic/train.csv')


# ## Explore Data

# In[ ]:


titanic.head()


# In[ ]:


titanic.info()


# In[ ]:


titanic.describe()


# In[ ]:


titanic.isna().sum()


# In[ ]:


sns.heatmap(titanic.isnull())


# ## Clean Data 

# In[ ]:


titanic.Age=titanic.Age.fillna(titanic.Age.mean())
titanic.Age


# In[ ]:


titanic.isna().sum()


# In[ ]:


titanic


# In[ ]:


titanic.drop(['SibSp','Ticket','Embarked','Parch','PassengerId','Name','Cabin'],axis='columns',inplace=True)


# In[ ]:


titanic


# In[ ]:


titanic.info()


# In[ ]:


titanic.Sex.value_counts()


# In[ ]:


sns.heatmap(titanic.isnull())


# ## Transform Data

# In[ ]:


sex=pd.get_dummies(titanic.Sex)
sex


# In[ ]:


titanic=pd.concat([titanic,sex],axis=1)
titanic


# In[ ]:


titanic.drop('female',axis=1,inplace=True)


# In[ ]:


titanic


# In[ ]:


titanic.rename(columns={'male':'gender'},inplace=True)
titanic


# In[ ]:


titanic.gender=titanic.gender.replace({True:1,False:0})


# In[ ]:


titanic


# In[ ]:


titanic.drop('Sex',axis=1,inplace=True)


# In[ ]:


titanic


# In[ ]:


x=titanic.drop('Survived',axis=1)
y=titanic.Survived


# In[ ]:


x


# In[ ]:


y


# In[ ]:


titanic.Survived.hist()


# In[ ]:


titanic.Age.hist(bins=20)


# ## Creat Model

# In[ ]:


from sklearn.tree import DecisionTreeClassifier 


# In[ ]:


model=DecisionTreeClassifier()


# In[ ]:


model.fit(x,y)


# In[ ]:


model.score(x,y)


# In[ ]:


model.predict([[2,30,7.2500,0]])


# In[ ]:




