#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# Input data files are available in the read-only "/kaggle/input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# **Loading the data**
# 

# In[ ]:


train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()


# In[ ]:


test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()


# **Exploring the data: Finding the meaningful patterns**

# In[ ]:


#DATA TYPE CHECK
train_data.info()


# In[ ]:


#ANALYZING AGE COLUMN
train_data.Age.describe()


# In[ ]:


sns.histplot(x=train_data['Age'], bins = 20)
plt.title("Age Hist")
plt.show()


# In[ ]:


#SINCE AGE WILL BE FEATURE ON WHICH MODEL WILL BE TRAINED WE WILL FE FILLING THE NULL VALUES WITH THE MEAN VALUES

train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())


# In[ ]:


#CHECKING FOR NULL VALUES

train_data.isnull().sum()


# In[ ]:


#CHECKING FOR DUPLICATES
train_data.duplicated().sum()


# In[ ]:


women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)


# In[ ]:


men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)
#as we can see survival rate of women is higher may be due to the women and children first policy while evacuating and through this we gained a pattern that if 
# its female the rate of survival is more and sex can be used as a main feature in model training


# In[ ]:


train_data.corr(numeric_only =True)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
y = train_data['Survived']
features = ["Pclass", "Sex", "SibSp","Parch","Age","SibSp","Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state = 1)
model.fit(X,y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived':predictions})
output.to_csv('submission.csv', index=False)

