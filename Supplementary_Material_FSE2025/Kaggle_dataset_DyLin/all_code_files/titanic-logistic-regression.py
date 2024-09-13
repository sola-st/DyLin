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


# In[ ]:


data =pd.read_csv('/kaggle/input/titanic/train.csv')


# In[ ]:


test=pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


test.head()


# In[ ]:


test_ids=test["PassengerId"]


# In[ ]:


def clean(data):
    data=data.drop(["Ticket","Cabin","Name","PassengerId"], axis=1)
    cols=["SibSp","Parch","Age","Fare"]
    
    for col in cols:
        data[col].fillna(data[col].median(), inplace= True)
    data["Embarked"].fillna("U", inplace= True)  
    
    return data
data= clean(data)
test=clean(test)   


# In[ ]:


data.head(5)


# In[ ]:


data.isnull().sum()


# In[ ]:


from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

cols=["Sex","Embarked"]

for col in cols:
    data[col]=le.fit_transform(data[col])
    test[col]=le.transform(test[col])
    print(le.classes_)
    
data.head(5)   


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
y=data['Survived']
X=data.drop(columns=["Survived"])


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[ ]:


clf=LogisticRegression(random_state=0, max_iter=1000)


# In[ ]:


clf.fit(X_train,y_train)


# In[ ]:


predictions=clf.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)


# In[ ]:


submission_preds= clf.predict(test)


# In[ ]:


df=pd.DataFrame({
    "PassengerId":test_ids.values,
    "Survived":submission_preds
})


# In[ ]:


df.to_csv("submission.csv", index=False)


# In[ ]:




