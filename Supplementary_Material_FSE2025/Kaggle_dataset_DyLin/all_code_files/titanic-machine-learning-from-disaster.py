#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df_train = pd.read_csv("/kaggle/input/titanic/train.csv")
df_test  = pd.read_csv("/kaggle/input/titanic/test.csv")
df = pd.concat([df_train, df_test], ignore_index=True)
df


# In[ ]:


df.info()


# In[ ]:


sex = {
    "male"  : 0,
    "female": 1
}
df['Sex']  =  df['Sex'].map(sex)


# In[ ]:


df['Sex']


# In[ ]:


df = df.drop(['Name','Ticket','Cabin','Embarked','PassengerId'],axis=1)


# In[ ]:


df


# In[ ]:


df.info()


# In[ ]:


df.dropna(subset=['Survived','Age'], inplace=True)


# In[ ]:


df.info()


# In[ ]:


sns.heatmap(df.corr(),annot=True)


# In[ ]:


plt.hist(df.Age)
plt.xlabel("Age")
plt.ylabel("number")
plt.show()


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[ ]:


X = df.drop(['Survived'],axis=1)
y = df.Survived


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=116)


# In[ ]:


model = LogisticRegression()
model.fit(X_train,y_train)


# In[ ]:


model.score(X_train,y_train)


# In[ ]:


from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

