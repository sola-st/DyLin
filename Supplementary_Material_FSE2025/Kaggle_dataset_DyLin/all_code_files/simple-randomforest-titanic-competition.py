#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# In[ ]:


train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


label_encoder = LabelEncoder()
train['Sex'] = label_encoder.fit_transform(train['Sex'])
test['Sex'] = label_encoder.transform(test['Sex'])


# In[ ]:


X = train[["Pclass", "Sex", "SibSp", "Parch"]]
y = train['Survived']
X_test = test[["Pclass", "Sex", "SibSp", "Parch"]]


# In[ ]:


model = RandomForestClassifier(n_estimators=100,max_depth=5, random_state=42)
model.fit(X, y)


# In[ ]:


preds = model.predict(X_test)
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': preds})
output.to_csv('submission.csv', index=False)


# In[ ]:



