#!/usr/bin/env python
# coding: utf-8

# # **Upvote for our friend Sigmoid**
# ![image.png](attachment:cadc89d7-c184-4375-b268-767b82e54765.png)
# **Excuse my handwriting**

# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


# In[ ]:


train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


survival_percentages = train.groupby('Sex')['Survived'].mean() * 100

plt.figure(figsize=(8, 6))
plt.bar(survival_percentages.index, survival_percentages, color=['blue', 'salmon'])
plt.xlabel('Gender')
plt.ylabel('Survival Percentage (%)')
plt.title('Survival Percentage by Gender')
plt.ylim(0, 100)
plt.show()


# In[ ]:


survival_percentages_by_class = train.groupby('Pclass')['Survived'].mean() * 100

plt.figure(figsize=(8, 6))

plt.bar(survival_percentages_by_class.index, survival_percentages_by_class, color=['blue','green','pink'])
plt.xlabel('Passenger Class')
plt.ylabel('Survival Percentage (%)')
plt.title('Survival Percentage by Passenger Class')
plt.ylim(0, 100)
plt.xticks(ticks=survival_percentages_by_class.index, labels=[f'Class {i}' for i in survival_percentages_by_class.index])

plt.show()


# In[ ]:


label_encoder = LabelEncoder()
train['Sex'] = label_encoder.fit_transform(train['Sex'])
test['Sex'] = label_encoder.transform(test['Sex'])


# In[ ]:


x_train = train[["Pclass", "Sex", "SibSp", "Parch"]]
y_train = train['Survived']


# In[ ]:


x_test = test[["Pclass", "Sex", "SibSp", "Parch"]]


# In[ ]:


x_train.head()


# In[ ]:


y_train.head()


# In[ ]:


logreg = LogisticRegression()
logreg.fit(x_train, y_train)


# In[ ]:


train_pred = logreg.predict(x_train)


# In[ ]:


y_pred = logreg.predict(x_test)


# In[ ]:


output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': y_pred})
output.to_csv('submission.csv', index=False)


# In[ ]:


output

