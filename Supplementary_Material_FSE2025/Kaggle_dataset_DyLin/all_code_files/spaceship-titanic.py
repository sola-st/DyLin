#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # **LOAD DATASETS**

# In[ ]:


train=pd.read_csv("/kaggle/input/spaceship-titanic/train.csv")
train.head(5)


# In[ ]:


test=pd.read_csv("/kaggle/input/spaceship-titanic/test.csv")
test.head(5)


# In[ ]:


sample_submission=pd.read_csv("/kaggle/input/spaceship-titanic/sample_submission.csv")
sample_submission.head(5)


# # **TRAIN DATASET**
# 
# 
# 

# In[ ]:


train.columns


# In[ ]:


train.info()


# In[ ]:


for col in train.columns:
    if train[col].dtype == "object":
        train[col] = train[col].astype(str)


# In[ ]:


train.info()


# In[ ]:


train.isna().sum()


# In[ ]:


train.ffill(inplace=True)


# In[ ]:


train['Age'] = train['Age'].fillna(train['Age'].mean())


# In[ ]:


train.duplicated().sum()


# In[ ]:


ax = sns.countplot(x="HomePlanet", data=train, palette="Set1")  # Change color palette
ax.set_xlabel("Home Planet")
ax.set_ylabel("Count")
ax.set_title("Distribution of Passengers by Home Planet")

# Display the plot
plt.show()


# In[ ]:


ax = sns.lineplot(x="HomePlanet", y="FoodCourt", data=train, palette="Set1")
ax.set_xlabel("Home Planet")
ax.set_ylabel("FoodCourt Spending")
ax.set_title("FoodCourt Spending by Home Planet")
plt.show()


# # **ML**

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
features = ['PassengerId', 'HomePlanet', 'CryoSleep', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall']
X = train[features]
y = train['Transported']
label_encoders = {}
for column in X.columns:
    if X[column].dtype == 'object':
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)


# In[ ]:


from sklearn.metrics import accuracy_score, classification_report

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

