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


import warnings
warnings.filterwarnings('ignore')

# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the Titanic dataset
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')



# Data Preprocessing
# Fill missing values
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)

train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)

# Convert categorical features to numeric
train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'], drop_first=True)

# Feature selection
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
X = train_data[features]
y = train_data['Survived']
X_test = test_data[features]

# Splitting the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=41)


# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# SVM model training
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Predicting on the validation set
y_pred = svm_model.predict(X_val)

# Evaluating the model
accuracy = accuracy_score(y_val, y_pred)

# Predicting on the test set
test_predictions = svm_model.predict(X_test)

# Preparing the submission file
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': test_predictions
})
submission.to_csv('submission.csv', index=False)


# In[ ]:


train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


pd.get_dummies(train_data, columns=['Sex', 'Embarked'], drop_first=True)


# In[ ]:


# Fill missing values
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)

train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)

# Convert categorical features to numeric
train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'], drop_first=True)

# Feature selection
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
X = train_data[features]
y = train_data['Survived']
X_test = test_data[features]

X


# ![image.png](attachment:63d98620-a4ad-432c-9c78-623086e4a327.png)

# ##### 참조 : https://www.machinelearningplus.com/machine-learning/train-test-split/
# 

# ![image.png](attachment:02997718-7360-4a2f-aa49-ca6e72534142.png)

# ##### 참조: https://heeya-stupidbutstudying.tistory.com/entry/%ED%86%B5%EA%B3%84-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%A0%84%EC%B2%98%EB%A6%AC-%EC%97%AC%EB%9F%AC-%EA%B0%80%EC%A7%80-feature-scaling-%EB%B0%A9%EB%B2%95?category=1009902
# 

# ![image.png](attachment:f7117013-bb19-424b-be9d-678d3feae355.png)

# ![image.png](attachment:9e775f41-63a5-4d66-b663-da3078953300.png)

# ### 참조 : CDS 공부자료

# ![image.png](attachment:a3d68a32-62cf-420a-9142-771e5579fd53.png)

# ##### 참조 : https://david-kim2028.tistory.com/entry/%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%A0%84%EC%B2%98%EB%A6%AC-fit-fittransform-transform%EC%9D%98-%EA%B0%9C%EB%85%90-%EC%9D%B5%ED%9E%88%EA%B8%B0

# ![image.png](attachment:a112a921-6e0d-4122-8eec-f14f2c5b8f90.png)

# ##### 참조 : https://velog.io/@glad415/1.-SVMSupport-Vector-Machin

# In[ ]:




