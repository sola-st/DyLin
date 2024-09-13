#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:5f780501-f3df-4236-89a5-d964539c3d92.png)
# # GridSearchCV + Pipeline 
# # Logistic Regression + XGBoost Classifier + Support Vector Classifier + Naive Bayes Classifier
# 
#  *One of the best ways to find the best classifier and to optimize our classifier by iterating through different parameters to find the best.*

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data=pd.read_csv("/kaggle/input/titanic/train.csv")
test=pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


data.head()


# In[ ]:


data.info()


# **Age, Cabin and Embarked has NaN values**

# **Using Salutaiton prefix to make out the age of the passanger to later age range wise encode it**

# In[ ]:


data['Initial']=0
for i in data:
    data['Initial']=data.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations


# In[ ]:


test['Initial']=0
for i in test:
    test['Initial']=test.Name.str.extract('([A-Za-z]+)\.')


# In[ ]:


pd.crosstab(data.Initial,data.Sex).T.style.background_gradient(cmap='summer_r') #Checking the Initials with the Sex


# **Fixing some typos and misspless in the datset for salutation**

# In[ ]:


data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)


# In[ ]:


test['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)


# In[ ]:


data.groupby('Initial')['Age'].mean()


# In[ ]:


# Assigning the NaN Values with the Ceil values of the mean ages
data.loc[(data.Age.isnull())&(data.Initial=='Master'),'Age']=5
data.loc[(data.Age.isnull())&(data.Initial=='Miss'),'Age']=22
data.loc[(data.Age.isnull())&(data.Initial=='Mr'),'Age']=33
data.loc[(data.Age.isnull())&(data.Initial=='Mrs'),'Age']=36
data.loc[(data.Age.isnull())&(data.Initial=='Other'),'Age']=46


# In[ ]:


test.loc[(test.Age.isnull())&(test.Initial=='Master'),'Age']=5
test.loc[(test.Age.isnull())&(test.Initial=='Miss'),'Age']=22
test.loc[(test.Age.isnull())&(test.Initial=='Mr'),'Age']=33
test.loc[(test.Age.isnull())&(test.Initial=='Mrs'),'Age']=36
test.loc[(test.Age.isnull())&(test.Initial=='Other'),'Age']=46


# **Replacing embarked NaN values with S since there are only 3 NaN values**

# In[ ]:


data['Embarked'].fillna('S',inplace=True)


# In[ ]:


test['Embarked'].fillna('S',inplace=True)


# In[ ]:


data.info()


# In[ ]:


data['Age_Group'] = 0  # Initialize a new column for Age groups

data.loc[data['Age'] <= 16, 'Age_Group'] = 0
data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age_Group'] = 1
data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age_Group'] = 2
data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age_Group'] = 3
data.loc[data['Age'] > 64, 'Age_Group'] = 4

data.head()


# In[ ]:


test['Age_Group'] = 0  # Initialize a new column for Age groups

test.loc[test['Age'] <= 16, 'Age_Group'] = 0
test.loc[(test['Age'] > 16) & (test['Age'] <= 32), 'Age_Group'] = 1
test.loc[(test['Age'] > 32) & (test['Age'] <= 48), 'Age_Group'] = 2
test.loc[(test['Age'] > 48) & (test['Age'] <= 64), 'Age_Group'] = 3
test.loc[test['Age'] > 64, 'Age_Group'] = 4


# In[ ]:


sns.countplot(x='Age_Group', data=data).set(title='Count Plot for Age')


# **Encoding Fare categories range wise**

# In[ ]:


data['Fare_cats']=0
data.loc[data['Fare']<=7.91,'Fare_cats']=0
data.loc[(data['Fare']>7.91)&(data['Fare']<=14.454),'Fare_cats']=1
data.loc[(data['Fare']>14.454)&(data['Fare']<=31),'Fare_cats']=2
data.loc[(data['Fare']>31)&(data['Fare']<=513),'Fare_cats']=3


# In[ ]:


test['Fare_cats']=0
test.loc[test['Fare']<=7.91,'Fare_cats']=0
test.loc[(test['Fare']>7.91)&(test['Fare']<=14.454),'Fare_cats']=1
test.loc[(test['Fare']>14.454)&(test['Fare']<=31),'Fare_cats']=2
test.loc[(test['Fare']>31)&(test['Fare']<=513),'Fare_cats']=3


# In[ ]:


sns.countplot(x='Fare_cats', hue='Survived', data=data).set(title='Count Plot of Fare Categories vs Survived')


# In[ ]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

data['Sex'] = label_encoder.fit_transform(data['Sex'])
data['Embarked'] = label_encoder.fit_transform(data['Embarked'])
data['Initial'] = label_encoder.fit_transform(data['Initial'])

data.head()


# In[ ]:


test['Sex'] = label_encoder.fit_transform(test['Sex'])
test['Embarked'] = label_encoder.fit_transform(test['Embarked'])
test['Initial'] = label_encoder.fit_transform(test['Initial'])


# In[ ]:


data.drop(['Name','Age','Ticket','Fare','Cabin','PassengerId'],axis=1,inplace=True)


# In[ ]:


test.drop(['Name','Age','Ticket','Fare','Cabin','PassengerId'],axis=1,inplace=True)


# In[ ]:


data.head()


# In[ ]:


test.head


# In[ ]:


test_dataset = test
test_dataset


# In[ ]:


train_Y = data['Survived'].ravel()
train_X = data.drop('Survived', axis=1)

test_X = test_dataset


# 

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import  GridSearchCV
from sklearn.ensemble import  VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

pipeline = Pipeline(steps=[
    ('classifier', VotingClassifier(estimators=[
        ('logreg', LogisticRegression(max_iter=1000)),
        ('gnb', GaussianNB()),
        ('xgb', XGBClassifier()),
        ('svc', SVC(probability=True))
    ], voting='soft'))
])


# In[ ]:


param_grid = {
    # Logistic Regression
    'classifier__logreg__C': [0.1, 1.0, 10],
    # Gaussian Naive Bayes
    'classifier__gnb__var_smoothing': [1e-9, 1e-8],
    # XGBoost
    'classifier__xgb__learning_rate': [0.01, 0.1],
    'classifier__xgb__n_estimators': [100, 200],
    'classifier__xgb__max_depth': [3, 5],
    # SVC
    'classifier__svc__C': [0.1, 1, 10],
    'classifier__svc__kernel': ['rbf', 'linear']
}


# In[ ]:


grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)

grid_search.fit(train_X, train_Y)


# In[ ]:


best_estimator = grid_search.best_estimator_


# In[ ]:


y_pred = best_estimator.predict(test_X)


# In[ ]:


from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(best_estimator, train_X, train_Y, cv=5, scoring='accuracy')


# In[ ]:


append =pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


submission = pd.DataFrame({
    'PassengerId': append['PassengerId'],
    'Survived': y_pred
})

submission.to_csv('submission.csv', index=False)


# In[ ]:


submission

