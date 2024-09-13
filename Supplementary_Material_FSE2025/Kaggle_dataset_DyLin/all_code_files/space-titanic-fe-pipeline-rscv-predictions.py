#!/usr/bin/env python
# coding: utf-8

# # Description

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import phik

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.ensemble import VotingClassifier

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import FunctionTransformer

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer

from sklearn.model_selection import RandomizedSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, 
    AdaBoostClassifier, VotingClassifier, StackingClassifier)


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:0.3f}'.format)

RANDOM_STATE = 42
TEST_SIZE = 0.25


# In[ ]:


data_train = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')
data_test = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')
data_sample = pd.read_csv('/kaggle/input/spaceship-titanic/sample_submission.csv')


# ## data_train

# In[ ]:


data_train.head()


# In[ ]:


data_train.info()


# In[ ]:


data_train.columns = data_train.columns.str.lower()
data_test.columns = data_test.columns.str.lower()


# In[ ]:


pd.DataFrame(round(data_train.isna().mean()*100,1)).style.background_gradient('coolwarm')


# In[ ]:


pd.DataFrame(round(data_test.isna().mean()*100,1)).style.background_gradient('coolwarm')

data_train = data_train.dropna()
# ### homeplanet

# In[ ]:


data_train['homeplanet'].describe()


# In[ ]:


data_train['homeplanet'].unique()


# In[ ]:


data_train_homeplanet_pivot_table = data_train.pivot_table(index='homeplanet', values='passengerid', aggfunc='count')
data_train_homeplanet_pivot_table.reset_index()


# In[ ]:


data = data_train_homeplanet_pivot_table.reset_index()

plt.figure(figsize=(10, 5))
plt.pie(data['passengerid'], labels=data['homeplanet'], autopct='%1.1f%%', startangle=90)

plt.axis('equal') 

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# ### cryosleep

# In[ ]:


data_train['cryosleep'].describe()


# In[ ]:


data_train['cryosleep'].unique()


# In[ ]:


data_train_cryosleep_pivot_table = data_train.pivot_table(index='cryosleep', values='passengerid', aggfunc='count')
data_train_cryosleep_pivot_table.reset_index()


# In[ ]:


data = data_train_cryosleep_pivot_table.reset_index()

plt.figure(figsize=(10, 5))
plt.pie(data['passengerid'], labels=data['cryosleep'], autopct='%1.1f%%', startangle=90)

plt.axis('equal') 

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# ### cabin

# In[ ]:


data_train['cabin'].describe()


# In[ ]:


data_train['cabin'].unique()


# ### destination

# In[ ]:


data_train['destination'].describe()


# In[ ]:


data_train['destination'].unique()


# In[ ]:


data_train_destination_pivot_table = data_train.pivot_table(index='destination', values='passengerid', aggfunc='count')
data_train_destination_pivot_table.reset_index()


# In[ ]:


data = data_train_destination_pivot_table.reset_index()

plt.figure(figsize=(10, 5))
plt.pie(data['passengerid'], labels=data['destination'], autopct='%1.1f%%', startangle=90)

plt.axis('equal') 

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# ### age

# In[ ]:


data_train['age'].describe()


# In[ ]:


data_train['age'].unique()


# In[ ]:


plt.figure(figsize=(10, 5))
data_train_age_plot = sns.histplot(data=data_train, x='age', hue='transported')

plt.show()


# ### vip

# In[ ]:


data_train['vip'].describe()


# In[ ]:


data_train['vip'].unique()


# In[ ]:


data_train_vip_pivot_table = data_train.pivot_table(index='vip', values='passengerid', aggfunc='count')
data_train_vip_pivot_table.reset_index()


# In[ ]:


data = data_train_vip_pivot_table.reset_index()

plt.figure(figsize=(10, 5))
plt.pie(data['passengerid'], labels=data['vip'], autopct='%1.1f%%', startangle=90)

plt.axis('equal') 

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# ### transported

# In[ ]:


data_train['transported'].describe()


# In[ ]:


data_train['transported'].unique()


# In[ ]:


data_train_trans_pivot_table = data_train.pivot_table(index='transported', values='passengerid', aggfunc='count')
data_train_trans_pivot_table.reset_index()


# In[ ]:


data = data_train_trans_pivot_table.reset_index()

plt.figure(figsize=(10, 5))
plt.pie(data['passengerid'], labels=data['transported'], autopct='%1.1f%%', startangle=90)

plt.axis('equal') 

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# ### roomservice

# In[ ]:


data_train['roomservice'].describe()


# In[ ]:


data_test['roomservice'].describe()


# In[ ]:


plt.figure(figsize=(10, 5))
data_train_roomservice_plot = sns.histplot(data=data_train, x='roomservice', binrange=(1, 5000), hue='transported')

plt.show()


# ### foodcourt

# In[ ]:


data_train['foodcourt'].describe()


# In[ ]:


data_test['foodcourt'].describe()


# In[ ]:


plt.figure(figsize=(10, 5))
data_train_foodcourt_plot = sns.histplot(data=data_train, x='foodcourt', binrange=(1, 5000), hue='transported')

plt.show()


# ### shoppingmall

# In[ ]:


data_train['shoppingmall'].describe()


# In[ ]:


data_test['shoppingmall'].describe()


# In[ ]:


plt.figure(figsize=(10, 5))
data_train_shoppingmall_plot = sns.histplot(data=data_train, x='shoppingmall', binrange=(1, 5000), hue='transported')

plt.show()


# ### spa

# In[ ]:


data_train['spa'].describe()


# In[ ]:


data_test['spa'].describe()


# In[ ]:


plt.figure(figsize=(10, 5))
data_train_spa_plot = sns.histplot(data=data_train, x='spa', binrange=(1, 5000), hue='transported')

plt.show()


# ### vrdeck

# In[ ]:


data_train['vrdeck'].describe()


# In[ ]:


data_test['vrdeck'].describe()


# In[ ]:


plt.figure(figsize=(10, 5))
data_train_vrdeck_plot = sns.histplot(data=data_train, x='vrdeck', binrange=(1, 5000), hue='transported')

plt.show()


# ## Feature Engeneering

# In[ ]:


data_train.info()


# In[ ]:


data_train['homeplanet'] = data_train['homeplanet'].fillna(data_train['homeplanet'].mode()[0])
data_test['homeplanet'] = data_test['homeplanet'].fillna(data_test['homeplanet'].mode()[0])


# In[ ]:


data_train['cryosleep'] = data_train['cryosleep'].fillna(data_train['cryosleep'].mode()[0])
data_test['cryosleep'] = data_test['cryosleep'].fillna(data_test['cryosleep'].mode()[0])

data_train['cryosleep'] = data_train['cryosleep'].replace({True: 1, False: 0})
data_test['cryosleep'] = data_test['cryosleep'].replace({True: 1, False: 0})


# In[ ]:


data_train['destination'] = data_train['destination'].fillna(data_train['destination'].mode()[0])
data_test['destination'] = data_test['destination'].fillna(data_test['destination'].mode()[0])


# In[ ]:


data_train['age'] = data_train['age'].fillna(data_train['age'].median())
data_test['age'] = data_test['age'].fillna(data_test['age'].median())


# In[ ]:


data_train['vip'] = data_train['vip'].fillna(data_train['vip'].mode()[0])
data_test['vip'] = data_test['vip'].fillna(data_test['vip'].mode()[0])

data_train['vip'] = data_train['vip'].replace({True: 1, False: 0})
data_test['vip'] = data_test['vip'].replace({True: 1, False: 0})


# In[ ]:


data_train['roomservice'] = data_train['roomservice'].fillna(data_train['roomservice'].median())
data_test['roomservice'] = data_test['roomservice'].fillna(data_test['roomservice'].median())


# In[ ]:


data_train['foodcourt'] = data_train['foodcourt'].fillna(data_train['foodcourt'].median())
data_test['foodcourt'] = data_test['foodcourt'].fillna(data_test['foodcourt'].median())


# In[ ]:


data_train['shoppingmall'] = data_train['shoppingmall'].fillna(data_train['shoppingmall'].median())
data_test['shoppingmall'] = data_test['shoppingmall'].fillna(data_test['shoppingmall'].median())


# In[ ]:


data_train['spa'] = data_train['spa'].fillna(data_train['spa'].median())
data_test['spa'] = data_test['spa'].fillna(data_test['spa'].median())


# In[ ]:


data_train['vrdeck'] = data_train['vrdeck'].fillna(data_train['vrdeck'].median())
data_test['vrdeck'] = data_test['vrdeck'].fillna(data_test['vrdeck'].median())


# In[ ]:





# In[ ]:


data_train.info()


# ## corr

# In[ ]:


data_train.sample()


# In[ ]:


corr_matrix = data_train.drop(['passengerid', 'cabin', 'name'], axis=1).phik_matrix(interval_cols=['age', 'roomservice', 'foodcourt', 
                                                                                'shoppingmall', 'spa', 'vrdeck'])


# In[ ]:


plt.figure(figsize=(8, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Phi-correlation')
plt.show()


# ## models

# In[ ]:


data_train.head()


# In[ ]:


data_test.head()


# In[ ]:


cols_to_delete = ['passengerid', 'cabin', 'name', 'transported']


# In[ ]:


X = data_train.drop(cols_to_delete, axis=1)
y = data_train['transported']

X_test = data_test.drop(['passengerid', 'cabin', 'name'], axis=1)


# In[ ]:


ohe_columns = ['homeplanet', 'destination']

ord_columns = []

num_columns = ['cryosleep', 'age', 'vip', 'roomservice', 'foodcourt', 'shoppingmall', 'spa', 'vrdeck']


# In[ ]:


ohe_pipe = Pipeline(
    [
        ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ]
)

ord_pipe = Pipeline(
    [
        ('simpleImputer_before_ord', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
        ('ord', OrdinalEncoder(categories=[
            #[0, 1],
            #[0, 1],
        ],
        handle_unknown='use_encoded_value', unknown_value=np.nan)),
        ('simpleImputer_after_ord', SimpleImputer(missing_values=np.nan, strategy='most_frequent'))
    ]
)


# In[ ]:


data_preprocessor = ColumnTransformer(
    [
        ('ohe', ohe_pipe, ohe_columns),
        ('ord', ord_pipe, ord_columns),
        ('num', MinMaxScaler(), num_columns)  
    ],
    remainder='passthrough'
)


# In[ ]:


pipe_final = Pipeline([
    ('preprocessor', data_preprocessor),
    ('models', VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(random_state=42)),
            ('svc', SVC(random_state=42, probability=True)),
            ('rf', RandomForestClassifier(random_state=42)),
            ('gb', GradientBoostingClassifier(random_state=42)),
            ('et', ExtraTreesClassifier(random_state=42)),
        ],
        voting='soft'
    ))
    
])


# In[ ]:


param_grid = [
    {
        'models': [LogisticRegression(random_state=42, solver='liblinear', penalty='l2')],
        'models__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'models__max_iter': [100, 200, 300],
        'models__solver': ['liblinear', 'saga'],
        'preprocessor__num': [StandardScaler(), MinMaxScaler(), RobustScaler(), 'passthrough']
    },
    {
        'models': [SVC(random_state=42)],
        'models__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'models__gamma': ['scale', 'auto'] + list(np.logspace(-4, 1, num=6)),
        'models__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'preprocessor__num': [StandardScaler(), MinMaxScaler(), RobustScaler(), 'passthrough']
    },
    {
        'models': [RandomForestClassifier(random_state=42)],
        'models__n_estimators': [50, 100, 200],
        'models__max_depth': [None] + list(range(2, 30)),
        'models__min_samples_split': range(2, 30),
        'models__min_samples_leaf': range(1, 30),
        'models__max_features': ['sqrt', 'log2'],
        'preprocessor__num': [StandardScaler(), MinMaxScaler(), RobustScaler(), 'passthrough']
    },
    {
        'models': [GradientBoostingClassifier(random_state=42)],
        'models__n_estimators': [50, 100, 200],
        'models__learning_rate': [0.01, 0.1, 0.2],
        'models__max_depth': [3, 4, 5, 6],
        'models__min_samples_split': range(2, 30),
        'models__min_samples_leaf': range(1, 30),
        'models__subsample': [0.8, 1.0],
        'preprocessor__num': [StandardScaler(), MinMaxScaler(), RobustScaler(), 'passthrough']
    },
    {
        'models': [ExtraTreesClassifier(random_state=42)],
        'models__n_estimators': [50, 100, 200],
        'models__max_depth': [None] + list(range(2, 30)),
        'models__min_samples_split': range(2, 30),
        'models__min_samples_leaf': range(1, 30),
        'models__max_features': ['sqrt', 'log2'],
        'preprocessor__num': [StandardScaler(), MinMaxScaler(), RobustScaler(), 'passthrough']
    },

]


# In[ ]:


rs = RandomizedSearchCV(
    pipe_final,
    param_grid,
    cv=5,
    scoring='accuracy',
    random_state=RANDOM_STATE,
    n_jobs=-1
)


# In[ ]:


rs.fit(X, y)


# In[ ]:


rs.best_estimator_


# In[ ]:


rs.best_params_


# In[ ]:


rs.best_score_


# In[ ]:


preds = rs.predict(X_test)
preds


# In[ ]:


data_final = pd.DataFrame({'PassengerId' : data_test['passengerid'], 'Transported' : preds})
data_final.head()


# In[ ]:


data_final.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:




