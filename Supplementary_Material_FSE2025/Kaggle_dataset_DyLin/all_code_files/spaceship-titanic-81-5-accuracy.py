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
        
import warnings
warnings.filterwarnings("ignore")       

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Overview

# - PassengerId - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.
# - HomePlanet - The planet the passenger departed from, typically their planet of permanent residence.
# - CryoSleep - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
# - Cabin - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
# - Destination - The planet the passenger will be debarking to.
# - Age - The age of the passenger.
# - VIP - Whether the passenger has paid for special VIP service during the voyage.
# - RoomService, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
# - Name - The first and last names of the passenger.
# - Transported - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.

# In[ ]:


df_train = pd.read_csv("/kaggle/input/spaceship-titanic/train.csv")
df_test = pd.read_csv("/kaggle/input/spaceship-titanic/test.csv")
# just taking a copy of test dataset
df_test_copy = df_test.copy()
submission = pd.read_csv("/kaggle/input/spaceship-titanic/sample_submission.csv")


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:


df_train.head()


# In[ ]:


# remove PassengerID
df_train.drop(columns='PassengerId',inplace=True)
df_test.drop(columns='PassengerId',inplace=True)


# In[ ]:


# remove name column
df_train.drop(columns='Name',inplace=True)
df_test.drop(columns='Name',inplace=True)


# In[ ]:


df_train.sample(5)


# ## Feature Engineering

# In[ ]:


def feature_eng(df):
    
    df['TotalService'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']
    df['TotalServiceByAge'] = df['TotalService'] / (df['Age'] + 1) 
    df[['Deck','Num','Side']] = df['Cabin'].str.split('/', expand=True)
    df['AgeGroup'] = pd.cut(df['Age'] , bins=[0,12,19,30,50,80] , labels=['children','teens','youngsters','adults','seniors'])
    
    
    return df

df_train = feature_eng(df_train)
df_test = feature_eng(df_test)
    


# In[ ]:


df_train.head()


# - Imputing Categorical Columns

# In[ ]:


def impute(df):
    categorical_cols = ['HomePlanet','CryoSleep','Cabin','Destination','VIP','AgeGroup','Deck','Side']
    for col in categorical_cols:
        if col in df.columns and df[col].dtype.name == 'category':
            if 'NoInformation' not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories('NoInformation')
            df[col].fillna('NoInformation',inplace=True)
        else:
            df[col].fillna('NoInformation',inplace=True)
            
    return df    
df_train = impute(df_train)
df_test = impute(df_test)


# - Imputing Numerical columns using iterative imputer

# In[ ]:


num_cols = ['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','TotalService','TotalServiceByAge','Num']
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
imputer = IterativeImputer()
df_train[num_cols] = imputer.fit_transform(df_train[num_cols]) 
df_test[num_cols] = imputer.transform(df_test[num_cols]) 


# In[ ]:




# - No more Missing datas 🥳

# In[ ]:


df_train.head()


# In[ ]:


# dropping cabin column
df_train.drop(columns='Cabin',inplace=True)
df_test.drop(columns='Cabin',inplace=True)


# - making sure categorical columns as category by updating it using function

# In[ ]:


def update(df):
    
    cat_cols = ['HomePlanet','CryoSleep','Destination','Deck','Side',
            'VIP','AgeGroup']
    
    for col in cat_cols:
        df[col] = df[col].astype('category')
    
    return df

df_train = update(df_train)
df_test = update(df_test)


# In[ ]:




# - encoding categorical columns

# In[ ]:


# Encoding
cols_to_encode = ['HomePlanet', 'CryoSleep','Destination','VIP','Deck','Side','AgeGroup']

df_train = pd.get_dummies(df_train , columns=cols_to_encode).astype(int)
df_test = pd.get_dummies(df_test, columns=cols_to_encode).astype(int)


# In[ ]:


df_train.head()


# - scaling

# In[ ]:


all_cols = df_train.columns
cols_to_scale = all_cols.drop('Transported')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

df_train[cols_to_scale] = scaler.fit_transform(df_train[cols_to_scale])
df_test[cols_to_scale] = scaler.transform(df_test[cols_to_scale])


# In[ ]:




# - Now our Dataset is ready for modelling 🙌 

# ## Modelling

# In[ ]:


# label and target split
X = df_train.drop(columns='Transported')
y = df_train['Transported']


# - Here we will be using 3 algos , Lets import them

# In[ ]:


from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier


# - Model training with Stratified k fold cross validation
# 

# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

def modelling(model, X, y):
    
    skf = StratifiedKFold(n_splits= 10, shuffle= True, random_state= 42)
    accuracies = []
    for fold, (train_index , test_index) in enumerate (skf.split(X,y),1):
        X_train , X_test = X.iloc[train_index] , X.iloc[test_index]
        y_train , y_test = y.iloc[train_index] , y.iloc[test_index]
        
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_pred, y_test)

        accuracies.append(score)
      
    
        print(f" fold {fold} - Accuracy : {score:.4f}")
    print('\n')    
    print(f"mean Accuracy {np.mean(accuracies):.4f}")
    
     # Return the model trained on the full dataset
    model.fit(X, y)
    return model


# ## XGB Classifier

# In[ ]:


xgb = XGBClassifier()
xgb_model = modelling(model=xgb, X= X, y=y)


# ## CatBoost Classifier

# In[ ]:


cb = CatBoostClassifier(verbose=0,random_state=42)
catboost_model = modelling(model=cb,X= X, y=y)


# ## LightGBM classifier

# In[ ]:


lgb = LGBMClassifier()
lgbm_model = modelling(model=lgb, X= X, y=y)


# #### Best Accuracy we got from Catboost Classifier

# In[ ]:


# Now you can use the trained model to make predictions on df_test
prediction = catboost_model.predict(df_test)


# In[ ]:


# Example of submission dataframe
pd.read_csv('/kaggle/input/spaceship-titanic/sample_submission.csv').head(2)


# In[ ]:


df_test_copy.head()


# In[ ]:


# check whether lenght of prediction equal to the lenght of passengerID
assert len(prediction) == len(df_test_copy['PassengerId'])


# In[ ]:



submission = pd.DataFrame()

submission['PassengerId'] = df_test_copy['PassengerId']
submission['Transported'] = prediction 
submission['Transported'] = submission['Transported'].astype(bool)
submission.to_csv('submission.csv',index=False)


# In[ ]:


pd.read_csv('/kaggle/working/submission.csv').head()


# ## THANK YOU

# # Happy Kaggling 💐
