#!/usr/bin/env python
# coding: utf-8

# # Binary Prediction of Poisonous Mushrooms
# (Playground Series - Season 4, Episode 8)
# 
# The goal of this competition is to predict whether a mushroom is edible or poisonous based on its physical characteristics.

# # Import libraries

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import matthews_corrcoef

from sklearn.preprocessing import LabelEncoder


# # Load data

# In[ ]:


train_dataframe = pd.read_csv('/kaggle/input/playground-series-s4e8/train.csv')
train_dataframe.head(3)


# In[ ]:


test_dataframe = pd.read_csv('/kaggle/input/playground-series-s4e8/test.csv')
test_dataframe.head(3)


# # Preprocessing

# In[ ]:


count_classes = train_dataframe['class'].value_counts()
count_classes / train_dataframe.shape[0]


# **The classes are quite balanced.**

# In[ ]:


train_dataframe.info()


# In[ ]:


nan_counter = train_dataframe.isnull().mean() * 100
nan_counter


# **So there are 3 numerical features and the rest are categorical. The are a lot of missing values**

# In[ ]:


# Let's check some feature values

[x for x in set(train_dataframe['cap-surface'])][:10]


# **As we can see, the recorded values contain strange values. Let's fix this in accordance with the documentation of the dataset.**

# In[ ]:


# First of all fill some values for closer

replacements = {
    '5 f': 'f', '7 x': 'x', 'is s': 's', 'is p': 'p',
    '3 x': 'x', 'b f': 'b', 'does n': 'n', 'does w': 'w',
    'is n': 'n', 'e n': 'e', 'is w': 'w', 'f has-ring': 'f',
    'does f': 'f', 'is h': 'h',
    'has d': 'd', 'has f': 'f', 'is a': 'a', 'is f': 'f',
    'p p': 'p', 'does t': 't', 'has h': 'h', 'is h': 'h',
    'is k': 'k', 'does h': 'h', 'does l': 'l', 'is p': 'p',
    'is s': 's'
}

def swap_data(df):
    object_cols = df.select_dtypes(include='object').columns
    df[object_cols] = df[object_cols].replace(replacements)
    

swap_data(train_dataframe)
swap_data(test_dataframe)


# In[ ]:


# Fill numerical NaN features as mean value by column

train_dataframe['cap-diameter'] = train_dataframe['cap-diameter'].fillna(train_dataframe['cap-diameter'].mean())
test_dataframe['cap-diameter'] = test_dataframe['cap-diameter'].fillna(test_dataframe['cap-diameter'].mean())
test_dataframe['stem-height'] = test_dataframe['stem-height'].fillna(test_dataframe['stem-height'].mean())


# **Let's encode categorical columns:**

# In[ ]:


from sklearn.preprocessing import OrdinalEncoder

categorical_features  = [
    'cap-shape',
    'cap-surface',
    'cap-color',
    'does-bruise-or-bleed',
    'gill-attachment',
    'gill-spacing',
    'gill-color',
    'stem-root',
    'stem-surface',
    'stem-color',
    'veil-type',
    'veil-color',
    'has-ring',
    'ring-type',
    'spore-print-color',
    'habitat',
    'season'
]

odEncoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

train_dataframe[categorical_features] = odEncoder.fit_transform(train_dataframe[categorical_features].astype(str))
test_dataframe[categorical_features] = odEncoder.transform(test_dataframe[categorical_features].astype(str))


# In[ ]:


# Class column to binary:

le = LabelEncoder()
le.fit(train_dataframe['class'])
train_dataframe['class'] = le.transform(train_dataframe['class'])
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
le_name_mapping


# In[ ]:


train_dataframe[categorical_features] = train_dataframe[categorical_features].astype('int').astype('category')
test_dataframe[categorical_features] = test_dataframe[categorical_features].astype('int').astype('category')


# **So check NaN values in DFs:**

# In[ ]:


train_dataframe.isnull().mean() * 100


# In[ ]:


test_dataframe.isnull().mean() * 100


# In[ ]:


X = train_dataframe.drop(columns=['id', 'class'])
Y = train_dataframe['class']


# # Training:

# In[ ]:


# Let's define scorer function:

def get_score_at_best_treshold(y_test, y_pred):
    mx_score = 0
    best_treshold = None
    
    for treshold in np.linspace(0.3, 0.7, 100):
        binary_pred = (y_pred >= treshold).astype(int)
        
        cur_score = matthews_corrcoef(y_test, binary_pred)
        
        if cur_score > mx_score:
            mx_score = cur_score
            best_treshold = treshold
            
    print(f"MCC score : {mx_score} with treshold {best_treshold}")
        


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# **First of all let's try xgboost model:**

# In[ ]:


# !pip install xgboost


# In[ ]:


from xgboost import XGBClassifier

xgboost = XGBClassifier(
    alpha=0.1,
    subsample=0.8,
    colsample_bytree=0.55,
    objective='binary:logistic',
    max_depth=14,
    min_child_weight=7,
    gamma=1e-6,
    random_state=42,
    n_estimators=100,
    device="cuda"
)


# In[ ]:


xgboost.fit(X_train, y_train)


# In[ ]:


prediction = xgboost.predict_proba(X_test)

get_score_at_best_treshold(y_test, prediction[:, 1])


# **Let's try catboost:**

# In[ ]:


# !pip install catboost


# In[ ]:


from catboost import CatBoostClassifier

catboost = CatBoostClassifier(n_estimators=200, max_depth=14, task_type='GPU', cat_features=categorical_features)


# In[ ]:


catboost.fit(X_train, y_train)


# In[ ]:


prediction = catboost.predict_proba(X_test)

get_score_at_best_treshold(y_test, prediction[:, 1])


# **And LightGBM:**

# In[ ]:


from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(
    n_estimators=500,
    random_state=42,
    max_bin=128,
    colsample_bytree=0.6,
    reg_lambda = 80,
    learning_rate= 0.1,
    device = 'gpu',
    verbose = -1
)


# In[ ]:


lgbm.fit(X_train, y_train)


# In[ ]:


prediction = lgbm.predict_proba(X_test)

get_score_at_best_treshold(y_test, prediction[:, 1])


# **Check stacking of two models:**

# In[ ]:


clf1 = XGBClassifier(
    alpha=0.1,
    subsample=0.8,
    colsample_bytree=0.55,
    objective='binary:logistic',
    max_depth=14,
    min_child_weight=7,
    gamma=1e-6,
    random_state=42,
    n_estimators=100,
    device="cuda"
)

clf2 = CatBoostClassifier(n_estimators=200, max_depth=14, task_type='GPU', cat_features=categorical_features)

estimators = [
    ('xgboost', clf1),
    ('catboost', clf2),
]
clf = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression()
)

clf.fit(X_train, y_train)


# In[ ]:


prediction = clf.predict_proba(X_test)

get_score_at_best_treshold(y_test, prediction[:, 1])


# **And add LGMB:**

# In[ ]:


clf1 = XGBClassifier(
    alpha=0.1,
    subsample=0.8,
    colsample_bytree=0.55,
    objective='binary:logistic',
    max_depth=14,
    min_child_weight=7,
    gamma=1e-6,
    random_state=42,
    n_estimators=100,
    device="cuda",
    enable_categorical=True
)

clf2 = CatBoostClassifier(n_estimators=200, max_depth=14, task_type='GPU', cat_features=categorical_features)

clf3 = LGBMClassifier(
    n_estimators=500,
    random_state=42,
    max_bin=128,
    colsample_bytree=0.6,
    reg_lambda = 80,
    learning_rate= 0.1,
    device = 'gpu',
    verbose = -1
)

estimators = [
    ('xgboost', clf1),
    ('catboost', clf2),
    ('lgbm', clf3)
]
stacking = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression()
)

stacking.fit(X_train, y_train)


# In[ ]:


prediction = stacking.predict_proba(X_test)

get_score_at_best_treshold(y_test, prediction[:, 1])


# # Saving results

# In[ ]:


prediction_to_sub = stacking.predict_proba(test_dataframe.drop(columns=['id']))


# In[ ]:


result = np.where(prediction_to_sub[:,1] >= 0.5949494949494949, 'p', 'e')


# In[ ]:


output_dataframe = pd.DataFrame({
    'id' : test_dataframe['id'],
    'class' : result
})


# In[ ]:


output_dataframe.to_csv('/kaggle/working/3boost.csv', index=False, sep=',')

