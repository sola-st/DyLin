#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import matthews_corrcoef, classification_report
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.ensemble import VotingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ### OPENING

# In[ ]:


df = pd.read_csv('/kaggle/input/playground-series-s4e8/train.csv')
df.head()


# In[ ]:


test = pd.read_csv('/kaggle/input/playground-series-s4e8/test.csv')
display(test)


# In[ ]:


display(df)


# In[ ]:


df.info()


# ### MISSING VALUES

# We will see if there are missing values in the proposed dataset, delete the columns that will not be used, and also remove duplicates.

# In[ ]:


df.isnull().mean()*100


# In[ ]:


df.nunique().sort_values(ascending=False)


# In[ ]:


# save the test id numbers and drop all id's columns for uselessness

idd = test.id
df.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)


# Also we drop the columns with most NaN uniques

# In[ ]:


df.drop(columns=['stem-root', 'stem-surface', 'veil-type', 'veil-color', 'spore-print-color'], inplace=True)
test.drop(columns=['stem-root', 'stem-surface', 'veil-type' , 'veil-color', 'spore-print-color'], inplace=True)


# In[ ]:


df = df.drop_duplicates()


# Extract categorical and numerical variables

# In[ ]:


numerical = df.select_dtypes(exclude='object').columns
numerical


# In[ ]:


categorical = df.select_dtypes(include='object').columns
categorical


# ### VISUAL

# Let's look at the distribution of available features

# In[ ]:




# In[ ]:


palette = sns.color_palette(['#7d92bf', '#636a9f', '#494373', '#2b1f3e'])


# In[ ]:


#categorical distributions

plt.figure(figsize=(24, 36))
for i, col in enumerate(['cap-shape', 'cap-surface', 'cap-color',
       'does-bruise-or-bleed', 'gill-attachment', 'gill-spacing', 'gill-color',
       'stem-color', 'has-ring', 'ring-type', 'habitat', 'season'], 1):
    plt.subplot(6, 2, i)
    sns.countplot(data=df, x=col, palette = palette)
    plt.title(f"{col} Distribution")
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[ ]:


#numerical distributions

plt.figure(figsize=(18, 6))
for i, col in enumerate(numerical):
    plt.subplot(1, 3, i+1)
    sns.boxplot(x=col, hue='class', data = df, palette = palette)
    plt.title(f"{col} boxplot", size=16)
plt.tight_layout()
plt.suptitle("Boxplots for numerical", y=1.05, size=30)
plt.show()


# In[ ]:


#target distribution

class_cnt = df['class'].value_counts().sort_index()
labels = ['Edible', 'Poisonous']

plt.figure(figsize=(4,5))
plt.pie(class_cnt, labels = labels, autopct = '%1.1f%%', colors = palette)
plt.title('Target Distribution')
plt.axis('equal')

plt.show()
        


# ### COMPLETING/PREPROCESSING

# Next, we fill in the missing values in the columns, both numeric and categorical. Then, in the categorical columns, we replace the little-represented values, labeling them as noise.
# 
# Then we use the Label Encoder to translate the categorical values in the target column, after which we use the Ordinal Encoder to translate all other categorical features.
# 

# In[ ]:


#fill the numerical columns

for num in numerical:
    df[num] = df[num].fillna(df[num].median())
    test[num] = test[num].fillna(test[num].median())


# In[ ]:


#fill the categorical columns

for c in categorical:
    
    if c != 'class':

        df[c] = df[c].fillna('Miss')
        test[c] = test[c].fillna('Miss')


# In[ ]:


#check if nulls still here

np.mean(df.isnull().mean()*100)


# In[ ]:


#find noises and mark them

min_count = 100
def threshold(col):
    
    for i in categorical:
        
        vcounts = df[i].value_counts(dropna=False)
        rubbish = vcounts[vcounts < min_count].index

        df[i] = df[i].apply(lambda x: 'noise' if x in rubbish else x)

    return df


# In[ ]:


df = threshold(categorical)


# In[ ]:


#choose one random column to check

df['does-bruise-or-bleed'].value_counts(dropna=False)


# In[ ]:


# remove outliers 

def outliers(df, col):
    q1 = df[col].quantile(0.03)
    q3 = df[col].quantile(0.97)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])


# In[ ]:


for col in numerical:
    outliers(df, col)
    outliers(test, col)


# After getting rid of outliers, we transform categorical data into numerical data for the correct operation of the algorithms

# In[ ]:


# tranform 'class' label

le = LabelEncoder()
df['class'] = le.fit_transform(df['class'])


# In[ ]:


# transforn other non-numerical labels

oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df[categorical[1:]] = oe.fit_transform(df[categorical[1:]])
test[categorical[1:]] = oe.transform(test[categorical[1:]])


# In[ ]:


df[categorical[1:]]


# ### TRAIN/TEST 

# We select the target values and all the columns that will participate in the implementation of the algorithm. Next we split the existing data set into training and validation parts.

# In[ ]:


X_col = ['cap-diameter', 'cap-shape', 'cap-surface', 'cap-color',
       'does-bruise-or-bleed', 'gill-attachment', 'gill-spacing', 'gill-color',
       'stem-height', 'stem-width', 'stem-color', 'has-ring',
       'ring-type', 'habitat', 'season']
y_col = 'class'


# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(df[X_col], df[y_col], test_size=0.1, random_state=42, stratify=df[y_col])


# ### MODELLING

# Choose three modelling algorithms (***Catboost, XGBoost, LightGBM***) that are suitable for evaluating classification results. Also use the ensemble model like ***Voting Classifier***.

# ##### CATBOOST

# In[ ]:


cat_params = {
    "n_estimators": 300,
    "learning_rate": 0.06,
    'random_strength': 1,
    'max_bin': 255, 
    'depth': 10,
    'l2_leaf_reg': 7,
    'boosting_type': 'Plain',
    'bootstrap_type': 'Bernoulli',
    'verbose': False
}


# In[ ]:


cat_model = CatBoostClassifier(**cat_params)


# In[ ]:


cat_model.fit(X_train, y_train)


# In[ ]:


cat_pred = cat_model.predict(X_valid)


# In[ ]:




# In[ ]:


mcc_score_cat = matthews_corrcoef(y_valid, cat_pred)


# ##### XGBOOST

# In[ ]:


xgb_params = {
    "colsample_bylevel": 0.927,
    "colsample_bynode": 0.958,
    "colsample_bytree": 0.52,  
    "enable_categorical": True,
    "gamma": 0.4,
    "grow_policy": "lossguide",
    "learning_rate": 0.041,
    "max_depth": 17,
    "min_child_weight": 0.65,
    "n_estimators": 700,
    "n_jobs": -1,
    "random_state": 6,
    "reg_alpha": 1.726,
    "reg_lambda": 94.6,
    "subsample": 0.759,
    "tree_method": "hist",
    "verbosity": 0
}


# In[ ]:


xgb_model = XGBClassifier(**xgb_params)


# In[ ]:


xgb_model.fit(X_train,y_train)


# In[ ]:


xgb_pred = xgb_model.predict(X_valid)


# In[ ]:




# In[ ]:


mcc_score_xgb = matthews_corrcoef(y_valid, xgb_pred)


# ##### LIGHTGBM

# In[ ]:


best_params = {
 'n_estimators': 650,
 'learning_rate': 0.046,
 'num_leaves': 88,
 'max_depth': 12,
 'min_data_in_leaf': 28,
 'feature_fraction': 0.69,
 'bagging_fraction': 0.82,
 'bagging_freq': 4,
 'lambda_l1': 0.3,
 'lambda_l2': 0.32,
 'min_gain_to_split': 0.044
}


# In[ ]:


lgbm_model = LGBMClassifier(**best_params)


# In[ ]:


lgbm_model.fit(X_train,y_train)


# In[ ]:


lgbm_pred = lgbm_model.predict(X_valid)


# In[ ]:




# In[ ]:


mcc_score_lgbm = matthews_corrcoef(y_valid, lgbm_pred)


# ##### VOTING

# In[ ]:


voting_model = VotingClassifier(
    estimators = [('xgb', xgb_model), ('lgbm', lgbm_model)], voting='soft', verbose = True)


# In[ ]:


voting_model.fit(X_train, y_train)


# In[ ]:


voting_pred = voting_model.predict(X_valid)


# In[ ]:




# In[ ]:


mcc_score_voting = matthews_corrcoef(y_valid, voting_pred)


# In[ ]:





# ##### GROUPBY MODELS

# In[ ]:


#collect all the coefficients together for a visual presentation of the work of the models

models = pd.DataFrame({'models': ['XGB', 'LGBM', 'CAT', 'VOT'], 'mcc_scores': [mcc_score_xgb, mcc_score_lgbm, mcc_score_cat, mcc_score_voting]})


# In[ ]:


models


# The best coefficient is obtained as a result of using XGBoost

# ### TEST

# In the final block, we look at the imported dataframe with test data. Using the best of all models (or their compilation), we get predictions based on test data, which we import into a csv file.

# In[ ]:


display(test)


# In[ ]:


# choose best model

test_labels = xgb_model.predict(test[X_col])
test_labels


# In[ ]:


test_labels = le.inverse_transform(test_labels)
test_labels


# In[ ]:


submission = pd.DataFrame({'id':idd, 'class':test_labels})


# In[ ]:


submission


# In[ ]:


submission.to_csv('sub_xgb_22ver.csv', index=False)


# In[ ]:


# test_labels2 = lgbm_model.predict(test[X_col])
# test_labels2 = le.inverse_transform(test_labels2)
# submission2 = pd.DataFrame({'id':idd, 'class':test_labels2})
# submission2.to_csv('sub_lgbm_19ver.csv', index=False)


# In[ ]:


# test_labels3 = cat_model.predict(test[X_col])
# test_labels3 = le.inverse_transform(test_labels3)
# submission3 = pd.DataFrame({'id':idd, 'class':test_labels3})
# submission3.to_csv('sub_cat_15ver.csv', index=False)


# In[ ]:


# test_labels4 = voting_model.predict(test[X_col])
# test_labels4 = le.inverse_transform(test_labels4)
# submission4 = pd.DataFrame({'id':idd, 'class':test_labels4})
# submission4.to_csv('sub_vot_6ver.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:




