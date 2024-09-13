#!/usr/bin/env python
# coding: utf-8

# # Spaceship Titanic End to End Machine Learning Project

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


# In[ ]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# # Import and observe data

# In[ ]:


train_df = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')
test_df = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


train_df.info()


# In[ ]:


test_df.info()


# In[ ]:


# drop 'Name' column, because it's unnecessary
train_df.drop('Name', axis=1, inplace=True)
test_df.drop('Name', axis=1, inplace=True)


# In[ ]:


# check if 'Name' column are successfully dropped
train_df.shape, test_df.shape


# In[ ]:


# Check for null values
def plot_null(df):
  plt.figure(figsize=(10,5))
  ax = sns.barplot(x=df.isna().sum().sort_values().index, 
                   y=df.isna().sum().sort_values().values)
  ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
  ax.bar_label(ax.containers[0])
  plt.show()

plot_null(train_df)


# In[ ]:


plot_null(test_df)


# # Data preprocessing

# ## handle null on numeric variables

# In[ ]:


num_var = train_df.select_dtypes(exclude=['O', bool]).columns.values


# In[ ]:


num_var


# In[ ]:


plot_null(train_df.loc[:, num_var])


# In[ ]:


plot_null(test_df.loc[:, num_var])


# In[ ]:


num_imputer = SimpleImputer(strategy='mean')

train_df.loc[:,num_var] = num_imputer.fit_transform(train_df.loc[:,num_var])
plot_null(train_df.loc[:, num_var])


# In[ ]:


test_df.loc[:, num_var] = num_imputer.transform(test_df.loc[:, num_var])
plot_null(test_df.loc[:, num_var])


# In[ ]:


# plot null on all variables
plot_null(train_df)


# In[ ]:


plot_null(test_df)


# In[ ]:


# drop the PassengerId column because the group some passenger travelling can be represent on their cabin
train_df.drop('PassengerId', axis=1, inplace=True)
passenger_id_df = test_df[['PassengerId']]
test_df = test_df.drop('PassengerId', axis=1)


# ## handle null values on categorical variables

# In[ ]:


cat_var = train_df.select_dtypes('O').columns.values


# In[ ]:


train_df.loc[:, cat_var].head()


# In[ ]:


train_df.loc[:, cat_var].describe()


# ### Handle Destination and HomePlanet null values

# In[ ]:


# handle Destination and HomePlanet columns null values
cat_imputer = SimpleImputer(strategy='most_frequent')
train_df.loc[:, ['HomePlanet', 'Destination']] = cat_imputer.fit_transform(train_df[['HomePlanet', 'Destination']])


# In[ ]:


test_df.loc[:, ['HomePlanet', 'Destination']] = cat_imputer.transform(test_df[['HomePlanet', 'Destination']])


# In[ ]:


plot_null(train_df)


# In[ ]:


plot_null(test_df)


# ### Make 3 new columns consist of cabin deck, number, and side

# In[ ]:


cabin_split_train_df = train_df['Cabin'].str.split('/', expand=True)
cabin_split_train_df.columns = ['cabin_deck', 'cabin_deck_num', 'cabin_side']
cabin_split_train_df.head()


# In[ ]:


cabin_split_test_df = test_df['Cabin'].str.split('/', expand=True)
cabin_split_test_df.columns = ['cabin_deck', 'cabin_deck_num', 'cabin_side']
cabin_split_test_df.head()


# In[ ]:


train_df = pd.concat([train_df, cabin_split_train_df], axis=1).drop('Cabin', axis=1)
test_df = pd.concat([test_df, cabin_split_test_df], axis=1).drop('Cabin', axis=1) 


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


cat_var_new = train_df.select_dtypes('O').columns.values


# In[ ]:


train_df.loc[:, cat_var_new].describe()


# In[ ]:


test_df.loc[:, cat_var_new].describe()


# Cabin deck's number can be represent with the cabin deck and cabin side, so I'll drop that column

# In[ ]:


train_df.drop('cabin_deck_num', axis=1, inplace=True)
test_df.drop('cabin_deck_num', axis=1, inplace=True)


# In[ ]:


cat_var_new = train_df.select_dtypes('O').columns.values


# ### Handle cabin_deck and cabin_side null values

# In[ ]:


train_df.loc[:, ['cabin_deck', 'cabin_side']] = cat_imputer.fit_transform(train_df.loc[:, ['cabin_deck', 'cabin_side']])
test_df.loc[:, ['cabin_deck', 'cabin_side']] = cat_imputer.transform(test_df.loc[:, ['cabin_deck', 'cabin_side']])


# In[ ]:


plot_null(train_df.loc[:, cat_var_new])


# In[ ]:


plot_null(test_df.loc[:, cat_var_new])


# In[ ]:


train_df[['CryoSleep', 'VIP']].describe()


# ### Handle CryoSleep and VIP null values

# In[ ]:


for var in num_var:
  sns.boxplot(x='CryoSleep', y=var, data=train_df)
  plt.show()


# In[ ]:


for var in num_var:
  sns.boxplot(x='VIP', y=var, data=train_df)
  plt.show()


# In[ ]:


# make new column: total_bill

train_df['total_bill'] = train_df.loc[:, num_var[1:]].sum(axis=1).values
test_df['total_bill'] = test_df.loc[:, num_var[1:]].sum(axis=1).values


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


for var in ['CryoSleep', 'VIP']:
  sns.boxplot(x=var, y='total_bill', data=train_df)
  plt.show()


# In[ ]:


for var in ['HomePlanet', 'Destination', 'cabin_deck', 'cabin_side']:
  sns.countplot(x=var, data=train_df, hue='CryoSleep')
  plt.show()


# In[ ]:


for var in ['HomePlanet', 'Destination', 'cabin_deck', 'cabin_side']:
  sns.countplot(x=var, data=train_df, hue='VIP')
  plt.show()


# Handling VIP and CryoSleep null values will be using classification model

# #### handle VIP null values

# In[ ]:


train_vip_df = train_df.loc[:, np.append(num_var, ['total_bill', 'VIP'])].copy()
test_vip_df = test_df.loc[:, np.append(num_var, ['total_bill', 'VIP'])].copy()


# In[ ]:


train_vip_df.shape, test_vip_df.shape


# In[ ]:


train_vip_df_nonull = train_vip_df.dropna(axis=0, how='any').copy()
train_vip_df_null = train_vip_df.loc[train_vip_df['VIP'].isna(),:].drop('VIP', axis=1)

test_vip_df_null = test_vip_df.loc[test_vip_df['VIP'].isna(),:].drop('VIP', axis=1)

train_vip_df_nonull.shape, train_vip_df_null.shape, test_vip_df_null.shape


# In[ ]:


train_vip_null_index = train_vip_df.loc[train_vip_df['VIP'].isna(),:].index
test_vip_null_index = test_vip_df.loc[test_vip_df['VIP'].isna(),:].index


# In[ ]:


train_vip_df_nonull.VIP.value_counts()


# In[ ]:


encoder = LabelEncoder()
train_vip_df_nonull.loc[:, 'VIP'] = encoder.fit_transform(train_vip_df_nonull.VIP)


# In[ ]:


X_vip = train_vip_df_nonull.drop('VIP', axis=1).values
y_vip = train_vip_df_nonull[['VIP']].values.ravel()


# In[ ]:


sm = SMOTE(sampling_strategy=.75)
X_vip_res, y_vip_res = sm.fit_resample(X_vip, y_vip)


# In[ ]:


X_vip_res.shape, y_vip_res.shape


# In[ ]:


y_vip_res.sum(), (1-y_vip_res).sum()


# In[ ]:


scaler = StandardScaler()
X_vip_res = scaler.fit_transform(X_vip_res)
train_vip_df_null = scaler.transform(train_vip_df_null.values)
test_vip_df_null = scaler.transform(test_vip_df_null.values)


# In[ ]:


X_train_vip, X_val_vip, y_train_vip, y_val_vip = train_test_split(X_vip_res, y_vip_res, test_size=.2, random_state=42)

X_train_vip.shape, X_val_vip.shape, y_train_vip.shape, y_val_vip.shape


# In[ ]:


forest_vip_clf = RandomForestClassifier()

forest_vip_clf.fit(X_train_vip, y_train_vip)


# In[ ]:


from sklearn.metrics import classification_report



# In[ ]:


replace_null_vip_on_train_df = forest_vip_clf.predict(train_vip_df_null)
replace_null_vip_on_train_df = np.where(replace_null_vip_on_train_df==0, False, True)
replace_null_vip_on_train_df


# In[ ]:


replace_null_vip_on_test_df = forest_vip_clf.predict(test_vip_df_null)
replace_null_vip_on_test_df = np.where(replace_null_vip_on_test_df==0, False, True)
replace_null_vip_on_test_df


# In[ ]:


train_df.loc[train_vip_null_index, 'VIP'] = replace_null_vip_on_train_df
train_df.loc[train_vip_null_index, :].head()


# In[ ]:


plot_null(train_df.loc[:, cat_var_new])


# In[ ]:


test_df.loc[test_vip_null_index, 'VIP'] = replace_null_vip_on_test_df
test_df.loc[test_vip_null_index, :].head()


# In[ ]:


plot_null(test_df.loc[:, cat_var_new])


# #### handle CryoSleep null values

# In[ ]:


train_cryo_df = train_df.loc[:, np.append(num_var, ['total_bill', 'HomePlanet', 'cabin_deck', 'CryoSleep'])].copy()
test_cryo_df = test_df.loc[:, np.append(num_var, ['total_bill', 'HomePlanet', 'cabin_deck', 'CryoSleep'])].copy()


# In[ ]:


train_cryo_df.shape, test_cryo_df.shape


# In[ ]:


train_cryo_df_nonull = train_cryo_df.dropna(axis=0, how='any').copy()
train_cryo_df_null = train_cryo_df.loc[train_cryo_df['CryoSleep'].isna(),:].drop('CryoSleep', axis=1)

test_cryo_df_nonull = test_cryo_df.dropna(axis=0, how='any').copy()
test_cryo_df_null = test_cryo_df.loc[test_cryo_df['CryoSleep'].isna(),:].drop('CryoSleep', axis=1)

train_cryo_df_nonull.shape, train_cryo_df_null.shape, test_cryo_df_nonull.shape, test_cryo_df_null.shape


# In[ ]:


train_cryo_null_index = train_cryo_df.loc[train_cryo_df['CryoSleep'].isna(),:].index
test_cryo_null_index = test_cryo_df.loc[test_cryo_df['CryoSleep'].isna(),:].index


# In[ ]:


train_cryo_df_nonull.CryoSleep.value_counts()


# In[ ]:


test_cryo_df_nonull.CryoSleep.value_counts()


# In[ ]:


train_cryo_df_nonull = pd.concat([train_cryo_df_nonull, test_cryo_df_nonull], ignore_index=True)
train_cryo_df_nonull


# In[ ]:


classencoder = LabelEncoder()
train_cryo_df_nonull.loc[:, 'CryoSleep'] = classencoder.fit_transform(train_cryo_df_nonull.CryoSleep)


# In[ ]:


ordinalencoder = OrdinalEncoder()

train_cryo_df_nonull.loc[:, ['HomePlanet', 'cabin_deck']] = ordinalencoder.fit_transform(train_cryo_df_nonull[['HomePlanet', 'cabin_deck']])
train_cryo_df_null.loc[:, ['HomePlanet', 'cabin_deck']] = ordinalencoder.transform(train_cryo_df_null[['HomePlanet', 'cabin_deck']])
test_cryo_df_null.loc[:, ['HomePlanet', 'cabin_deck']] = ordinalencoder.transform(test_cryo_df_null[['HomePlanet', 'cabin_deck']])


# In[ ]:


X_cryo = train_cryo_df_nonull.drop('CryoSleep', axis=1).values
y_cryo = train_cryo_df_nonull.CryoSleep.values


# In[ ]:


scaler = StandardScaler()

X_cryo = scaler.fit_transform(X_cryo)
train_cryo_df_null = scaler.transform(train_cryo_df_null.values)
test_cryo_df_null = scaler.transform(test_cryo_df_null.values)


# In[ ]:


X_cryo_train, X_cryo_val, y_cryo_train, y_cryo_val = train_test_split(X_cryo, y_cryo, test_size=.2, random_state=42)

X_cryo_train.shape, X_cryo_val.shape, y_cryo_train.shape, y_cryo_val.shape


# In[ ]:


forest_cryo_clf = RandomForestClassifier()

forest_cryo_clf.fit(X_cryo_train, y_cryo_train)


# In[ ]:




# In[ ]:


replace_null_cryo_on_train_df = forest_cryo_clf.predict(train_cryo_df_null)
replace_null_cryo_on_train_df = np.where(replace_null_cryo_on_train_df==0, False, True)
replace_null_cryo_on_train_df


# In[ ]:


replace_null_cryo_on_test_df = forest_cryo_clf.predict(test_cryo_df_null)
replace_null_cryo_on_test_df = np.where(replace_null_cryo_on_test_df==0, False, True)
replace_null_cryo_on_test_df


# In[ ]:


train_df.loc[train_cryo_null_index, 'CryoSleep'] = replace_null_cryo_on_train_df
train_df.loc[train_cryo_null_index, :].head()


# In[ ]:


plot_null(train_df.loc[:, cat_var_new])


# In[ ]:


test_df.loc[test_cryo_null_index, 'CryoSleep'] = replace_null_cryo_on_test_df
test_df.loc[test_cryo_null_index, :].head()


# In[ ]:


plot_null(test_df.loc[:, cat_var_new])


# ## ALL NULL VALUES SUCCESSFULLY HANDLED

# In[ ]:


plot_null(train_df)


# In[ ]:


plot_null(test_df)


# In[ ]:


train_df.info()


# In[ ]:


test_df.info()


# ## Make preprocessing pipeline for modelling

# In[ ]:


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
  def __init__(self, cat_var=cat_var_new):
    self.cat_var = cat_var_new
    self.encoder = OneHotEncoder()
  
  def fit(self, df):
    self.encoder.fit(df.loc[:, self.cat_var])
    return self
  
  def transform(self, df):
    onehot_arr = self.encoder.transform(df.loc[:, self.cat_var]).toarray()
    onehot_df = pd.DataFrame(onehot_arr)
    df_new = df.drop(self.cat_var, axis=1).copy()
    return pd.concat([onehot_df, df_new], axis=1)


# In[ ]:


preprocess_pipeline = Pipeline([('encoder', CustomOneHotEncoder()),
                                ('scaler', StandardScaler())])


# In[ ]:


X = train_df.drop('Transported', axis=1).copy()
y = train_df.Transported.values.ravel()


# In[ ]:


encoder = LabelEncoder()

y = encoder.fit_transform(y)


# In[ ]:


X = preprocess_pipeline.fit_transform(X)
X.shape


# In[ ]:


test_set = preprocess_pipeline.transform(test_df)
test_set.shape


# ## Split train and val set

# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2, random_state=42)

X_train.shape, X_val.shape, y_train.shape, y_val.shape


# # Modelling

# In[ ]:


log = LogisticRegression()
svc = SVC()
rand_forest = RandomForestClassifier(n_jobs=-1)
boost = GradientBoostingClassifier()
knn = KNeighborsClassifier()
vote = VotingClassifier([('log', log), ('svc', svc), ('forest', rand_forest),
                         ('gboost', boost), ('knn', knn)],
                        n_jobs=-1, voting='hard')
vote.fit(X_train, y_train)


# In[ ]:




# In[ ]:


params = {'log__C':[1e-1, 1, 10, 50],
          'svc__C':[1e-1, 1, 10, 50],
          'svc__kernel':['poly','rbf','sigmoid'],
          'knn__n_neighbors':[3,5,7,10],
          'forest__n_estimators':[50,100,150]}

rand_search = RandomizedSearchCV(vote, params, n_iter=20, scoring='accuracy',
                                 cv=3, verbose=10)
rand_search.fit(X_train, y_train)


# In[ ]:


rand_search.best_params_, rand_search.cv_results_['mean_test_score'].max()


# In[ ]:


pd.DataFrame(rand_search.cv_results_).sort_values('rank_test_score').head()


# In[ ]:


model = rand_search.best_estimator_
model.fit(X_train, y_train)


# In[ ]:




# In[ ]:


y_pred = (model.predict(test_set) >= 0.5)
y_pred


# In[ ]:


passenger_id_df['Transported'] = y_pred


# In[ ]:


passenger_id_df


# In[ ]:


passenger_id_df.to_csv('submission.csv', index=False)


# In[ ]:




