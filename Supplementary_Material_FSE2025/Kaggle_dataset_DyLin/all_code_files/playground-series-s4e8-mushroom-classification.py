#!/usr/bin/env python
# coding: utf-8

# # **Environment Setting**

# In[ ]:


# Install if necessary
#get_ipython().system('pip install ucimlrepo')


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from scipy.stats import chi2_contingency

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost
from sklearn.metrics import matthews_corrcoef, make_scorer

pd.set_option('display.max_columns', 500)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Notebook Parameters
import_sample_data = True # change this when necessary
sample_data_size = 0.5
random_state = 42
nulls_limit = 0.80


# In[ ]:


# Import Data
original = fetch_ucirepo(id=848).data.original
train = pd.read_csv('/kaggle/input/playground-series-s4e8/train.csv', index_col='id')
if import_sample_data:
    train = train.sample(frac=sample_data_size, random_state=random_state)
test = pd.read_csv('/kaggle/input/playground-series-s4e8/test.csv', index_col='id')



# In[ ]:


sample_sub = pd.read_csv('/kaggle/input/playground-series-s4e8/sample_submission.csv')
sample_sub.sample(5)


# # **Feature Engineering**

# In[ ]:


# Create target variable
train['edible'] = np.where(train['class'] == 'e', 1, 0)
train.sample(5, random_state=random_state)


# As the train and test data come from an original data published on UC Irvine Machine Learning Repository, first we will limit only those values of the categorical features that are also present on the original data.  
# 
# For example, consider cap-shape: 

# In[ ]:


# Train data
train['cap-shape'].value_counts(dropna=False)


# In[ ]:


# Original data
original['cap-shape'].value_counts(dropna=False)


# We see that on the original data, cap-shape is supposed to have only a few letters as categories. However, on the train data there are some values that should not be there. Because of this we will drop all those values that do not match with the original dataset, in order to cast them correctly as categories.

# In[ ]:


# Isolate categorical columns
train_numerical_features = train.select_dtypes(include=[float, int]).columns
train_cat_features = train.columns.difference(train_numerical_features)

# Cast categorical columns to category type
for c in train_cat_features:
    possible_values = set(original[c].unique())
    train.loc[~train[c].isin(possible_values), c] = np.nan
    train[c] = train[c].astype('category')
    if c != 'class':
        test.loc[~test[c].isin(possible_values), c] = np.nan
        test[c] = test[c].astype('category')


# Now we have only allowed values, an the rest are missing. Before moving one, we will evaluate the percentage of missing values, to determine if they need to be dropped later.

# In[ ]:


def percentage_missings(df):
    df_nulls = df.isnull().sum().reset_index(drop=False)
    df_nulls = df_nulls.rename(columns={'index': 'column', 0: 'n_missings'})
    df_nulls['p_nulls'] = (df_nulls['n_missings'] / df.shape[0]).round(2)
    df_nulls = df_nulls.sort_values('p_nulls', ascending=False)
    return df_nulls


# In[ ]:


train_nulls = percentage_missings(train)
train_nulls.head()


# In[ ]:


test_nulls = percentage_missings(test)
test_nulls.head()


# By following this method, we end up with 4 features that have a high percentage of missing values, with rates exceeding 85%. This is the threshold we will follow to determine the columns to drop.

# In[ ]:


train = train.drop(columns=train_nulls.loc[train_nulls['p_nulls'] > nulls_limit, 'column'])
test = test.drop(columns=test_nulls.loc[test_nulls['p_nulls'] > nulls_limit, 'column'])


# The other columns will follow an imputation. In this case, we will use the mode.

# In[ ]:


def impute_missings(df, num_method='mean', cat_method='most_frequent'):
    numerical_features = df.select_dtypes(include=[float, int]).columns
    categorical_features = df.select_dtypes(include=['category', 'object']).columns
    
    num_imputer = SimpleImputer(strategy=num_method)
    df_num_imputed = pd.DataFrame(num_imputer.fit_transform(df[numerical_features]), columns=numerical_features)

    cat_imputer = SimpleImputer(strategy=cat_method)
    df_cat_imputed = pd.DataFrame(cat_imputer.fit_transform(df[categorical_features]), columns=categorical_features)

    df_imputed = pd.concat([df_num_imputed, df_cat_imputed], axis=1)

    for c in df_imputed.columns:
        if c in numerical_features:
            df_imputed[c] = df_imputed[c].astype(int)
        else:
            df_imputed[c] = df_imputed[c].astype('category')

    return df_imputed


# In[ ]:


train_imputed = impute_missings(train)
test_imputed = impute_missings(test)

train_imputed.dtypes


# # **Features Distribution**

# In[ ]:


def thousands_formatter(x, pos):
    if x >= 1e6:
        return f'{x*1e-6:.1f}M'
    elif x >= 1e3:
        return f'{x*1e-3:.0f}K'
    else:
        return f'{x:.0f}'

def plot_feature_distribution(df: pd.DataFrame):
    fig, axs = plt.subplots(4, 5, figsize=(20, 16))
    plt.tight_layout()
    df = df.drop(columns='class')

    for c, ax in zip(df.columns, axs.flatten()):
        if pd.api.types.is_numeric_dtype(df[c]):
            sns.histplot(df[c], ax=ax, color='lightgreen')
        else:
            df[c].value_counts(sort=False).plot(kind='bar', ax=ax, color='skyblue')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(f"{c}", fontsize=12)
        ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    
    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.show()

plot_feature_distribution(train_imputed)


# These distributions are interesting to know. The ones that stand out are the histograms of the features with float values. In fact the three of them are right-skewed, so most of the values are on the lower side of the distribution.

# # **Features Relationships**

# Additionaly, we will evaluate the relationships between the features and the target variable. For this, we will use chi-square tests for the categorical variables and pearson correlation for numerical variables.

# In[ ]:


# Chi-squared results for categorical features
chi2_results = {}
cat_features = train_imputed.select_dtypes(include='category').columns
for cat in cat_features:
    contingency_table = pd.crosstab(train_imputed['edible'], train_imputed[cat], margins=True)
    p = chi2_contingency(contingency_table).pvalue
    chi2_results[cat] = p

chi2_results_table = pd.Series(chi2_results).reset_index(drop=False)
chi2_results_table = chi2_results_table.rename(columns={'index': 'column', 0: 'p_value'})
chi2_results_table = chi2_results_table.sort_values(by='p_value', ascending=False).reset_index(drop=True)
chi2_results_table['p_value'] = chi2_results_table['p_value'].round(4)
chi2_results_table


# In[ ]:


# Pearson correlation for numerical features
pearson_corr = train[train_numerical_features].corr()
pearson_corr


# The p-value for each row dictates if the categorical feature is dependent or independent from the target variable. As all of the p-values are below the threshold of 5%, then we can determine that all the features have a statistical relationship to the target variable. The same logic can be applied to numerical features, but the Pearson-correlation actually indicate the strength of the relationship.  
#   
# For example, if we were not to include a feature, we would choose to first drop the stem-height since this has a low linear relationship to the target variable.

# # **Models**

# We need to prepare input data. Because we already have a test data but without a target, we will use this to predict the final classification. Before that, we need to split the data into train and test

# In[ ]:


X_train = train_imputed.drop(columns=['edible', 'class']).copy()
y_train = train_imputed['edible']
X_test = test_imputed.copy()

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state)


# We will create a pipeline to transform columns. Numerical features need to be scaled and categorical features need to be encoded into dummies.

# In[ ]:


class CustomColumnTransformer:
    def __init__(self, df):
        self.df = df
        self.cat_features = df.select_dtypes(include='category').columns
        self.num_features = df.select_dtypes(include=[float, int]).columns
        self.pipeline = self._create_pipeline()

    def _create_pipeline(self):
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.num_features),
                ('cat', OneHotEncoder(drop='first'), self.cat_features)
            ]
        )
        pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
        return pipeline
    
    def get_transformed_data(self):
        return self.pipeline.fit_transform(self.df).toarray()

    def get_feature_names(self):
        cat_feature_names = self.pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out()
        all_feature_names = list(X_train.select_dtypes(include=[int, float]).columns) + list(cat_feature_names)
        return all_feature_names


# In[ ]:


# Input data for training models
X_train_transformed = CustomColumnTransformer(X_train).get_transformed_data()
X_val_transformed = CustomColumnTransformer(X_val).get_transformed_data()
X_test_transformed = CustomColumnTransformer(X_test).get_transformed_data()


# In[ ]:


# Logistic Regression
# logreg = LogisticRegression(random_state=random_state, max_iter=1000, n_jobs=-1)
# logreg.fit(X_train_transformed, y_train)
# y_val_pred = logreg.predict(X_val_transformed)

# mcc = matthews_corrcoef(y_val, y_val_pred)
# print(f'Matthews Correlation Coefficient (MCC) with Logistic Regression: {mcc:.4f}')

# Notes: Logistic regression seems to be performing poorly. We will discard this from version 3.


# In[ ]:


# K-Nearest Neighbors
knn = KNeighborsClassifier(n_jobs=-1)
knn.fit(X_train_transformed, y_train)
y_val_pred = knn.predict(X_val_transformed)

mcc = matthews_corrcoef(y_val, y_val_pred)


# In[ ]:


# Random Forest
rf = RandomForestClassifier(random_state=random_state, n_estimators=200, n_jobs=-1)
rf.fit(X_train_transformed, y_train)
y_val_pred = rf.predict(X_val_transformed)

mcc = matthews_corrcoef(y_val, y_val_pred)


# In[ ]:


# XGboost
xgb_params = {
    'objective': 'binary:logistic',
    'random_state': random_state,
    'eval_metric': 'logloss',
    'lambda': 3
    #'base_score': np.mean(y_train)
}

dmat_train = xgboost.DMatrix(X_train_transformed, y_train, enable_categorical=True)
dmat_val = xgboost.DMatrix(X_val_transformed, y_val, enable_categorical=True)
dmat_test = xgboost.DMatrix(X_test_transformed, enable_categorical=True)

cv_results = xgboost.cv(
    params = xgb_params,
    dtrain = dmat_train,
    num_boost_round = 500,
    nfold = 5,
    early_stopping_rounds = 5,
    verbose_eval = 10,
    seed = random_state,
    as_pandas = True,
    stratified = True
)
optimal_rounds = cv_results.shape[0]

# model training
xgb_model = xgboost.train(params=xgb_params, dtrain=dmat_train, num_boost_round=optimal_rounds)

# predict and evaluate
scores = xgb_model.predict(dmat_val)
y_val_pred = (scores >= 0.5).astype(int)
mcc = matthews_corrcoef(y_val, y_val_pred)


# # **Final Prediction**

# In[ ]:


def predict_and_export(model, test_df, y_val, model_type='xgboost'):
    if model_type == 'xgboost':
        scores = model.predict(test_df)
        y_test_pred = np.where(scores >= 0.5, 1, 'p')
    else:
        y_test_pred = model.predict(test_df)
        
    y_test_pred_class = np.where(y_test_pred == 1, 'e', 'p')
    submission = pd.DataFrame({'id': test.index, 'class': y_test_pred_class})
    print(f'{model_type} predictions shape: {submission.shape}')
    submission.to_csv('submission.csv', index=False)
    print(f'{model_type} predictions exported to csv')
    return submission


# So far de Random Forest is the best model. Still we will export the three predicted variables, and choose the top 2 highest performing results.

# In[ ]:


submission_knn = predict_and_export(knn, X_test_transformed, y_val, 'knn')
submission_rf = predict_and_export(rf, X_test_transformed, y_val, 'rf')
submission_xgb = predict_and_export(xgb_model, dmat_test, y_val, 'xgboost')

