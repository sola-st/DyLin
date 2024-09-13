#!/usr/bin/env python
# coding: utf-8

# # Binary Prediction of Poisonous Mushrooms using Stacking Classifier 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Input data files are available in the read-only "/kaggle/input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # 0. Loading the dataset

# In[ ]:


# Load dataset
train = pd.read_csv('/kaggle/input/playground-series-s4e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e8/test.csv')


# # 1. Inspecting the dataset

# In[ ]:


# Display first few rows of the training data
train.head(3)


# In[ ]:


test.head(3)


# In[ ]:


train.shape, test.shape # ((3116945, 22), (2077964, 21))


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train.describe()


# In[ ]:


test.describe()


# # 2. Pre-processing

# ## 2.0 Remove unecessary columns

# In[ ]:


# ID = test['id'].copy()

# train = train.drop(columns = 'id')
# test = test.drop(columns = 'id')


# ## 2.1 Handling missing values

# In[ ]:


missing_values = train.isna().sum()
percentage = train.isnull().mean() * 100
missing = pd.DataFrame({'Missing Values(train)':missing_values, 'Percentage':percentage})
missing


# In[ ]:


missing_values_test = test.isna().sum()
percentage_test = test.isnull().mean() * 100
missing_test = pd.DataFrame({'Missing Values(test)':missing_values_test, 'Percentage':percentage_test})
missing_test


# In[ ]:


train_columns_with_missing_data = percentage[percentage > 50].index
train_columns_with_missing_data


# In[ ]:


test_columns_with_missing_data = percentage_test[percentage_test > 50].index
test_columns_with_missing_data


# In[ ]:


train = train.drop(columns = train_columns_with_missing_data)
test = test.drop(columns = test_columns_with_missing_data)


# In[ ]:


train.isna().sum()
test.isna().sum()


# In[ ]:


# replace NA by the mean for continuous variables
train_conti = train.select_dtypes(exclude = 'object').copy()
test_conti = test.select_dtypes(exclude = 'object').copy()

train_conti = train_conti.fillna(train_conti.mean())
test_conti = test_conti.fillna(test_conti.mean())


# In[ ]:


# replace NA by the mode for discrete variables
train_category = train.select_dtypes('object').copy()
test_category = test.select_dtypes('object').copy()

for col in train_category.columns:
    if train_category[col].isnull().any():  
        mode = train_category[col].mode()[0]  
        train_category[col] = train_category[col].fillna(mode)
        test_category[col] = test_category[col].fillna(mode)


# In[ ]:


train = pd.concat([train_conti, train_category], axis = 1)
test = pd.concat([test_conti, test_category], axis = 1)


# In[ ]:


train.isna().sum()
test.isna().sum()


# In[ ]:


train.shape, test.shape


# ## 2.2 Pre-processing categorical columns

# In[ ]:


train.select_dtypes('object').nunique()
test.select_dtypes('object').nunique()


# In[ ]:


def replace_low_frequency_with_mode(df):
    object_columns = df.select_dtypes(include='object').columns
    
    for column in object_columns:
        value_counts = df[column].value_counts()
        
        #Filters out the values that occur 100 times or fewer in the column.
        low_freq_values = value_counts[value_counts <= 100].index
        
        mode_value = df[column].mode()[0]
        
        #Replace Low-Frequency Values with the Mode
        df[column] = df[column].apply(lambda x: mode_value if x in low_freq_values else x)

    return df

train = replace_low_frequency_with_mode(train)
test = replace_low_frequency_with_mode(test)


# In[ ]:


train.select_dtypes('object').nunique()
test.select_dtypes('object').nunique()


# ## 2.3 Pre-processing numerical columns

# In[ ]:


def detect_outliers_iqr(df):
    outliers = pd.DataFrame()
    for column in df.select_dtypes(include=['float64', 'int64']):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1 
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers[column] = ((df[column] < lower_bound) | (df[column] > upper_bound))
    return outliers


# In[ ]:


outliers_iqr = detect_outliers_iqr(train)


# In[ ]:


outliers_iqr = detect_outliers_iqr(test)


# In[ ]:


# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

def plot_boxplot(df, title):
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=df.select_dtypes(include=['float64', 'int64']))
    plt.title(f'Boxplot of {title} Data', fontsize=16)
    plt.xticks(rotation=360, fontsize=12)
    plt.xlabel("Features", fontsize=14)
    plt.ylabel("Values", fontsize=14)
    plt.show()
    
train_copy = train.drop(columns = 'id')

plot_boxplot(train_copy, 'Train')


# In[ ]:


def filter_numeric_columns(df):
     return df.select_dtypes(include=[np.number])

def remove_outliers_iqr(df):
    df_numeric = filter_numeric_columns(df)
    Q1 = df_numeric.quantile(0.25)
    Q3 = df_numeric.quantile(0.75)
    IQR = Q3 - Q1
    is_outlier = ((df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR))).any(axis=1)
    df_cleaned = df[~is_outlier]
    return df_cleaned


# In[ ]:


train_cleaned = remove_outliers_iqr(train)
test_cleaned = remove_outliers_iqr(test)


# In[ ]:


train_cleaned_copy = train_cleaned.drop(columns = 'id')

plot_boxplot(train_cleaned_copy, 'Train')


# In[ ]:


test_cleaned_copy = test_cleaned.drop(columns = 'id')

plot_boxplot(test_cleaned_copy, 'Train')


# ## 2.4 Data Splitting

# In[ ]:


x_train = train.drop(columns = 'class').copy()
y_train = train['class'].copy()
x_test = test.copy()


# In[ ]:


from sklearn.model_selection import train_test_split

#X_TRAIN, X_VAL, Y_TRAIN, Y_VAL = train_test_split(x_train, y_train, random_state = 42, test_size = 0.2, stratify = y_train, shuffle=True)

X_TRAIN, temp_x, Y_TRAIN, temp_y = train_test_split(x_train, y_train, test_size=0.95, random_state=16, stratify = y_train, shuffle=True)
drop_x, X_VAL, drop_y, Y_VAL = train_test_split(temp_x, temp_y, test_size=0.015, random_state=16, stratify = temp_y, shuffle=True)


# In[ ]:


X_TRAIN.shape, X_VAL.shape, Y_TRAIN.shape, Y_VAL.shape, x_test.shape


# ## 2.5 Encoding

# In[ ]:


#Transforms the categorical columns in the sets into a one-hot encoded forma

from sklearn.preprocessing import OneHotEncoder

X_TRAIN_category = X_TRAIN.select_dtypes('object')
X_VAL_category = X_VAL.select_dtypes('object')
X_TEST_category = x_test.select_dtypes('object')

enc = OneHotEncoder(handle_unknown = 'ignore', sparse_output = False).fit(X_TRAIN_category)

X_TRAIN_OH = enc.transform(X_TRAIN_category)
X_VAL_OH = enc.transform(X_VAL_category)
X_TEST_OH = enc.transform(X_TEST_category)


# ## 2.6 Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler

X_TRAIN_conti = X_TRAIN.select_dtypes('float64')
X_VAL_conti = X_VAL.select_dtypes('float64')
X_TEST_conti = x_test.select_dtypes('float64')

scale = StandardScaler().fit(X_TRAIN_conti)

X_TRAIN_STD = scale.transform(X_TRAIN_conti)
X_VAL_STD = scale.transform(X_VAL_conti)
X_TEST_STD = scale.transform(X_TEST_conti)


# ## 2.7 Preparing the input dataset

# In[ ]:


X_TRAIN = np.concatenate([X_TRAIN_OH, X_TRAIN_STD], axis = 1)
X_VAL = np.concatenate([X_VAL_OH, X_VAL_STD], axis = 1)
X_TEST = np.concatenate([X_TEST_OH, X_TEST_STD], axis = 1)

Y_TRAIN = Y_TRAIN.values.ravel()
Y_VAL = Y_VAL.values.ravel()


# In[ ]:


X_TRAIN.shape, X_VAL.shape, X_TEST.shape, Y_TRAIN.shape, Y_VAL.shape


# # 3. Models training

# ## 3.0 Setting up the models

# In[ ]:


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


# Initialize classifiers
rf = RandomForestClassifier(n_estimators=100, random_state=42,  n_jobs = -1)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)

# Train classifiers
rf.fit(X_TRAIN, Y_TRAIN)
gb.fit(X_TRAIN, Y_TRAIN)
knn.fit(X_TRAIN, Y_TRAIN)

# Predict on the validation set
y_pred_rf = rf.predict(X_VAL)
y_pred_gb = gb.predict(X_VAL)
y_pred_knn = knn.predict(X_VAL)


# In[ ]:


from sklearn.metrics import accuracy_score



# ## 3.1 Combine the models using Stacking

# In[ ]:


from sklearn.ensemble import StackingClassifier

# Define base learners
base_learners = [
    ('rf', rf),
    ('gb', gb),
    ('knn', knn)
]

# Define meta-learner
meta_learner = LogisticRegression()

# Initialize Stacking Classifier
stacking_clf = StackingClassifier(estimators=base_learners, final_estimator=meta_learner)

# Train Stacking Classifier
stacking_clf.fit(X_TRAIN, Y_TRAIN)

# Predict on validation set
y_pred_stacking = stacking_clf.predict(X_VAL)



# In[ ]:


from sklearn.metrics import matthews_corrcoef

mcc = matthews_corrcoef(Y_VAL, y_pred_stacking)



# # 4. Result

# In[ ]:


# Predict on test data
y_pred_test = stacking_clf.predict(X_TEST)


# In[ ]:




# In[ ]:


# Create a DataFrame for submission
submission = pd.DataFrame({'id': test['id'], 'class': y_pred_test})


# In[ ]:


submission.head()


# In[ ]:


# Save submission file
submission.to_csv('/kaggle/working/submission.csv', index=False)


