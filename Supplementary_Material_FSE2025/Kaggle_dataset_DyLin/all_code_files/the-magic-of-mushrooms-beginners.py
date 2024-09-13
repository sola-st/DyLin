#!/usr/bin/env python
# coding: utf-8

# # The Magic of Mushrooms
# 
# **Introduction**
# 
# This project is coded in the simplest way possible by following the fundamental steps of machine learning. My goal is to demonstrate how effective results can be achieved even with basic methods. Therefore, it is designed to be suitable for those who are new to machine learning. The code is straightforward and easy to follow, making it accessible to anyone at the beginner level. This work will be useful for those looking to better understand the machine learning process and see how strong results can be achieved with basic techniques.

# In[ ]:


#We are loading all the necessary libraries.

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import matthews_corrcoef


# In[ ]:


#We are loading the datasets
df = pd.read_csv('/kaggle/input/playground-series-s4e8/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s4e8/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s4e8/sample_submission.csv')


# **Exploring the dataset**
# 
# It is important to examine the first five rows of our dataset to better understand what we are working with.

# In[ ]:


df.head()


# In[ ]:


df_test.head()


# As you can see, our dataset has some missing values (NaN). To handle these missing values, let's take a look at the proportion of missing data. Instead of just looking at the number of missing values, I find it more useful to examine the ratio of missing data to the total number of data points

# In[ ]:


missing_value = df.isnull().mean() * 100


# There is a significant amount of missing data in the dataset, and I believe that removing these missing values will yield better results to ensure they do not affect the outcome

# In[ ]:


#Removing columns with a high amount of missing data

deleted_columns = ['id','stem-root','stem-surface','veil-type','veil-color','spore-print-color']

df = df.drop(columns = deleted_columns)


# In[ ]:


#We are performing the same operations for the test data as well.

df_test = df_test.drop(columns = deleted_columns)


# In[ ]:


missing_value = df.isnull().mean() * 100
missing_value


# In[ ]:


#Different operations may be required for categorical and numerical columns, so we need to separate them.
categorical_columns = df.select_dtypes(include=['object']).columns
numerical_columns = ['cap-diameter', 'stem-height', 'stem-width']

categorical_columns_test = df_test.select_dtypes(include=['object']).columns


# In[ ]:


for column in categorical_columns:
    label_encoder = LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])


for column in categorical_columns_test:
    label_encoder = LabelEncoder()
    df_test[column] = label_encoder.fit_transform(df_test[column])


# In[ ]:


missing_value = df.isnull().mean() * 100
missing_value


# There is a small proportion of missing data in the cap-diameter column, which we can fill with the mean values.

# In[ ]:


df['cap-diameter'] = df['cap-diameter'].fillna(df['cap-diameter'].mean())


# In[ ]:


missing_value = df.isnull().mean() * 100
missing_value


# In[ ]:


missing_value_test = df_test.isnull().mean() * 100
missing_value_test


# We are also filling the columns with a small proportion of missing data in the test dataset with the mean values.

# In[ ]:


target_columns_test = ['cap-diameter', 'stem-height']
df_test[target_columns_test] = df_test[target_columns_test].fillna(df_test[target_columns_test].mean())


# In[ ]:


missing_value_test = df_test.isnull().mean() * 100
missing_value_test


# In[ ]:


df.head()


# In[ ]:


df_test.head()


# Now that we have handled the missing data, we can start dealing with outliers. First, let’s examine the outliers.
# 
# The boxplot visualizes the distribution of data and helps us identify outliers. Here’s a brief explanation of how to interpret the boxplot for outlier detection:
# 
#   **Boxplot Components**
#   
#   Box: Represents the interquartile range (IQR), which includes the middle 50% of the data, spanning from the 25th percentile (Q1) to the 75th percentile (Q3).
#         
#   Whiskers: Extend from the edges of the box to the smallest and largest values within 1.5 times the IQR from the quartiles. Values outside this range are considered outliers.
#         
#   Outliers: Displayed as individual points beyond the whiskers. These points are significantly higher or lower than the rest of the data.

# In[ ]:


def plot_boxplot(df, column):
    plt.figure(figsize=(10, 6))
    

    sns.boxplot(x=df[column])
    
    plt.title(f'Boxplot for {column}')
    plt.xlabel(column)
    plt.show()

for column in df.columns:
    if pd.api.types.is_numeric_dtype(df[column]):
        plot_boxplot(df, column)


# In[ ]:


def handling_outliers(df):
    for column in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])

    return df

df = handling_outliers(df)


# Using the explanation of the boxplot above, you can examine the resulting graphs after addressing the outliers.

# In[ ]:


def plot_boxplot(df, column):
    plt.figure(figsize=(10, 6))
    

    sns.boxplot(x=df[column])
    
    plt.title(f'Boxplot for {column}')
    plt.xlabel(column)
    plt.show()

for column in df.columns:
    if pd.api.types.is_numeric_dtype(df[column]):
        plot_boxplot(df, column)


# In[ ]:


##We are performing the same operations for the test data as well.

df_test = handling_outliers(df_test)


# In[ ]:


for column in df_test.columns:
    if pd.api.types.is_numeric_dtype(df_test[column]):
        plot_boxplot(df_test, column)


# **Normalization process**
# 
# After dealing with the outliers, we can now begin normalizing our data. Normalization is crucial for using machine learning algorithms effectively. This process helps improve the performance of our model.
# 
# Min-Max Scaler is a normalization technique used to rescale features to a fixed range, usually [0, 1]. 

# In[ ]:


scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)


# In[ ]:


df_test = pd.DataFrame(scaler.fit_transform(df_test), columns=df_test.columns)


# In[ ]:


df.head()


# In[ ]:


df_test.head()


# Among the three different models I have trained, which I believe will provide the best results, I am applying the model that performed the best on the test data.

# In[ ]:


X = df.drop(columns=['class'])  
y = df['class'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modeller
models = {
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    'LightGBM': lgb.LGBMClassifier(),
    'CatBoost': cb.CatBoostClassifier(silent=True)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} Model Sonuçları:")
    print("Doğruluk Skoru:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


# We are making predictions using the CatBoost model.

# In[ ]:


y_pred = models['CatBoost'].predict(df_test)


# In[ ]:


y_pred = ['e' if label == 0 else 'p' for label in y_pred]

results = pd.DataFrame({
    'id': submission['id'],
    'class':y_pred
})

results.head()


# In[ ]:


results.to_csv('/kaggle/working/submission_final.csv', index=False)


# **Conclusion**
# 
# I undertook this project to enhance my beginner-level skills in machine learning and to provide inspiration for other programmers at a similar level. If you have any questions or need further clarification, feel free to reach out, and I'll do my best to address them.
# 
# I would greatly appreciate it if you could upvote this work as a recognition of the effort put into it. Thank you!
