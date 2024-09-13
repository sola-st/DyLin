#!/usr/bin/env python
# coding: utf-8

# <h1 style = "color: yellow"> Binary Prediction of Poisonous Mushrooms </h1>
# 
# ![image.png](attachment:79ad55d0-e4e8-4bd9-a209-975951ac4bc5.png)

# # 1. **Introduction**

# <h3 style = "color: orange"> Description </h3>
# 
# <p> The dataset for this competition (both train and test) was generated from a deep learning model trained on the UCI Mushroom dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance. </p>
# 
# <h3 style = "color: orange"> Notice </h3>
# 
# <p> Unlike many previous Tabular Playground datasets, data artifacts have not been cleaned up. There are categorical values in the dataset that are not found in the original. It is up to the competitors how to handle this. </p>
# 
# <h3 style = "color: orange"> Goal </h3>
# 
# <p> The goal of this competition is to predict whether a mushroom is edible or poisonous based on its physical characteristics. </p>

# # 2. **CSV, Module Imports & AutoML Installation**

# In[ ]:


# Automated Machine Learning -> AutoGluon from AWS (Amazon)
#get_ipython().system('pip install autogluon.tabular')


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


# Data Visualization
import matplotlib.pyplot as plt

# Data Preprocessing
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce

# model
from autogluon.tabular import TabularDataset, TabularPredictor


# # 3. **Data Visualization / EDA**

# In[ ]:


# train dataset (with target class)
train_csv = TabularDataset("/kaggle/input/playground-series-s4e8/train.csv")
train_csv.head()


# In[ ]:


# test dataset (without target class)
test_csv = TabularDataset("/kaggle/input/playground-series-s4e8/test.csv")
test_csv.head()


# In[ ]:


# shapes of datasets


# In[ ]:


# Count of the NaN values

# train
train_csv.isna().sum()


# In[ ]:


# total number of NaN for train & test


# In[ ]:


# test
test_csv.isna().sum()


# In[ ]:


# train NaN features vs test

# train
train_csv_bool = dict(train_csv.isna().sum() == 0)
for feature, boolean in train_csv_bool.items():
    if boolean == True:
        print(f"Feature \"{feature}\" has 0 NaN values")


# In[ ]:


# test
test_csv_bool = dict(test_csv.isna().sum() == 0)
for feature, boolean in test_csv_bool.items():
    if boolean == True:
        print(f"Feature \"{feature}\" has 0 NaN values")


# In[ ]:


# the single test NaN feature value compared to train is "stem-height"
var = dict(test_csv.isna().sum())


# In[ ]:


# Analyzing the distribution of the target variable

# count the occurrences of each unique value in the 'class' column
class_counts = train_csv['class'].value_counts()



# In[ ]:


# plot the result
fig, ax = plt.subplots(1, 2, figsize=(12, 6), subplot_kw=dict(aspect="equal"), layout='constrained')

class_counts.plot(kind="bar", ax=ax[0])
ax[0].set_title("Class Distribution (Count)")
ax[0].set_xlabel("Class")
ax[0].set_ylabel("Count")

ax[1].pie(class_counts, labels=class_counts.index, autopct="%1.1f%%", startangle=140)
ax[1].set_title("Class Distribution (Percentage)")

plt.show()


# <h3 style = "color: green"> Observations </h3>
# 
# <p>The dataset appears to be imbalanced with a higher proportion of poisonous mushrooms.</p>

# In[ ]:


# Explore the distributions of categorical and numerical features

# categorical
categorical_cols = train_csv.select_dtypes(include='object').columns.tolist()
categorical_cols.remove('class')

for col in categorical_cols:
    print(f"\nTop 10 values for {col}:\n", train_csv[col].value_counts().head(10).to_markdown(numalign="left", stralign="left"))


# In[ ]:


# numerical

numerical_cols = train_csv.select_dtypes(include='float64').columns.tolist()

train_csv[numerical_cols].hist(bins=30, figsize=(15, 10), layout=(2, 2))

plt.suptitle('Histograms of Numerical Columns', fontsize=14)

plt.show()


# <h3 style = "color: green"> Observations </h3>
# 
# <p>Most categorical columns have null values and some columns like 'veil-type' have a dominant category.</p>

# In[ ]:


# Visualize the relationship between the categorical features and the target variable using bar plots of relative frequencies

num_cols = 2
num_rows = int(np.ceil(len(categorical_cols) / num_cols))

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
axes = axes.flatten()  

for i, col in enumerate(categorical_cols):
    grouped_data = train_csv.groupby(col)['class'].value_counts(normalize=True).unstack()

    grouped_data.plot(kind='bar', ax=axes[i])

    axes[i].set_title(f'Relative Frequency of Class by {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Relative Frequency')
    axes[i].legend(title='Class')

for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


# <h3 style = "color: green"> Observations </h3>
# 
# <p>Certain categories within categorical features exhibit strong associations with the target variable ('class'), suggesting their potential importance in predicting mushroom edibility.</p>

# In[ ]:


# The relationship between the numerical features and the target variable using box plots and KDE plots

# box plots with overlayers
num_cols = 2
num_rows = int(np.ceil(len(numerical_cols) / num_cols))

fig1, axes1 = plt.subplots(num_rows, num_cols, figsize=(10, 4*num_rows))

axes1 = axes1.flatten()

for i, col in enumerate(numerical_cols):
    train_csv.boxplot(column=col, by='class', ax=axes1[i])

    axes1[i].set_title(f'Box Plot of {col} by Class')
    axes1[i].set_xlabel('Class')
    axes1[i].set_ylabel(col)

for j in range(i+1, len(axes1)):
    axes1[j].axis('off')

fig1.tight_layout()


# In[ ]:


# KDE plots
num_cols = 2
num_rows = int(np.ceil(len(numerical_cols) / num_cols))
fig2, axes2 = plt.subplots(num_rows, num_cols, figsize=(10, 4*num_rows))
axes2 = axes2.flatten()

for i, col in enumerate(numerical_cols):
    for class_val in train_csv['class'].unique():
        subset = train_csv[train_csv['class'] == class_val]
        subset[col].plot(kind='kde', ax=axes2[i], label=class_val)

    axes2[i].set_title(f'KDE Plot of {col} by Class')
    axes2[i].set_xlabel(col)
    axes2[i].set_ylabel('Density')
    axes2[i].legend(title='Class')

for j in range(i+1, len(axes2)):
    axes2[j].axis('off')

fig2.tight_layout()
plt.show()


# <h3 style = "color: green"> Observations </h3>
# 
# <p>The distributions of numerical features vary across the two classes ('e' and 'p'), indicating their potential usefulness in distinguishing between edible and poisonous mushrooms.</p>

# # 4. **Model Training, Prediction & Evaluating**

# In[ ]:


# first remove ids from test and train

train_csv = train_csv.drop('id', axis = 1)
test_csv = test_csv.drop('id', axis = 1)

# AutoGluon

predictor = TabularPredictor(label = 'class', path = '/kaggle/working/', eval_metric = 'mcc', problem_type='binary')
predictor.fit(train_csv, presets='best_quality', time_limit = 3600 * 10, excluded_model_types = ['KNN'])


# In[ ]:


# top models
predictor.leaderboard()


# In[ ]:


# predict on test data
y_pred = predictor.predict(test_csv)


# # 5. **Submit Prediction**

# In[ ]:


# submission
sub = pd.read_csv('/kaggle/input/playground-series-s4e8/sample_submission.csv')
sub['class'] = y_pred
sub.to_csv('submission.csv', index = False)

