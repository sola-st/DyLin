#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv("/kaggle/input/playground-series-s4e8/train.csv")
df.info()


# In[ ]:


df.head()


# In[ ]:


df['class'].value_counts()


# In[ ]:


df.head(10)


# In[ ]:


df.isna().sum()


# In[ ]:


df_cat_col = [col for col in df.columns if df[col].dtype == 'object']
df_cat_col


# In[ ]:


df_num_col = [col for col in df.columns if df[col].dtype in ['int64','float64']]
df_num_col


# ## Categorical columns cleaning and analysis

# In[ ]:


df_cat = df[df_cat_col]
df_cat.isnull().sum()/len(df_cat)


# In[ ]:


# remove cols with more than 50% missing values
df_cat = df_cat[[col for col in df_cat.columns if df_cat[col].isna().sum() < len(df_cat)*0.4]]


# In[ ]:


df_cat.info()


# In[ ]:


df_cat.isnull().sum()/len(df_cat)


# In[ ]:


# fill the missing values with mode values in all remaining columns
df_cat['cap-surface'] = df_cat['cap-surface'].fillna(df_cat['cap-surface'].mode()[0])
df_cat['gill-attachment'] = df_cat['gill-attachment'].fillna(df_cat['gill-attachment'].mode()[0])


# In[ ]:


df_cat.isnull().sum()


# ## Numerical data analysis and cleaning

# In[ ]:


df_num = df[df_num_col]
df_num.isnull().sum()


# In[ ]:


df_num = df_num.drop('id', axis=1)
df_num.columns


# In[ ]:


df_num.head()


# In[ ]:


df_num.info()


# In[ ]:


df_cat.head()


# In[ ]:


df_num.head()


# In[ ]:


len(df_cat)


# In[ ]:


len(df_num)


# ### Merge the categorical and numerical columns and EDA

# In[ ]:


# this is new df with updated rows
ndf = pd.concat([df_cat, df_num], axis=1)


# In[ ]:


ndf.info()


# In[ ]:


ndf.isnull().sum()


# In[ ]:


# drop the rows with missing values
ndf = ndf.dropna()


# In[ ]:


ndf.info()


# In[ ]:


ndf.isnull().sum()


# In[ ]:


ndf['class'].value_counts()


# In[ ]:


# visualize the numerical features with class as hue
plt.figure(figsize=(10, 5))
sns.boxplot(x='class', y='cap-diameter', data=ndf)
plt.title('Boxplot of cap diameter by Class')
plt.show()


# In[ ]:


# violin plot for the distribution analysis
plt.figure(figsize=(10, 5))
sns.boxplot(x='class', y='stem-height', data=ndf)
plt.title('Violin plot of stem hieght by Class')
plt.show()


# In[ ]:


plt.figure(figsize=(10, 5))
sns.boxplot(x='class', y='stem-width', data=ndf)
plt.title('Violin plot of stem width by Class')
plt.show()


# In[ ]:


# unique values in each column
ndf.nunique()


# In[ ]:


ndf.head()


# In[ ]:


ndf.describe()


# In[ ]:


ndf.info()


# In[ ]:


# create a pie chart of season
season_counts = ndf['season'].value_counts()

# plot the pie chart 
plt.pie(season_counts, labels=season_counts.index, autopct='%1.1f%%', startangle=90)
plt.title("seasons in mushrooms dataset")
plt.show()


# In[ ]:


ndf.groupby(['cap-shape'])['cap-diameter'].mean()


# In[ ]:


# correlation between num cols
ndf[[col for col in ndf.columns if ndf[col].dtype == 'float64']].corr()


# In[ ]:


ndf.nunique()


# In[ ]:


ndf.groupby(['class','season'])[['cap-diameter','stem-height','stem-width']].agg(['mean','median'])


# In[ ]:


fig, axs = plt.subplots(3, 1, figsize=(10, 15))
sns.histplot(ndf['cap-diameter'], kde=False, bins=10, ax=axs[0])
axs[0].set_title("Histogram of cap diameter")

sns.histplot(ndf["stem-height"], kde=False, bins=10, ax=axs[1])
axs[1].set_title("Histogram of stem height")

sns.histplot(ndf['stem-width'], kde=False, bins=10, ax=axs[2])
axs[2].set_title("Histogram of stem width")

plt.suptitle("Histograms of numerical columns")
plt.show()


# In[ ]:


pd.pivot_table(ndf, values='cap-shape', index='class', columns='season', aggfunc='count').T.plot(kind='bar')


# In[ ]:


pd.pivot_table(ndf, values=['cap-surface','cap-color'], index='class', columns=['season'], aggfunc='count').T.plot(kind='bar')


# ### Data encoding and transformations

# In[ ]:


ndf.head()


# In[ ]:


# encode the cat cols and transform the num cols

