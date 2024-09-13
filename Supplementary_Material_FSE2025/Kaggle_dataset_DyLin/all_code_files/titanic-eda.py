#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings

plt.style.use('seaborn-v0_8-whitegrid')
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Load and Check Data

# In[ ]:


train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")


# In[ ]:


train_df.columns


# In[ ]:


train_df.head()


# In[ ]:


train_df.describe()


# # Variable Descripton
# 
# 1. PassengerId : unique id number to each passenger
# 1. Survived : passenger survive(1) or dead (0)
# 1. Pclass : passenger class
# 1. Name : name
# 1. Sex : gender of passenger
# 1. Age : age of passenger
# 1. SibSp : number of siblings/spouses
# 1. Parch : number of parents/children
# 1. Ticket : ticket number
# 1. Fare : amount of spent on ticket
# 1. Cabin : cabin category
# 1. Embarked : port where passenger embarked (C= Cherbourg, Q=Queenstown, S= Southampton)

# In[ ]:


train_df.info()


# # Univariate Variable Analysis

# ## Categorical Variable Analysis

# In[ ]:


def bar_plot(variable:str):
    """
    input: variable ex: "Sex"
    output: barplot & value count
    """
    # get feature
    var = train_df[variable]
    
    #count number of categorical variable(value/sample)
    varValue = var.value_counts()
    
    #visualize 
    plt.figure(figsize=(9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    
    print("{}: \n {}".format(variable, varValue))


# In[ ]:


category1 = ["Survived", "Sex", "Pclass", "Embarked", "SibSp", "Parch"]
for c in category1:
    bar_plot(c)


# ## Numerical Variable Analysis

# In[ ]:


def plot_hist(variable):
    plt.figure(figsize= (9,3))
    plt.hist(train_df[variable], bins=100)
    plt.xlabel(variable)
    plt.ylabel("frequency")
    plt.title("{} distrubition with hist".format(variable))
    plt.show()


# In[ ]:


numericVar =["Fare", "Age", "PassengerId"]
for n in numericVar:
    plot_hist(n)


# # Basic Data Analysis

# In[ ]:


#Pclass vs Survived

train_df[["Pclass", "Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by="Survived", ascending=False)


# In[ ]:


#Sex vs Survived

train_df[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by="Survived", ascending=False)


# In[ ]:


#SibSp vs Survived

train_df[["SibSp","Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by="Survived", ascending=False)


# In[ ]:


#Parch vs Survived

train_df[["Parch","Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by="Survived", ascending=False)


# # Outlier Detection

# In[ ]:


def outliers(df, features):
    outlier_indices = []
    
    for c in features:
        #first quartile
        Q1 = np.percentile(df[c], 25)
        
        #third quartile
        Q3 = np.percentile(df[c], 75)
        
        #IQR
        IQR = Q3 - Q1
        
        #outlier step 
        step = IQR*1.5
        
        # detect 
        
        outlier_list_col = df[(df[c] < Q1 - step) | (df[c]> Q3 + step)].index
        
        outlier_indices.extend(outlier_list_col)
        
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v>2)
    return multiple_outliers


# In[ ]:


train_df.loc[outliers(train_df, ["Age", "SibSp", "Parch", "Fare"])]

train_df = train_df.drop(outliers(train_df, ["Age", "SibSp", "Parch", "Fare"]), axis = 0).reset_index(drop=True)
# # Missing Value

# In[ ]:


len_train = len(train_df)
train_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)


# ## Find Missing Value

# In[ ]:


train_df.columns[train_df.isnull().any()]


# In[ ]:


train_df.isnull().sum()


# ## Fill Missing Value

# In[ ]:


train_df[train_df["Embarked"].isnull()]


# In[ ]:


train_df.boxplot(column='Fare', by="Embarked")
plt.show()


# In[ ]:


train_df["Embarked"] = train_df["Embarked"].fillna("C")
train_df[train_df["Embarked"].isnull()]


# In[ ]:


train_df[train_df["Fare"].isnull()]


# In[ ]:


train_df["Fare"] = train_df["Fare"].fillna((np.mean(train_df[train_df["Pclass"]==3]["Fare"])))


# # Visualization

# ## Correlation Between SibSp -- Parch -- Age -- Fare -- Survived

# In[ ]:


list1 = ["SibSp", "Parch", "Age", "Fare", "Survived"]
sns.heatmap(train_df[list1].corr(), annot = True, fmt = ".2f", cmap = "BuPu")
plt.show()


# ## SibSp -- Survived

# In[ ]:


ss = sns.catplot(x = "SibSp", y = "Survived", data = train_df, kind = "bar")
ss.set_ylabels("Survived Probability")
plt.show()


# * Having a lot of SibSp have less chance to survive.
# * if SibSp == 0 or 1 or 2, passenger has more chance to survive
# 

# ## Parch -- Survived

# In[ ]:


ps = sns.catplot(x = "Parch", y = "Survived", data = train_df, kind = "bar")
ps.set_ylabels("Survived Probability")
plt.show()


# * SibSp and Parch can be used for new feature extraction with th = 3
# * Small families have more chance to survive.
# * There is a std in survial of passenger with Parch = 3

# ## Pclass -- Survived

# In[ ]:


pcs = sns.catplot(x = "Pclass", y = "Survived", data = train_df, kind = "bar")
pcs.set_ylabels("Survived Probability")
plt.show()


# ## Age -- Survived

# In[ ]:


ags = sns.FacetGrid(train_df, col = "Survived")
ags.map(sns.distplot, "Age", bins = 25)
plt.show()


# * age <= 10 has a high survival rate.
# * oldest passenger survived.
# * large number of 20 years old did not survive.
# * most passenger are in 15 - 35 range.
# * use age feature in training
# * use age distrubition for missing values of age.

# ## Pclass -- Survived -- Age

# In[ ]:


psa = sns.FacetGrid(train_df, col = "Survived", row = "Pclass")
psa.map(plt.hist, "Age", bins = 25)
psa.add_legend()
plt.show()


# * Pclass is an important feature for model training.

# ## Embarked -- Sex -- Pclass -- Survived

# In[ ]:


esps = sns.FacetGrid(train_df, row = "Embarked")
esps.map(sns.pointplot, "Pclass", "Survived", "Sex")
esps.add_legend()
plt.show()


# * Female passengers have much better survival rate than males.
# * males have better survival rate in pclass 3 in C.
# * embarked and sex will be used in training

# ## Embarked -- Sex -- Fare -- Survived

# In[ ]:


esfs = sns.FacetGrid(train_df, row = "Embarked", col = "Survived")
esfs.map(sns.barplot, "Sex", "Fare")
esfs.add_legend()
plt.show()


# * passengers who pay higher fare have better survival. fare can be used as categorical for training.

# ## Fill Missing : Age Feature

# In[ ]:


train_df[train_df["Age"].isnull()]


# In[ ]:


sns.catplot(x = "Sex", y="Age", data = train_df, kind = "box")
plt.show()


# * sex is not informative for age prediction. age distrubition seems to be same.

# In[ ]:


sns.catplot(x = "Sex", y="Age", hue = "Pclass", data = train_df, kind = "box")
plt.show()


# * first class passengers are older than second class passengers and second class passengers are older than third class passengers.

# In[ ]:


sns.catplot(x = "Parch", y="Age",  data = train_df, kind = "box")
sns.catplot(x = "SibSp", y="Age",  data = train_df, kind = "box")

plt.show()


# In[ ]:


train_df["Sex"] = [1 if i == "male" else 0 for i in train_df["Sex"]]


# In[ ]:


sns.heatmap(train_df[["Age", "Sex", "SibSp", "Parch", "Pclass"]].corr(), annot=True)
plt.show()


# In[ ]:


index_nan_age = list(train_df["Age"][train_df["Age"].isnull()].index)
for i in index_nan_age:
    age_pred = train_df["Age"][(train_df["SibSp"]== train_df.iloc[i]["SibSp"]) &  (train_df["Parch"]== train_df.iloc[i]["Parch"]) & (train_df["Pclass"]== train_df.iloc[i]["Pclass"])]
    age_med = train_df["Age"].median()
    
    if not np.isnan(age_pred[i]):
        train_df["Age"].iloc[i] = age_pred

    else:
        train_df["Age"].iloc[i] = age_med


# # Feature Engineering

# ## Name -- Title

# In[ ]:


train_df["Name"].head(10)


# In[ ]:


name = train_df["Name"]
train_df["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in name]


# In[ ]:


train_df["Title"].head(10)


# In[ ]:


sns.countplot(x="Title", data = train_df)
plt.xticks(rotation = 60)
plt.show()


# In[ ]:


#convert to categorical
train_df["Title"] = train_df["Title"].replace(["Lady", "the Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"], "other")
train_df["Title"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i == "Mlle" or i == "Mrs" else 2 if i == "Mr" else 3 for i in train_df["Title"]]


# In[ ]:


sns.countplot(x="Title", data = train_df)
plt.xticks()
plt.show()


# In[ ]:


t = sns.catplot(x = "Title", y = "Survived", data = train_df, kind = "bar")
t.set_xticklabels(["Master",  "Mrs", "Mr", "Other"])
t.set_ylabels("Survival Probability")
plt.show()


# In[ ]:


train_df.drop(labels = ["Name"], axis = 1 , inplace=True)


# In[ ]:


train_df.head()


# In[ ]:


train_df = pd.get_dummies(train_df, columns = ["Title"])
train_df.head()


# ## Family Size

# In[ ]:


train_df["Fsize"] = train_df["SibSp"] + train_df["Parch"] + 1


# In[ ]:


train_df.head()


# In[ ]:


fsize = sns.catplot(x= "Fsize", y = "Survived", data = train_df, kind = "bar")
fsize.set_ylabels("Survival")
plt.show()


# In[ ]:


train_df["family_size"] = [1 if i < 5 else 0 for i in train_df["Fsize"]]


# In[ ]:


train_df.head(20)


# In[ ]:


sns.countplot(x = "family_size", data = train_df)
plt.show()


# In[ ]:


fsize = sns.catplot(x= "family_size", y = "Survived", data = train_df, kind = "bar")
fsize.set_ylabels("Survival")
plt.show()


# small families have more chance to survive than large families.

# In[ ]:


train_df = pd.get_dummies(train_df, columns = ["family_size"])
train_df.head()


# In[ ]:




