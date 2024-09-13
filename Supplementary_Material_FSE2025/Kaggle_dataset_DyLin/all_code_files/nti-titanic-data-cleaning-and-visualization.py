#!/usr/bin/env python
# coding: utf-8

# # **Import Libraries**

# In[ ]:


import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.express as px


# ## **Check for File Existence**

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## **Read CSV File**

# In[ ]:


df = pd.read_csv("/kaggle/input/titanic/train.csv")


# In[ ]:


df


# ## **Explore The Data**

# In[ ]:


df_explore = df.copy()


# In[ ]:


df_explore.info()


# In[ ]:


df_explore.dtypes


# In[ ]:


df_explore.columns


# In[ ]:


df_explore.describe().T


# In[ ]:


df_explore.describe(include="O") #include object


# ## **Missing Values**

# In[ ]:


df_miss = df_explore.copy()


# In[ ]:


df_miss.isnull().sum()


# In[ ]:


## persantage of null values

df_miss.isnull().sum() * 100 / len(df_miss)


# # **Data Cleaning**

# In[ ]:


df_clean = df_miss.copy()


# In[ ]:


sns.heatmap(df_clean.isnull())


# In[ ]:


df_clean.dropna(inplace = True) #to be able to change in the original data


# In[ ]:


sns.heatmap(df_clean.isnull())


# In[ ]:


df_clean.isnull().sum()


# In[ ]:


len(df_clean) 


# In[ ]:


len(df)


# Just 183 Out of 891 !!!

# In[ ]:


df_clean.duplicated().sum()


# ## **Filling Null Values**

# In[ ]:


df_fillTheNull = df_miss.copy()


# In[ ]:


df_fillTheNull.columns


# In[ ]:


sns.heatmap(df_fillTheNull.isnull())


# In[ ]:


df_fillTheNull.isnull().sum()


# In[ ]:


df_fillTheNull.drop(["Cabin"],axis = 1, inplace = True)


# In[ ]:


df_fillTheNull.isnull().sum()


# In[ ]:


df_fillTheNull["Age"].value_counts()


# In[ ]:


df_fillTheNull["Age"].isnull()


# In[ ]:


df_fillTheNull["Age"].isnull().sum()


# In[ ]:


df_fillTheNull[df_fillTheNull["Age"].isnull()]


# In[ ]:


df_fillTheNull.isnull().sum()


# In[ ]:


df_fillTheNull["Age"].interpolate(method= "linear", inplace = True)


# In[ ]:


df_fillTheNull.isnull().sum()


# In[ ]:


df_fillTheNull["Embarked"] = df_fillTheNull["Embarked"].fillna(method="ffill")


# In[ ]:


df_fillTheNull.isnull().sum()


# In[ ]:


sns.heatmap(df_fillTheNull.isnull())


# In[ ]:


len(df_fillTheNull)


# In[ ]:


len(df)


# ## Far better than before!

# # **Data Visualization**

# In[ ]:


df_viz = df_fillTheNull.copy()


# In[ ]:


df_viz.columns


# ## *1. What is the overall survival rate?*

# In[ ]:


# Calculate survival rate
survival_rate = df_viz['Survived'].mean()

# Visualization
plt.figure(figsize=(6, 4))
df_viz['Survived'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral'])
plt.title('Overall Survival Rate')
plt.ylabel('')
plt.show()


# ## *2. How does survival rate differ between males and females?*

# In[ ]:


# Group by sex and calculate survival rate
survival_by_sex = df_viz.groupby('Sex')['Survived'].mean()

# Visualization
plt.figure(figsize=(6, 4))
sns.barplot(x='Sex', y='Survived', data=df_viz, palette='pastel')
plt.title('Survival Rate by Sex')
plt.ylabel('Survival Rate')
plt.show()


# ## *3. What is the distribution of ages among passengers?*

# In[ ]:


# Descriptive statistics for age
age_description = df_viz['Age'].describe()

# Visualization
plt.figure(figsize=(8, 6))
sns.histplot(df_viz['Age'], kde=True, color='lightblue')
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# ## *4. How does survival rate vary with age?*

# In[ ]:


# Visualization
plt.figure(figsize=(10, 6))
sns.histplot(df_viz[df_viz['Survived'] == 1]['Age'], kde=True, color='green', label='Survived')
sns.histplot(df_viz[df_viz['Survived'] == 0]['Age'], kde=True, color='red'  , label='Did not Survive')
plt.title('Survival Rate by Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# ## *5. What is the survival rate for different passenger classes (Pclass)?*

# In[ ]:


# Group by Pclass and calculate survival rate
survival_by_class = df_viz.groupby('Pclass')['Survived'].mean()

# Visualization
plt.figure(figsize=(6, 4))
sns.barplot(x='Pclass', y='Survived', data=df_viz, palette='muted')
plt.title('Survival Rate by Passenger Class')
plt.ylabel('Survival Rate')
plt.show()


# ## *6. What is the distribution of fares paid by passengers?*

# In[ ]:


# Descriptive statistics for fare
fare_description = df_viz['Fare'].describe()

# Visualization
plt.figure(figsize=(8, 6))
sns.histplot(df_viz['Fare'], kde=True, color='purple')
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()


# ## *7. How does survival rate vary with fare?*

# In[ ]:


# Visualization
plt.figure(figsize=(10, 6))
sns.histplot(df_viz[df_viz['Survived'] == 1]['Fare'], kde=True, color='blue', label='Survived')
sns.histplot(df_viz[df_viz['Survived'] == 0]['Fare'], kde=True, color='orange', label='Did not Survive')
plt.title('Survival Rate by Fare')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# ## *8. What is the distribution of passengers by embarkation point (Embarked)?*
# 
# ### S -> England
# ### C -> France
# ### Q -> Ireland

# In[ ]:


# Count of passengers by embarkation point
embarked_count = df_viz['Embarked'].value_counts()

# Visualization
plt.figure(figsize=(6, 4))
sns.countplot(x='Embarked', data=df_viz, palette='coolwarm')
plt.title('Passenger Count by Embarkation Point')
plt.ylabel('Count')
plt.show()


# ## *9. How does survival rate vary by embarkation point?*
# 
# ### S -> England
# ### C -> France
# ### Q -> Ireland

# In[ ]:


# Group by Embarked and calculate survival rate
survival_by_embarkation = df_viz.groupby('Embarked')['Survived'].mean()

# Visualization
plt.figure(figsize=(6, 4))
sns.barplot(x='Embarked', y='Survived', data=df_viz, palette='coolwarm')
plt.title('Survival Rate by Embarkation Point')
plt.ylabel('Survival Rate')
plt.show()


# ## *10. What is the distribution of family size on board (SibSp + Parch)?*

# In[ ]:


# Add a new column for family size
df_viz['FamilySize'] = df_viz['SibSp'] + df_viz['Parch']

df_viz


# In[ ]:


# Descriptive statistics for family size
family_size_description = df_viz['FamilySize'].describe()

# Visualization
plt.figure(figsize=(8, 6))
sns.histplot(df_viz['FamilySize'], kde= True, bins=8, color='purple')
plt.title('Family Size Distribution')
plt.xlabel('Family Size (SibSp + Parch)')
plt.ylabel('Frequency')
plt.show()


# ## *11. How does survival rate vary with family size?*

# In[ ]:


# Visualization
plt.figure(figsize=(10, 6))
sns.barplot(x='FamilySize', y='Survived', data=df_viz, palette='viridis')
plt.title('Survival Rate by Family Size')
plt.ylabel('Survival Rate')
plt.show()


# ## *12. What is the survival rate for passengers traveling alone vs. with family?*

# In[ ]:


# Create a column indicating if the passenger is alone
df_viz['IsAlone'] = (df_viz['FamilySize'] == 0).astype(int)

df_viz


# In[ ]:


# Group by IsAlone and calculate survival rate
survival_by_alone = df_viz.groupby('IsAlone')['Survived'].mean()

# Visualization
plt.figure(figsize=(6, 4))
sns.barplot(x='IsAlone', y='Survived', data=df_viz, palette='magma')
plt.title('Survival Rate: Traveling Alone vs With Family')
plt.ylabel('Survival Rate')
plt.xticks([0, 1], ['With Family', 'Alone'])
plt.show()


# ## *13. What is the correlation between different features in the dataset?*

# In[ ]:


# Select only numeric columns
numeric_df = df_viz.select_dtypes(include=['float64', 'int64'])

# Calculate the correlation matrix
corr_matrix = numeric_df.corr()

# Visualization
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# ## *14. How does the ticket class (Pclass) distribution look like?*

# In[ ]:


# Count of passengers by Pclass
pclass_count = df_viz['Pclass'].value_counts()

# Visualization
plt.figure(figsize=(6, 4))
sns.countplot(x='Pclass', data=df_viz, palette='coolwarm')
plt.title('Passenger Count by Ticket Class')
plt.ylabel('Count')
plt.xticks([0, 1, 2], ['1st', '2nd', '3rd'])
plt.show()


# ## *15. What is the survival rate by ticket class and sex?*

# In[ ]:


# Visualization
plt.figure(figsize=(10, 6))
sns.catplot(x='Pclass', y='Survived', hue='Sex', data=df, kind='bar', palette='muted', height=6)
plt.title('Survival Rate by Ticket Class and Sex')
plt.ylabel('Survival Rate')
plt.show()

