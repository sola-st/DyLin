#!/usr/bin/env python
# coding: utf-8

# # Introduction
# This is a beginner level notebook which covers Exploratory Data Analysis, Feature Engineering, and Modeling to make predictions for the Titanic: Machine Learning from Disaster competition. 
# 
# My goal with this notebook is to provide some techniques and ideas that might be helpful to others when approaching this competition.
# 
# ![titanic.png](attachment:f682b3dd-79f3-4e9d-b99f-70b3a9937559.png)
# 
# I'd like to credit this [notebook](https://www.kaggle.com/code/gunesevitan/titanic-advanced-feature-engineering-tutorial?scriptVersionId=27280410) for some of its ideas which inspired me.

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, cross_val_score


# # Exploratory Data Analysis

# ### Data overview
# - **PassengerId**: Unique id for each passenger. No effect on the target feature.
# - **Survived**: Whether or not the passenger survived. This is the target feature.
#   - 0 = No, 1 = Yes
# - **Pclass**: Reflects the socio-economic status of the passenger.
#   - 1 = 1st, Upper Class
#   - 2 = 2nd, Middle Class
#   - 3 = 3rd, Lower Class
# - **Name**: The name of the passenger. Includes the title of the passenger, such as "Mr.", "Mrs.", and "Master.".
# - **Sex**: Gender of the passenger, either "male" or "female". 
# - **Age**: The age of the passenger in years.
# - **SibSp**: # of siblings / spouses aboard the Titanic.
# - **Parch**: # of parents / children aboard the Titanic.
# - **Ticket**: The Ticket number.
# - **Fare**: Passenger fare.
# - **Cabin**: Cabin number of the passenger.
# - **Embarked**: Which port the passenger embarked from.
#   - C = Cherbourg
#   - Q = Queenstown
#   - S = Southampton

# In[ ]:


train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# ### Features and Survival

# In[ ]:


plt.figure(figsize=(6, 4))

plt.hist(train_data[train_data['Survived'] == 1]['Age'], bins=30, alpha=0.5, label='Survived', color='green', edgecolor='black')
plt.hist(train_data[train_data['Survived'] == 0]['Age'], bins=30, alpha=0.5, label='Did Not Survive', color='red', edgecolor='black')

plt.title('Age Distribution by Survival Status')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)

plt.show()


# - Younger passengers, especially those < 5 years old, seem to survive at a higher rate.
# - Older passengers seem to have a lower survival rate, especially around 40 - 75 years old.

# In[ ]:


plt.figure(figsize=(6, 4))

plt.hist(train_data[train_data['Survived'] == 1]['Fare'], bins=30, alpha=0.5, label='Survived', color='green', edgecolor='black')
plt.hist(train_data[train_data['Survived'] == 0]['Fare'], bins=30, alpha=0.5, label='Did Not Survive', color='red', edgecolor='black')

plt.title('Fare Distribution by Survival Status')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)

plt.show()


# - Lower fares seemed to have survived less.
# - Higher fares seemed to survive more.
# - This can be correlated with the socio-economic status of the passenger.

# In[ ]:


df = train_data.copy()
df['Sex_Label'] = df['Sex'].map({'male': 'Male', 'female': 'Female'})

survival_counts = df.groupby(['Sex_Label', 'Survived']).size().unstack()
survival_counts.plot(kind='bar', stacked=False, figsize=(6, 4), color=['salmon', 'skyblue'])

plt.title('Survival Counts by Sex')
plt.xlabel('Sex')
plt.ylabel('Number of People')
plt.xticks(rotation=0)
plt.legend(title='Survived', labels=['Did not Survive', 'Survived'])
plt.grid(True)

plt.show()


# - Females have a higher rate of survival than Males.

# In[ ]:


survival_counts = train_data.groupby(['Embarked', 'Survived']).size().unstack()

survival_counts.plot(kind='bar', stacked=False, figsize=(6, 4), color=['salmon', 'skyblue'])

plt.title('Survival Counts by Embarked')
plt.xlabel('Embarked')
plt.ylabel('Number of People')
plt.xticks(rotation=0)
plt.legend(title='Survived', labels=['Did not Survive', 'Survived'])
plt.grid(True)

plt.show()


# - Those who embarked from "S" had the lowest rate of survival.
# - Those who embarked from "C" had the highest rate to survive.

# In[ ]:


survival_counts = train_data.groupby(['Pclass', 'Survived']).size().unstack()

survival_counts.plot(kind='bar', stacked=False, figsize=(6, 4), color=['salmon', 'skyblue'])

plt.title('Survival Counts by Pclass')
plt.xlabel('Pclass')
plt.ylabel('Number of People')
plt.xticks(rotation=0)
plt.legend(title='Survived', labels=['Did not Survive', 'Survived'])
plt.grid(True)

plt.show()


# - 1st class passengers have a high chance of survival, while 3rd class passengers have a high chance of dying.
# - Socio-economic status seems to play a part for survival.

# # Feature engineering

# ### Filling in missing values

# In[ ]:


train_data.isnull().sum()


# Missing values in train data:
# - Age (177 missing values)
# - Cabin (687 missing values)
# - Embarked (2 missing values)

# In[ ]:


test_data.isnull().sum()


# Missing values in test data:
# - Age (86 missing values)
# - Cabin (327 missing values)
# - Fare (1 missing value)

# In[ ]:


all_data = pd.concat([train_data, test_data])


# **Update Sex to binary value**
# - I change the Sex feature to a binary value, 0 = male, 1 = female. Later, this will be changed to one-hot encoded values.

# In[ ]:


all_data['Sex'] = all_data['Sex'].map({'male': 0, 'female': 1})


# **Age**
# - I find the highest correlated feature with age. I then impute the missing values based on the median age for all rows with the same value of that correlated feature.

# In[ ]:


features = ["Survived", "Pclass", "Sex", "SibSp", "Parch", "Fare", "Age"]

# Only select rows without missing Age values
age_present = all_data[all_data['Age'].notna()]
age_data = age_present[features]

corr = age_data.corr()

plt.figure(figsize=(7, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


# - The highest correlated feature with Age is the Pclass. I fill the missing values by the median of the age by Pclass.

# In[ ]:


median_ages = all_data.groupby("Pclass")['Age'].transform('median')
all_data['Age'] = all_data['Age'].fillna(median_ages)


# **Cabin**  
# - I replace this column with the following two features:
#   - Deck: The first letter in the cabin number. Set to "U" if Cabin is missing.
#   - CabinMissing: 1 if cabin is missing, 0 otherwise.

# In[ ]:


all_data["Deck"] = all_data["Cabin"].str[0].fillna("U")
all_data["CabinMissing"] = all_data["Cabin"].isna().astype(int)


# In[ ]:


survival_counts = all_data[:891].groupby(['Deck', 'Survived']).size().unstack()

survival_counts.plot(kind='bar', stacked=False, figsize=(10, 6), color=['salmon', 'skyblue'])

plt.title('Survival Counts by Deck')
plt.xlabel('Deck')
plt.ylabel('Number of People')
plt.xticks(rotation=0)
plt.legend(title='Survived', labels=['Did not Survive', 'Survived'])
plt.grid(True)

plt.show()


# **Embarked**
# - I fill the missing values with the most common value for similar rows. 
# - There are only 2 missing values, so I set them to the mode embarked value of the passengers who are also female and on Deck "B".

# In[ ]:


missing_embarked = all_data[all_data["Embarked"].isna()]
missing_embarked


# In[ ]:


def mode_or_nan(series):
    mode = series.mode()
    return mode[0] if not mode.empty else np.nan

all_data["Embarked"] = all_data.groupby(["Sex", "Deck"])["Embarked"].transform(
    lambda x: x.fillna(mode_or_nan(x))
)


# **Fare**
# - I fill the missing fare values with the mean fare value for the highest correlated feature, which is the Pclass.

# In[ ]:


all_data[all_data.Fare.isna()]


# In[ ]:


all_data["Fare"] = all_data.groupby("Pclass")["Fare"].transform(
    lambda x: x.fillna(x.mean())
)


# ### New Features

# **TicketCount**
# - This is set equal to the number of passengers with the same Ticket value. This might imply that they are part of the same group or family.

# In[ ]:


all_data["TicketCount"] = all_data.groupby("Ticket")["Ticket"].transform("count")


# **FamilySize**
# - This is a new feature that is equal to SibSp + Parch + 1 (the passenger themself).
# - The family size might be another indicator for the survival probability of the passenger.

# In[ ]:


all_data["FamilySize"] = all_data["SibSp"] + all_data["Parch"] + 1


# **IsAlone**
# - Set to 1 if the passenger is traveling alone, 0 otherwise. This is based on the FamilySize feature's value.

# In[ ]:


all_data["IsAlone"] = (all_data["FamilySize"] == 1).astype(int)


# **AgeBin**
# - This is an ordinal feature that groups the passengers by what range their age falls in.

# In[ ]:


all_data["Age"].describe()


# In[ ]:


bins = [x*10 for x in range(10)]
labels = range(1, len(bins))

all_data["AgeBin"] = pd.cut(all_data["Age"], bins=bins, labels=labels, right=False)


# **Title**
# - This is extracted from the Name feature of each passenger.
# - It seems to carry potential information such as their gender (Mr. for male, Mrs. for female) and their age (Master. is given to boys).

# In[ ]:


all_data["Title"] = all_data["Name"].apply(lambda x: x.split(',')[1].split()[0])
all_data["Title"].unique()


# In[ ]:


plt.figure(figsize=(7, 6))
all_data["Title"].value_counts().plot(kind='bar', color='skyblue')
plt.show()


# In[ ]:


match_list = ["the", "Jonkheer.", "Dona.", "Mlle.", "Mme.", "Don."]

all_data[all_data["Name"].apply(lambda x: x.split(',')[1].split()[0]).isin(match_list)]


# - Jonkheer., Dona., Mlle., MMe. and Don. seem to be the names of passengers.
# - "the" is for PassengerId 760. It's followed by the term "Countess".
# 
# I'll group together these titles by similarity into one of four groups:
# - Mr
# - Mrs/Miss
# - Master
# - Officer/Professional

# In[ ]:


replacements = {
 'Mr.': 'Mr', 
 'Mrs.': 'Mrs/Miss', 
 'Miss.': 'Mrs/Miss', 
 'Master.': 'Master', 
 'Don.': 'Mr', 
 'Rev.': 'Officer/Professional', 
 'Dr.': 'Officer/Professional', 
 'Mme.': 'Mrs/Miss',
 'Ms.': 'Mrs/Miss', 
 'Major.': 'Officer/Professional', 
 'Lady.': 'Mrs/Miss', 
 'Sir.': 'Mr', 
 'Mlle.': 'Mrs/Miss', 
 'Col.': 'Officer/Professional', 
 'Capt.': 'Officer/Professional', 
 'the': 'Mrs/Miss',
 'Jonkheer.': 'Mr', 
 'Dona.': 'Mrs/Miss'
}

all_data["Title"] = all_data["Title"].replace(replacements)

all_data["Title"].unique()


# ### One-hot encoding
# - I transform some of the categorical variables into one-hot encoded features. This makes it possible for the features to be fed to the models which require numerical input.

# In[ ]:


one_hot_columns = ["Pclass", "Title", "Embarked", "Deck", "Sex"]

encoder = OneHotEncoder(sparse=False, dtype=int)
encoded_features = encoder.fit_transform(all_data[one_hot_columns])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(one_hot_columns))
encoded_df.index = all_data.index


# In[ ]:


df_final = pd.concat([all_data, encoded_df], axis=1)


# ### Drop columns

# In[ ]:


drop_cols = ["Ticket", "Cabin", "Name", "PassengerId", "Age", "SibSp", "Parch", "Pclass", "Embarked", "Title", "Deck", "Sex"]

df_final.drop(columns=drop_cols, inplace=True)


# In[ ]:


df_final.columns


# In[ ]:


df_final.head()


# In[ ]:


# Split the data back into the train and test sets
df_train = df_final[:891]
df_test = df_final[891:]


# In[ ]:




# In[ ]:


df_test.drop(columns=['Survived'], inplace=True)


# In[ ]:


X_train = df_train.drop(columns="Survived")
y_train = df_train["Survived"].values

X_test = df_test


# # Modeling

# ### Random Forest Classifier
# - I choose to use the Random Forest Classifier as it seems to perform well on tabular data.

# ### Grid search
# - I use grid search with 5-fold cross validation. 

# In[ ]:


rf_classifier = RandomForestClassifier(random_state=42)


# In[ ]:


param_grid = {
    'n_estimators': [1100, 1500, 1750],
    'max_depth': [5, 6, 7],
    'min_samples_split': [5, 6],
    'min_samples_leaf': [5, 6],
    'oob_score': [True]
}


# In[ ]:


grid_search_cv = GridSearchCV(estimator=rf_classifier, param_grid=param_grid,
                             cv=5, n_jobs=-1, verbose=0, scoring='accuracy')

grid_search_cv.fit(X_train, y_train)

best_params = grid_search_cv.best_params_
best_model = grid_search_cv.best_estimator_


best_accuracy = grid_search_cv.best_score_


# ### Submission

# In[ ]:


predictions = best_model.predict(X_test)

output = pd.DataFrame({
    'PassengerId': test_data.PassengerId,
    'Survived': predictions.astype(int)
})
output.to_csv('submission.csv', index=False)

