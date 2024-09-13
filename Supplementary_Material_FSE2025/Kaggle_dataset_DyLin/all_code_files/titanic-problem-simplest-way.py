#!/usr/bin/env python
# coding: utf-8

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


# **-Importing the necessary dependencies and libraries**

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


# In[ ]:


train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')


# **-Understanding the basic details in the datasets**

# In[ ]:




# This shows that there are significant crucial missing points in both Train and Test datasets. These need to be addressed in the first place.
# 
# 
# - As the datasets shows that the Cabin column has a lot of missing data which cannot be engineered.
# - The Ticket column is mix of categorical and numerical data and drawing relation to the rate of survival is difficult and lengthens the process.
# - The range of distribution of Fare column is huge, that could interfere with the training.
# Therefore all three columns are being dropped.

# In[ ]:


train_data = train_data.drop(columns=['Cabin', 'Ticket', 'Fare'], axis=1)
test_data = test_data.drop(columns=['Cabin', 'Ticket', 'Fare'] , axis=1)


# Filling the missing values in Age column(train data only) with Mean Age, similarly Embarked column with Mode.
# Age column in test data has a significant number of data missing that needs to be fixed and cannot be filled with mean, meadian or mode. Those would change the nature of the data.
# 

# In[ ]:


train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])


# **-Drawing the first relation : between Title & Sex**

# In[ ]:


combine = [train_data,test_data]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(r'([A-Za-z]+)\.', expand=False)



# This shows the type of people were on the ship. There were:
# 
#  - Officers/Crew,
#  - Royals
#  - Awardees(Ms,Mme,Mlle{these are awards bestowed in England})
#  - Rev : stands for Reverends, who were the religious or priestly class in England.
#  - The Adult aged Men
#  - The Adult aged Women
#  - Young Boys
#  - Young Girls
#   Thus the next step is to group them together.

# In[ ]:


for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Dr', 'Major'], 'Officers')
    dataset['Title'] = dataset['Title'].replace(['Rev'], 'RevD')
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Don', 'Jonkheer', 'Lady', 'Sir'], 'Royals')
    dataset['Title'] = dataset['Title'].replace(['Mme', 'Mlle', 'Ms'], 'Awardees')
    dataset['Title'] = dataset['Title'].replace(['Mr'], 'OldM')
    dataset['Title'] = dataset['Title'].replace(['Mrs'], 'OldW')
    dataset['Title'] = dataset['Title'].replace(['Master'], 'YoungM')
    dataset['Title'] = dataset['Title'].replace(['Miss'], 'YoungW')


# **-Relation No. 2 : between Survival and Title**

# In[ ]:


title_survived_relation  = (['Title','Survived'])
plt.figure(figsize=(8, 6))
sns.barplot(x='Title', y='Survived', data=train_data, palette='viridis')
plt.title('Survival Rate by Title')
plt.xlabel('Title')
plt.ylabel('Survival Rate')
plt.xticks(rotation=45)
plt.show()


# This show the clear relation between title and survival rate.

# **-Relation No. 3 : between Sex and Survival**

# This show the clear relation between title and survival rate.
# - Survival rate of Older Men is low.
# - Reverends (are older men as well)
# - Rate of Older Women and Young Women survival is high
# - Some Royals too survived
# - Most Officers died.
# - All Awardees survived.
# 
# There is a clear overlapping of Gender and Title. Thus the next step would be to find relation between Gender and Survival Rate.
# 
# 

# In[ ]:


gender_survived_relation  = (['Sex','Survived'])
plt.figure(figsize=(8, 6))
sns.barplot(x='Sex', y='Survived', data=train_data, palette='viridis')
plt.title('Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Survival Rate')
plt.xticks(rotation=45)
plt.show()


# #Clearly evident that the survival rate of the Female is higher than Male.

# **-Data Encoding: Encoding the Categorical Data to Numerical Data.**

# In[ ]:


title_encode = {'Officers' : 1, 'RevD' : 2, 'Royals' : 3, 'Awardees' : 4, 'OldM' : 5, 'OldW' : 6, 'YoungM' : 7, 'YoungW' : 8}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_encode)
    dataset['Title'] = dataset['Title'].fillna(0)

train_data = train_data.drop(columns='Name', axis=1)
test_data = test_data.drop(columns='Name', axis=1)


train_data = train_data.replace({'Sex': {"male" :0, "female" : 1}, 'Embarked': {"S" : 0, "C" : 1, "Q" : 2}})
test_data = test_data.replace({'Sex': {"male" :0, "female" : 1}, 'Embarked': {"S" : 0, "C" : 1, "Q" : 2}})


# **-Managing the Age Column in Test Data**

# In[ ]:


mean_age_by_title = train_data.groupby('Title')['Age'].mean()

#Since there is a relation between Title, Age and Gender with survival rate. I've used title to fill the missing age columns in the test data by mean age of the title.

def fill_missing_age(row, mean_age_by_title):
    if pd.isnull(row['Age']):
        return mean_age_by_title.get(row['Title'], np.nan)
    return row['Age']

for dataset in [train_data, test_data]:
    dataset['Age'] = dataset.apply(lambda row: fill_missing_age(row, mean_age_by_title), axis=1)


# **-Age Binning**

# In[ ]:


bins = [0, 10, 30, 40, 50, 60, 70, 80, 90]
labels = ['0-10', '11-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90']
train_data['AgeBin'] = pd.cut(train_data['Age'], bins=bins, labels=labels, right=True)
test_data['AgeBin'] = pd.cut(test_data['Age'], bins=bins, labels = labels, right=True)

agebin_encode = {'0-10': 1, '11-30': 2, '31-40': 3, '41-50': 4, '51-60': 5, '61-70': 6, '71-80': 7, '81-90': 8}
for dataset in [train_data, test_data]:
    dataset['AgeBin'] = dataset['AgeBin'].astype(str)
    dataset['AgeBin'] = dataset['AgeBin'].map(agebin_encode)
    dataset['AgeBin'] = dataset['AgeBin'].fillna(0).astype(int)


train_data = train_data.drop(columns='Age', axis=1)
test_data = test_data.drop(columns='Age', axis=1)


# **-Final relation between Gender, Age and Title**

# In[ ]:


age_gender_survival = train_data.groupby(['AgeBin', 'Sex'])['Survived'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(x='Sex', y='Survived', hue='AgeBin', data=age_gender_survival)
plt.title('Survival Rate by Gender-Title and Age Binned')
plt.xlabel('Gender')
plt.ylabel('Survival Rate')
plt.legend(title='Age Bin')
plt.show()

g = sns.FacetGrid(train_data, col='Sex', row='AgeBin', margin_titles=True)
g.map_dataframe(sns.barplot, x='AgeBin', y='Survived', estimator=lambda x: x.mean())
g.set_axis_labels('Age Bin', 'Survival Rate')
g.set_titles(col_template='{col_name}', row_template='{row_name}')
plt.show()


# **-Model Training**
# 
# I have used Cross-Val Score to evaluate all the scores of the models

# In[ ]:



x = train_data.drop(['PassengerId', 'Survived'], axis=1)
y = train_data['Survived']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=2)


models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVC':SVC(),
    'GaussianNB':GaussianNB(),
    'Random Forest': RandomForestClassifier(),
    'KNeighbour': KNeighborsClassifier(),
    'XGBoost':XGBClassifier(),
}

for name, model in models.items():
    cv_scores = cross_val_score(model, x, y, cv=StratifiedKFold(n_splits=5))
    print(f'{name} Cross-Validation Scores: {cv_scores}')
    print(f'{name} Average Cross-Validation Score: {np.mean(cv_scores):.4f}')
    print('-' * 50)


# ****

# **-Using SVC as it scored more than 81% accuracy.**

# In[ ]:


best_model = SVC()
best_model.fit(x_train, y_train)
ids = test_data['PassengerId']
predictions = best_model.predict(test_data.drop('PassengerId', axis=1))

output = pd.DataFrame({'PassengerId': ids, 'Survived': predictions})
output.to_csv('submissionfile.csv', index=False)


# In[ ]:


#Thank You.

