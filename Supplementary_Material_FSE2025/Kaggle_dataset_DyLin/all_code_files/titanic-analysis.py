#!/usr/bin/env python
# coding: utf-8

# In[ ]:





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


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


data = pd.read_csv("/kaggle/input/titanic/train.csv")
data


# In[ ]:


df = data.copy()
df


# In[ ]:


sns.heatmap(df.isnull())


# In[ ]:


df.isna().mean()*100


# In[ ]:


df.drop(['Cabin'] , axis = 1  , inplace = True)


# In[ ]:


df.columns


# In[ ]:


male_avg = int(df[df['Sex'] == 'male']['Age'].mean())

female_avg = int(df[df['Sex'] == 'female']['Age'].mean())


# In[ ]:


male_avg


# In[ ]:


def fill_age(row):
    if pd.isna(row['Age']):
        if row['Sex'] == 'male':
            return male_avg
        else:
            return female_avg
    else:
        return row['Age']


df['Age'] = df.apply(fill_age, axis=1)   
df        


# In[ ]:


Embarked_mode = df["Embarked"].mode()[0]
Embarked_mode


# In[ ]:


df["Embarked"].fillna(Embarked_mode , inplace = True)


# In[ ]:


sns.heatmap(df.isnull())


# # What is the precentage of surviving in this dataset?

# In[ ]:


df['Survived'].replace({0: 'Dead', 1: 'Survived'}).value_counts()


# In[ ]:


labels = df['Survived'].replace({0: 'Dead', 1: 'Survived'}).value_counts()


# In[ ]:


labels.plot(kind = 'pie' , autopct='%1.1f%%')


# # Number of males and females on the titanic #

# In[ ]:


df.Sex.value_counts()


# In[ ]:


plt.pie(df.Sex.value_counts() , labels = ["male" , "female"] , autopct = "%1.1f%%")
plt.title("Genders on the Titanic")


# In[ ]:





# In[ ]:


df[(df['Sex'] == 'male') & (df['Survived'] == 1) ]["Survived"].sum()


# In[ ]:


df[(df['Sex'] == 'female') & (df['Survived'] == 1) ]["Survived"].sum()


# In[ ]:


gender = ['Males' , 'Females']
survival = [109 , 233]


# In[ ]:


plt.pie(survival , labels = gender , autopct='%1.1f%%')
plt.title('Survivors gender')


# In[ ]:


s_female = df[(df['Sex'] == 'female') & (df['Survived'] == 1) ]["Survived"].sum()
s_male = df[(df['Sex'] == 'male') & (df['Survived'] == 1) ]["Survived"].sum()
male_rate = (s_male/df[df['Sex'] == 'male'].value_counts().sum())*100
female_rate = (s_female/df[df['Sex'] == 'female'].value_counts().sum())*100


# In[ ]:


df[df['Sex'] == 'female'].value_counts().sum()


# In[ ]:


female_rate = s_female/df[df['Sex'] == 'female'].value_counts().sum()*100
female_rate


# ## Average age of survival
# 

# In[ ]:


df[df["Survived"] == 1]["Age"].mean()
df.Age.mean()


# In[ ]:


survivors = df[df["Survived"] == 1]


# In[ ]:


plt.hist(survivors['Age'], bins=9, edgecolor='black')

plt.xlabel('Age')
plt.ylabel('Number of Survivors')
plt.title('Histogram of Ages of Survivors')


# ## Average Fare paid by the survivors

# In[ ]:


survivors.Fare.mean()


# In[ ]:


survivors.Fare.median()


# ## Max and min fare paid for by a survivor

# In[ ]:


survivors.Fare.max()


# In[ ]:


survivors.Fare.min()


# ## Max and min fare paid for by a victim

# In[ ]:


df[df.Survived == 0].Fare.max()


# In[ ]:


df[df.Survived == 0].Fare.min()


# In[ ]:


plt.hist(survivors['Fare'] , bins = 20)


# In[ ]:


df.Fare.value_counts()


# ## Classes of survivors

# In[ ]:


survivors["Pclass"].value_counts()


# In[ ]:


plt.pie(survivors["Pclass"].value_counts() ,labels = survivors["Pclass"].value_counts().index ,autopct = "%1.1f%%")
plt.title("Survivors classes")


# ## Let's think in the opposite way!

# In[ ]:


class1_sur = df[(df["Pclass"] == 1) & (df["Survived"] == 1)]["Survived"].value_counts().sum()
class1_rate=class1_sur/df[df["Pclass"] == 1].value_counts().sum() *100
class2_sur = df[(df["Pclass"] == 2) & (df["Survived"] == 1)]["Survived"].value_counts().sum()
class2_rate=class2_sur/df[df["Pclass"] == 2].value_counts().sum() *100
class3_sur = df[(df["Pclass"] == 3) & (df["Survived"] == 1)]["Survived"].value_counts().sum()
class3_rate=class3_sur/df[df["Pclass"] == 3].value_counts().sum() *100



# In[ ]:


df['Pclass'].value_counts()


# In[ ]:


classes = ['Class 1', 'Class 2', 'Class 3']
rates = [class1_rate, class2_rate, class3_rate]

plt.bar(classes, rates, color=['blue', 'orange', 'green'])
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate (%)')
plt.title('Survival Rate by Passenger Class')


# In[ ]:


df[df["Pclass"] == 1]["Survived"].value_counts()


# ## Did survivors have sibiligns on the titanic?

# In[ ]:


surv_withsb = survivors[survivors["SibSp"] > 0].SibSp.value_counts().sum()


# In[ ]:


percent = surv_withsb / survivors.value_counts().sum() * 100


# In[ ]:


surv_withsb_df=survivors[survivors["SibSp"] > 0]
surv_withsb_df


# In[ ]:


plt.pie(survivors.SibSp.value_counts() ,labels = survivors.SibSp.value_counts().index ,  autopct= "%1.1f%%")
plt.title("Number of siblings with the survivors ")


# In[ ]:


not_survivors = df[df["Survived"] == 0]
plt.pie(not_survivors.SibSp.value_counts() ,labels = not_survivors.SibSp.value_counts().index ,  autopct= "%1.1f%%")
plt.title("Number of siblings with the not survivors ")


# ## Did Survivor have parents or child on the Titanic ?

# In[ ]:


surv_withpar = survivors[survivors["Parch"] > 0].SibSp.value_counts().sum()
percent2 = surv_withpar / survivors.value_counts().sum() * 100


# In[ ]:


plt.pie(survivors.Parch.value_counts() ,labels = survivors.Parch.value_counts().index ,  autopct= "%1.1f%%")
plt.title("Number of parents or childs with the survivors ")


# In[ ]:



plt.pie(not_survivors.Parch.value_counts() ,labels = not_survivors.Parch.value_counts().index ,  autopct= "%1.1f%%")
plt.title("Number of parents or childs with the not survivors ")


# # Did Embarked collerate with survival ? #

# In[ ]:


plt.pie(df.Embarked.value_counts() , labels = ["S" , "C" , "Q"] , autopct = "%1.1f%%")
plt.title("")


# In[ ]:


plt.pie(survivors.Embarked.value_counts() , labels = ["S" , "C" , "Q"] , autopct = "%1.1f%%")
plt.title("")

