#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# In[ ]:


df=pd.read_csv('/kaggle/input/titanic/train.csv')
test=pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


df.head(10)


# In[ ]:


test


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# - There is alot of null values within the table so we will remove the cabin column and we will remove the null rows of age because age is an essential factor and remove the null values of embarked 

# In[ ]:


df.drop(['Cabin'], axis=1,inplace=True)


# - Dropped Cabin Column because too many null values

# In[ ]:


df.dropna(inplace=True)


# In[ ]:


df.info()


# - Now there is no null values

# In[ ]:


df.duplicated().sum()


# - There is no Duplicates Values

# # UniVariate Analysis

# ### Survived Column

# In[ ]:


survived_counts = df['Survived'].value_counts()
survived_counts


# In[ ]:


plt.figure(figsize=(8, 6))
plt.pie(survived_counts, labels=['No', 'Yes'], autopct='%1.1f%%', colors=['#FF9999', '#66B2FF'], startangle=90)
plt.title('Survival Proportion')
plt.show()


# In[ ]:



plt.figure(figsize=(8, 6))
sns.barplot(x=survived_counts.index, y=survived_counts.values, palette='viridis')
plt.title('Survival Count')
plt.xlabel('Survived (1 = Yes, 0 = No)')
plt.ylabel('Number of Passengers')
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()


# - More People Died than Survived

# ### Pclass Column

# In[ ]:


df['Pclass'].value_counts()


# In[ ]:


sns.barplot(x=[1,2,3],y=df['Pclass'].value_counts(),palette='viridis')
plt.title("Ticket Class Count")
plt.ylabel('Number of tickets')
plt.xlabel('Ticket Class')
plt.show()


# - Most of the tickets were first Class and second and third class people were nearly the same count

# ### Sex Column

# In[ ]:


df.Sex.value_counts()


# In[ ]:


sns.barplot(x=['Male','Female'],y=df.Sex.value_counts(),palette='viridis')
plt.title("Gender Table")
plt.xlabel('Gender Type')
plt.ylabel('Gender Count')
plt.show()


# - More Male than Female where on the ship

# ### Age Column

# In[ ]:


df.Age.describe()


# In[ ]:


sns.histplot(x=df['Age'],bins=20)
plt.title("Age Distribution Table")
plt.xlabel("Age")
plt.ylabel('Age Count')
plt.show()


# - Most ages were from 20 to 30 years but there was a lot of babies too 

# ### SibSp column

# In[ ]:


df.SibSp.value_counts()


# In[ ]:


sns.barplot(x=['0','1','2','3','4','5'],y=df.SibSp.value_counts(),palette='viridis')


# - Most people were alone/ had no siblings on the Ship

# ### Parch Column

# In[ ]:


df.Parch.value_counts()


# In[ ]:


sns.barplot(x=['0','1','2','3','4','5','6'],y=df.Parch.value_counts(),palette='viridis')


# - Most People had no Parents/ Children on the ship

# ### Fare Column
# 

# In[ ]:


df.Fare.describe()


# In[ ]:


sns.histplot(df['Fare'], bins=20)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Fair Price")


# - The most fare from 0 to 100 but there is alot of outliers so we must detect the outliers

# #### Detecting Outliers

# In[ ]:


Q1= df['Fare'].quantile(0.25)
Q3=df['Fare'].quantile(0.75)

IQR=Q3-Q1

upper=Q3 + 1.5*IQR
Lower=Q1 - 1.5*IQR



# - There is alot of upper outliers and we are going to point them out 

# In[ ]:


df[df["Fare"]>70]


# - There are 95 People that paid higher than average amount of Fare which can be understood because people who paid alot more than average may be because they reserved later and so on

# ### Embarked Column

# In[ ]:


Embarked_counts=df['Embarked'].value_counts()
Embarked_counts


# In[ ]:


sns.barplot(x=['S','C','Q'],y=Embarked_counts,palette='viridis')


# # Multi Variate Analysis

# In[ ]:


df.groupby(['Sex'])['Survived'].value_counts()


# - Alot more Males Died than Females because at this instances males priotize women to survive than men
# - More Men Died than Survived
# - More Female Survived than Died

# In[ ]:


df.groupby(['Pclass'])['Survived'].value_counts()


# - Most People Died from Class 3 because this was the closest to the ocean 
# - People from Class 1 where the most ones survived as they were the highest from the ocean and they were priotized to get on boats 
# - People from Class 2 where almost 50/50 to survive

# In[ ]:


df.groupby(['Pclass','Sex'])['Survived'].value_counts()


# - Least men died where from Class 1 and only 3 women died from class 1 
# - Most men and women died where from Class 3
# - The only Class that more Females Died than Survived where in Class 1
# - The least percantage of men that died where on Class 1
# 

# - This Concludes that Class 1 was the most priotized and safest place on the Ship

# In[ ]:


bins = [0, 18, 35, 50, 65, float('inf')]
labels = ['0-18', '19-35', '36-50', '51-65', '66+']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)


# In[ ]:


df[df['Survived']==0].groupby(['AgeGroup'])['Survived'].value_counts()


# - Most people died were from ages 19-35 after it people from ages 36-50

# In[ ]:


df.groupby(['AgeGroup','Sex'])['Survived'].value_counts()


# In[ ]:


df.groupby(['SibSp'])['Survived'].value_counts()


# In[ ]:


df.groupby(['Parch'])['Survived'].value_counts()


# In[ ]:


df.head()


# In[ ]:


test.head()


# # Machine Learning

# ### Logistic Regression

# #### Preproccessing

# In[ ]:


df.info()


# In[ ]:


df.corr(numeric_only=True)


# In[ ]:


df.drop(['PassengerId','Ticket','Name','AgeGroup'],axis=1,inplace=True)


# In[ ]:


df.head()


# In[ ]:


encoder= LabelEncoder()
df['Sex']=encoder.fit_transform(df['Sex'])
df['Embarked']=encoder.fit_transform(df['Embarked'])
df['Survived']=encoder.fit_transform(df['Survived'])
test['Sex'] = encoder.fit_transform(test['Sex'])


# In[ ]:


X=df.drop(columns=["Survived",'Fare','Embarked'])
y=df.Survived


# In[ ]:


X


# In[ ]:


y


# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


X_train = df[["Pclass", "Sex", "SibSp", "Parch"]]
X_valid=df[["Pclass", "Sex", "SibSp", "Parch"]]
y_train = df['Survived']
y_valid=df['Survived']


# In[ ]:


X_test = test[["Pclass", "Sex", "SibSp", "Parch"]]


# In[ ]:


X_test


# ## Logistic Regression

# In[ ]:


model=LogisticRegression(max_iter=12000)
model.fit(X_train,y_train)


# In[ ]:


X_train


# In[ ]:


X_test


# In[ ]:


y_pred=model.predict(X_valid)


# In[ ]:


model.score(X_train,y_train)*100


# In[ ]:


model.score(X_valid,y_valid)*100


# In[ ]:


rate=accuracy_score(y_valid,y_pred)*100


# In[ ]:




# In[ ]:




# In[ ]:


y_pred=model.predict(X_test)


# In[ ]:


output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': y_pred})
output.to_csv('submission.csv', index=False)


# In[ ]:


output


# In[ ]:




