#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


# <img src="header.png">

# In[ ]:


df = pd.read_csv(r"/kaggle/input/titanic/train.csv")


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.isna().sum()


# In[ ]:


mean_value = df['Age'].mean()
df['Age'] = df['Age'].fillna(mean_value)


# In[ ]:


df['Embarked'] = df['Embarked'].fillna('S')


# In[ ]:


df.info()


# In[ ]:


df = df.drop(['Name'],axis=1)
df = df.drop(['Ticket'],axis=1)
df = df.drop(['Cabin'],axis=1)


# In[ ]:


df.info()


# In[ ]:


# --> male = 0 , female = 1
df['Sex'] = df['Sex'].replace({'male': 0, 'female': 1})


# In[ ]:


df['Sex'] = df['Sex'].astype(int)


# In[ ]:


df.info()


# In[ ]:


# C = 0 , Q = 1 , S = 2
df['Embarked'] = df['Embarked'].replace({'C': 0, 'Q': 1 , 'S':2})
df['Embarked'] = df['Embarked'].astype(int)
df.info()


# In[ ]:


df.to_csv('Cleaned_titanic.csv', index=False)


# # Exploratory Data Analysis (EDA)
# 
# Explore the relationships between variables and identify patterns Relationship between survival and other variables

# In[ ]:


# plt.pie(df['Survived'].value_counts())

plt.pie(df['Survived'].value_counts(),
        labels = df['Survived'].value_counts().index,
        autopct = '%1.1f%%',
        startangle=90)

plt.title('Survival Distribution')
plt.show()


# In[ ]:


# Explore the distribution of passengers' ages
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x='Age', kde=True, bins=20)
plt.title("Age Distribution")
plt.show()


# In[ ]:


# parch of parents / children aboard the Titanic

plt.figure(figsize=(8, 4))
sns.histplot(data=df, x='Parch', bins=12)
plt.title("parch Distribution")
plt.show()


# In[ ]:


# fare	Passenger fare

plt.figure(figsize=(8, 4))
sns.histplot(data=df, x='Fare', bins=12)
plt.title("Fare Distribution")
plt.show()


# In[ ]:


sns.heatmap(df.corr(), annot=True, fmt=".2f")


# In[ ]:


sns.countplot(x="Survived", hue="Sex", data=df ,palette='flare')
plt.title("Survival by Sex")
plt.show()


# In[ ]:


sns.countplot(x = 'Survived' , hue = 'Parch' ,data = df ,palette='flare')
plt.title("Survival by Parch")
plt.show()


# In[ ]:


sns.countplot(x = 'Survived' , hue = 'Pclass' ,data = df ,palette='flare')
plt.title("Survival by Pclass")
plt.show()


# In[ ]:


sns.countplot(x = 'Survived' , hue = 'Embarked' ,data = df ,palette='flare')
plt.title("Survival by Embarked")
plt.show()


# In[ ]:


sns.histplot(x = 'Age' , hue = 'Survived' ,data = df ,kde=True ,palette='rocket')
plt.title("Survival by Parch")
plt.show()


# # modling

# ### 1- LogisticRegression

# In[ ]:


x = df.drop(columns=['Survived'])
y = df['Survived']

x_train , x_test , y_train ,y_test = train_test_split(x,y ,test_size=0.3)


# In[ ]:


LRM = LogisticRegression(penalty='l2',max_iter=100,solver='newton-cholesky')
LRM


# In[ ]:


LRM.fit(x_train,y_train)
LRM.score(x_train,y_train)


# In[ ]:


LRM.score(x_test,y_test)


# In[ ]:


y_pred = LRM.predict(x_test)
y_pred


# In[ ]:


CM =confusion_matrix(y_test,y_pred)

sns.heatmap(CM,annot=True)


# ### 2- SVC

# In[ ]:


SVC = SVC(C=10 ,kernel='linear')
SVC.fit(x_train,y_train)


# In[ ]:


SVC.score(x_train,y_train)


# In[ ]:


SVC.score(x_test,y_test)


# In[ ]:


y_pred = SVC.predict(x_test)
y_pred


# In[ ]:


CM =confusion_matrix(y_test,y_pred)

sns.heatmap(CM,annot=True)


# ### 3- Neural Network

# In[ ]:


MLPC = MLPClassifier(activation='identity',
                     solver='lbfgs',
                     alpha=0.0001,
                     batch_size='auto',
                     learning_rate='constant',
                     learning_rate_init=0.001,
                     power_t=0.5,
                     max_iter=200)

MLPC.fit(x_train,y_train)


# In[ ]:


y_prid = MLPC.predict(x_test)


# In[ ]:


accuracy_score(y_test,y_prid)


# In[ ]:


CM =confusion_matrix(y_test,y_prid)

sns.heatmap(CM,annot=True)


# ### 4- K-Means

# In[ ]:


km = KMeans(n_clusters=2)


# In[ ]:


x['cluster'] = km.fit_predict(x)
x['cluster']


# In[ ]:


x


# In[ ]:


CM = confusion_matrix(df['Survived'],x['cluster'])
CM


# # the best Model is LogisticRegression
# with accuracy --> 84%

# In[ ]:




