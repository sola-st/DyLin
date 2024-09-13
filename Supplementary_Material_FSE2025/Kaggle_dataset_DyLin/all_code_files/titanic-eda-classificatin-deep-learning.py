#!/usr/bin/env python
# coding: utf-8

# # Titanic Classification project

# In[ ]:


import numpy as np 
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Data Dictionary
# PassengerId: Unique identifier for each passenger.
# 
# Pclass: Ticket class of the passenger (1 = First Class, 2 = Second Class, 3 = Third Class).
# 
# Name: Full name of the passenger.
# 
# Sex: Gender of the passenger (male = male, female = female).
# 
# Age: Age of the passenger.
# 
# SibSp: Number of siblings and spouses aboard with the passenger.
# 
# Parch: Number of parents and children aboard with the passenger.
# 
# Ticket: Ticket number of the passenger.
# 
# Fare: Fare paid by the passenger.
# 
# Cabin: Cabin number of the passenger.
# 
# Embarked: Port where the passenger boarded the ship (C = Cherbourg, Q = Queenstown, S = Southampton).
# 

# <img src="https://png.pngtree.com/thumb_back/fh260/background/20230516/pngtree-titanic-is-a-big-old-ship-in-the-water-image_2572380.jpg">

# # importing Library

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error


# ## Importing Data 

# In[ ]:


import pandas as pd 


# In[ ]:


df=pd.read_csv("/kaggle/input/titanic/test.csv")


# ## İnfo About data

# In[ ]:


df.head(3)


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()  #boş değerleri sahip sutunlar 

df.describe()
# # <span style="color: green;">Steps to Follow</span>
# 
# 1- Reading the Data
# 
# 2- Exploring the Data
# 
# 3- Selecting Useful Columns
# 
# 4- Filling Missing Data with EDA
# 
# 5- Converting Object Data to int or float
# 
# 6- Defining x and y for Regression and Classification
# 
# 7- Model Creation and Training
# 

# In[ ]:


df=pd.read_csv("/kaggle/input/titanic/train.csv")
df.head(2)


# # EDA

# In[ ]:


df2=pd.read_csv("/kaggle/input/titanic/test.csv") 
df2.shape


# In[ ]:


df=pd.concat([df,df2])
df.shape


# In[ ]:


df.isnull().sum()


# In[ ]:


df.head()


# In[ ]:


df.corr(numeric_only=True)


# In[ ]:


df.describe()


# ## Boş değerlerin Doldurluması

# In[ ]:


df['Age'].value_counts()


# In[ ]:


df['Title']=df['Name'].str.extract('([A-Za-z]+)\.')


# In[ ]:


df['Title'].value_counts()


# In[ ]:


df['Title']=df['Title'].replace(['Ms','Mlle'],'Miss')
df['Title']=df['Title'].replace(['Mme','Countess','Lady','Dona'],"Mrs")
df['Title']=df['Title'].replace(['Rev','Dr','Col','Major','Sir','Don','Jonkheer','Capt'],'Mr')


# In[ ]:


df.Title.value_counts()


# In[ ]:


df['Age']=df['Age'].fillna(df.groupby('Title')['Age'].transform('median'))


# In[ ]:


df.Age.isnull().sum()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.Fare.describe()


# In[ ]:


ortalama_fiyat=df.Fare.mean()
df.Fare=df.Fare.fillna(df.Fare.mode()[0]) 
# Fiyat değerini boş değerlerini fiyatın mod değerlerini yani ortanca olarak doldurumuz


# In[ ]:


df.isnull().sum()


# In[ ]:


df.Cabin.value_counts()


# In[ ]:


df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])


# In[ ]:


import seaborn as sns
sns.countplot(x=df.Embarked)


# In[ ]:


sns.countplot(x=df.Pclass)  # sınıflarına göre kaö 


# In[ ]:


df.info()


# In[ ]:


df.columns


# # x ve y Değerlerinin Belirleme

# In[ ]:


train=df[:891]  # 891 ine kadar al ve train e ata
test=df[891:]   # 891 den başlar ve değerleri al
train.shape,test.shape


# In[ ]:


x=train.drop(['PassengerId','Name','Cabin','Ticket','Survived'],axis=1)
y=train[['Survived']]


# In[ ]:


x=pd.get_dummies(x,drop_first=True)  #ob


# In[ ]:


x.shape


# In[ ]:


x.head()


# In[ ]:


from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier


# In[ ]:


g=GaussianNB()
b=BernoulliNB()
r=RandomForestClassifier()
gr=GradientBoostingClassifier()


# In[ ]:


test.head(3)


# In[ ]:


test=test.drop(['PassengerId','Name','Cabin','Ticket','Survived'],axis=1)


# In[ ]:


test.head(3)


# In[ ]:


test=pd.get_dummies(test,drop_first=True)
test.head(6)


# """def hesapla(x,y):
#     #fit 
#     g.fit(x,y)
#     b.fit(x,y)
#     r.fit(x,y)
#     gr.fit(x,y)
#     # predic
#     pg=g.predict(test)
#     pb=b.predict(test)
#     pr=r.predict(test)
#     pgr=gr.predict(test)
#     
#     return pg,pb,pr,pgr
# hesapla(x,y)"""    

# In[ ]:


test


# ## Model oluşturma ve eğitimi

# In[ ]:


g.fit(x,y)
tahmin=g.predict(test)
tahmin


# In[ ]:


tahmin1=pd.DataFrame()
tahmin1['PassengerId']=df2['PassengerId']
tahmin1['Survived']=tahmin


# In[ ]:


tahmin1
type(tahmin1)


# In[ ]:


tahmin1['Survived']=tahmin1.Survived.astype('int32')


# In[ ]:


tahmin1.to_csv('Titanic_Tahmin1.csv',index=False)


# ## Diğerleri için aynı yöntem 

# In[ ]:


b.fit(x,y)
r.fit(x,y)
gr.fit(x,y)

btahmin=b.predict(test)
rtahmin=r.predict(test)
grtahmin=gr.predict(test)

tahmin2=pd.DataFrame()
tahmin3=pd.DataFrame()
tahmin4=pd.DataFrame()

tahmin2['PassengerId']=df2['PassengerId']
tahmin3['PassengerId']=df2['PassengerId']
tahmin4['PassengerId']=df2['PassengerId']

tahmin2['Survived']=btahmin.astype('int32')
tahmin3['Survived']=rtahmin.astype('int32')
tahmin4['Survived']=grtahmin.astype('int32')

tahmin2.to_csv('Titanic_Tahmin2.csv',index=False)
tahmin3.to_csv('Titanic_Tahmin3.csv',index=False)
tahmin4.to_csv('Titanic_Tahmin4.csv',index=False)


# In[ ]:





# In[ ]:


btahmin.shape


# In[ ]:


btahmin,rtahmin,grtahmin


# # Problem is solved :)

# In[ ]:


df.head()


# # Deep Learning

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Veriyi ayır
train = df[:891]
test = df[891:]

x = train.drop(['PassengerId', 'Name', 'Cabin', 'Ticket', 'Survived'], axis=1)
y = train['Survived']

test = test.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1)


# In[ ]:


combined = pd.concat([x, test], keys=['train', 'test'])

# One-hot encoding
combined = pd.get_dummies(combined, drop_first=True)

x = combined.xs('train')
test = combined.xs('test')

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
test_scaled = scaler.transform(test)


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(units=64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=4, activation='relu'))

model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:





# In[ ]:


history = model.fit(X_train, y_train, epochs=11, validation_data=(X_val, y_val), batch_size=50)


# In[ ]:


y_pred_val = (model.predict(X_val) > 0.5).astype("int32")

accuracy = accuracy_score(y_val, y_pred_val)

y_pred_test = (model.predict(test_scaled) > 0.5).astype("int32")


# In[ ]:


submission = pd.DataFrame({
    "PassengerId": df[891:]["PassengerId"],
    "Survived": y_pred_test.ravel()  # Tahmin sonuçlarını düzleştiriyoruz
})

submission.to_csv('submission.csv', index=False)

