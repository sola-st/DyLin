#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


def create_csv_file(x,y,filename):
    df=pd.DataFrame({'PassengerId':x, 'Transported':y})
    df.to_csv(filename, index=False)


# In[ ]:


# Read datasets
df_test = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv',index_col='PassengerId')
df_train = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv',index_col='PassengerId')


# In[ ]:


# Data Analysis


# In[ ]:


df_train


# In[ ]:


sns.displot(df_train['HomePlanet'])


# In[ ]:


sns.displot(df_train['Destination'])


# In[ ]:


sns.displot(df_train['Age'])


# In[ ]:


df_train['Transported'].value_counts().plot(kind='bar', color=['blue', 'orange']);


# In[ ]:


df_train['VIP'].value_counts().plot(kind='bar', color=['blue', 'orange']);


# In[ ]:


# Size of each dataset


# In[ ]:


df_train.head(5)


# In[ ]:


# Remove redundant information
# Redundant samples: check for duplicated samples

# Redudant features: check for columns with low variance (few values) and columns with only one value: std < |+-0.5|
std = df_train.select_dtypes(include=[np.number]).std()


# #### NaN Values Replacement

# In[ ]:


# Check for NaN values
df_train.isna().sum()


# In[ ]:


df_test.isna().sum()


# In[ ]:


## Remove Rows with NaN values ##
df_train_1 = df_train.copy()
df_test_1 = df_test.copy()

df_train_1 = df_train.dropna()


# In[ ]:


## Since we cannot remove rows of the test dataset, we're going to try replace NaN values with other values ##

# HomePlanet: Replace the missing values with the most frequent values
imp_1=SimpleImputer(strategy='most_frequent')
df_test_1['HomePlanet']=imp_1.fit_transform(df_test_1['HomePlanet'].values.reshape(-1,1)).flatten()

# CryoSleep: Replace the missing values with the mean value
imp_2=SimpleImputer(missing_values=np.NaN,strategy='mean')
df_test_1['CryoSleep']=imp_2.fit_transform(df_test_1['CryoSleep'].values.reshape(-1,1)).flatten()

# Cabin: Replace the missing values with the most frequent values
df_test_1['Cabin']=imp_1.fit_transform(df_test_1['Cabin'].values.reshape(-1,1)).flatten()

# Destination: Replace the missing values with the most frequent values
df_test_1['Destination']=imp_1.fit_transform(df_test_1['Destination'].values.reshape(-1,1)).flatten()

# Age: Replace the missing ages with the mean value
df_test_1['Age']=imp_2.fit_transform(df_test_1['Age'].values.reshape(-1,1)).flatten()

# VIP: Replace the missing values with the most frequent values
df_test_1['VIP']=imp_1.fit_transform(df_test_1['VIP'].values.reshape(-1,1)).flatten()

# RoomService: Replace the missing values with the mean value
df_test_1['RoomService']=imp_2.fit_transform(df_test_1['RoomService'].values.reshape(-1,1)).flatten()

# FoodCourt: Replace the missing values with the mean value
df_test_1['FoodCourt']=imp_2.fit_transform(df_test_1['FoodCourt'].values.reshape(-1,1)).flatten()

# ShoppingMall: Replace the missing values with the mean value
df_test_1['ShoppingMall']=imp_2.fit_transform(df_test_1['ShoppingMall'].values.reshape(-1,1)).flatten()

# Spa: Replace the missing values with the mean value
df_test_1['Spa']=imp_2.fit_transform(df_test_1['Spa'].values.reshape(-1,1)).flatten()

# VRDeck: Replace the missing values with the mean value
df_test_1['VRDeck']=imp_2.fit_transform(df_test_1['VRDeck'].values.reshape(-1,1)).flatten()

# Name: Replace the missing values with the most frequent values
df_test_1['Name']=imp_1.fit_transform(df_test_1['Name'].values.reshape(-1,1)).flatten()


# In[ ]:




# In[ ]:


#check datatypes of each column in dataset
df_train.dtypes


# In[ ]:


# Remove some columns with object as data type
df_train_1.drop(['Cabin','Name'],axis=1,inplace=True)
df_test_1.drop(['Cabin','Name'],axis=1,inplace=True)


# In[ ]:


# Replacing HomePlanet, CryoSleep, Destination and VIP values using label encoder (for each item gives an identification number)
le=LabelEncoder()

df_train_1['HomePlanet']=le.fit_transform(df_train_1['HomePlanet'])
df_test_1['HomePlanet']=le.fit_transform(df_test_1['HomePlanet'])

df_train_1['CryoSleep']=le.fit_transform(df_train_1['CryoSleep'])
df_test_1['CryoSleep']=le.fit_transform(df_test_1['CryoSleep'])

df_train_1['Destination']=le.fit_transform(df_train_1['Destination'])
df_test_1['Destination']=le.fit_transform(df_test_1['Destination'])

df_train_1['VIP']=le.fit_transform(df_train_1['VIP'])
df_test_1['VIP']=le.fit_transform(df_test_1['VIP'])


# In[ ]:


df_train_1


# In[ ]:


# Correlation Analysis - Pearson Correlation
plt.figure(figsize=(12,10))
cor = df_train_1.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[ ]:


#Selecting highly correlated features
relevant_features = cor[cor>0.5]
relevant_features


# In[ ]:


# Check for outliers using boxplot
df_train_1.plot(kind='box',color='Green',subplots=True,layout=(3,4),figsize=(12,12))


# In[ ]:


#plot distribution plot to check the age column distribution 
sns.distplot(df_train_1['Age'],color='Green')


# In[ ]:


# Separate the data into X and Y
y_1=df_train_1.iloc[:,-1]
x_1=df_train_1.iloc[:,:-1]

index_1 = df_test_1.index


# In[ ]:


# Normalize the data
sc=StandardScaler()
x_1=sc.fit_transform(x_1)
df_test_1 = sc.fit_transform(df_test_1)


# In[ ]:


# Split the training data into train and test sets
x_train_1,x_test_1,y_train_1,y_test_1=train_test_split(x_1,y_1,test_size=0.20,random_state=42)


# In[ ]:


# Logistic Regression using grid search
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.1, 1, 10, 100]
}

lr = LogisticRegression()
clf = GridSearchCV(lr, param_grid, cv=5)
clf.fit(x_train_1, y_train_1)


# In[ ]:


lr_1=LogisticRegression(penalty='l2', C=1)
lr_1.fit(x_train_1,y_train_1)
lgpred=lr_1.predict(x_test_1)


# In[ ]:


pred_1=lr_1.predict(df_test_1)
create_csv_file(index_1,pred_1, 'LRsubmission.csv')


# In[ ]:


# k-nearest neighbors using grid search
k_range = list(range(1, 31))
param_grid = dict(n_neighbors = k_range)

knn = KNeighborsClassifier()
grid_knn = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy',verbose=1)
grid_knn.fit(x_train_1, y_train_1)


# In[ ]:


knn_1=KNeighborsClassifier(n_neighbors=21)
knn_1.fit(x_train_1,y_train_1)
knnpred=knn_1.predict(x_test_1)


# In[ ]:


pred_1=knn_1.predict(df_test_1)
create_csv_file(index_1,pred_1,'KNNsubmission.csv')


# In[ ]:


# Decision tree using grid search
param_grid = {
    'criterion': ['gini', 'entropy']
}

dt = DecisionTreeClassifier()
clf_dt = GridSearchCV(dt, param_grid, cv=10, scoring='accuracy',verbose=1)
clf_dt.fit(x_train_1, y_train_1)


# In[ ]:


dt_1=DecisionTreeClassifier(criterion='gini')
dt_1.fit(x_train_1,y_train_1)
dtpred=dt_1.predict(x_test_1)


# In[ ]:


pred_1=dt_1.predict(df_test_1)
create_csv_file(index_1,pred_1, 'DTsubmission.csv')


# In[ ]:


# Support Vector Classification using grid search
param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [1, 10, 100]
}

svc=SVC()
clf_svc = GridSearchCV(svc, param_grid, cv=10, scoring='accuracy',verbose=1)
clf_svc.fit(x_train_1, y_train_1)


# In[ ]:


svc_1=SVC(kernel='rbf', C=1)
svc_1.fit(x_train_1,y_train_1)
svcpred=svc_1.predict(x_test_1)


# In[ ]:


pred_1=svc_1.predict(df_test_1)
create_csv_file(index_1,pred_1, 'SVMsubmission.csv')


# In[ ]:


# Random Forest using grid search
param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

rf=RandomForestClassifier(random_state=42)
rf_svc = GridSearchCV(rf, param_grid, cv=10, scoring='accuracy',verbose=1)
rf_svc.fit(x_train_1, y_train_1)


# In[ ]:


rf_1=RandomForestClassifier(n_estimators=500, max_features='sqrt',max_depth=7,criterion='gini')
rf_1.fit(x_train_1,y_train_1)
rfpred=rf_1.predict(x_test_1)


# In[ ]:


pred_1=rf_1.predict(df_test_1)
create_csv_file(index_1,pred_1, 'RFsubmission.csv')

