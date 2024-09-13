#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style='darkgrid', font_scale=2)
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC


# In[ ]:


train_df=pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')
test_df=pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')


# # **understand data**

# In[ ]:


train_df.shape


# In[ ]:


data=pd.concat([train_df,test_df],axis=0)
data


# In[ ]:


categorical_col=[col for col in data.columns if data[col].dtype=='O']
numerical_col=[col for col in data.columns if data[col].dtype!='O']
categorical_col


# In[ ]:


data.info()


# In[ ]:


data.isnull().sum()/data.shape[0]*100


# In[ ]:


data.describe()


# In[ ]:


data.describe(include='O')


# In[ ]:


data.nunique()


# # feature engeering

# In[ ]:


data.reset_index(drop=True,inplace=True)


# In[ ]:


data.index.duplicated().sum()


# In[ ]:


list=[]
for i in range(len(data)):
    if data['Age'][i]<10:
        list.append(0)
    elif data['Age'][i]<20:
        list.append(1)
    elif data['Age'][i]<30:
        list.append(2)
    elif data['Age'][i]<40:
        list.append(3)
    elif data['Age'][i]<50:
        list.append(4)
    elif data['Age'][i]<60:
        list.append(5)
    else:
        list.append(6)
    


# In[ ]:


data['Age_groups']=pd.DataFrame(list)


# In[ ]:


data['total_spent']=data['Spa']+data['ShoppingMall']+data['FoodCourt']+data['RoomService']+data['VRDeck']


# In[ ]:


numerical_col.extend(['Age_groups','total_spent'])


# # **data visualization**

# In[ ]:


data.hist(bins=20,figsize=(18,25),color='brown')


# In[ ]:


# Melt the DataFrame to long form
train_df_melted = train_df.melt(id_vars='Transported', value_vars='HomePlanet')

# Create the countplot using the melted DataFrame
sns.countplot(x='value', hue='Transported', data=train_df_melted, palette='dark')

# Set labels and title
plt.xlabel('Home Planet')
plt.ylabel('Count')
plt.title('Count Plot of Home Planet by Transported')
plt.show()


# In[ ]:


# Melt the DataFrame to long form
train_df_melted = train_df.melt(id_vars='Transported', value_vars='VIP')

# Create the countplot using the melted DataFrame
sns.countplot(x='value', hue='Transported', data=train_df_melted, palette='dark')

# Set labels and title
plt.xlabel('Home Planet')
plt.ylabel('Count')
plt.title('Count Plot of VIP by Transported')
plt.show()


# In[ ]:


# Melt the DataFrame to long form
train_df_melted = train_df.melt(id_vars='Transported', value_vars='CryoSleep')

# Create the countplot using the melted DataFrame
sns.countplot(x='value', hue='Transported', data=train_df_melted, palette='dark')

# Set labels and title
plt.xlabel('Home Planet')
plt.ylabel('Count')
plt.title('Count Plot of CryoSleep by Transported')
plt.show()


# In[ ]:


# Melt the DataFrame to long form
plt.figure(figsize=(10,5))
train_df_melted = train_df.melt(id_vars='Transported', value_vars='Destination')

# Create the countplot using the melted DataFrame
sns.countplot(x='value', hue='Transported', data=train_df_melted, palette='dark')

# Set labels and title
plt.xlabel('Home Planet')
plt.ylabel('Count')
plt.title('Count Plot of Destination by Transported')
plt.show()


# In[ ]:


plt.pie(data.Transported.value_counts(), shadow=True,  autopct='%.1f%%')
plt.title('Transported ', size=18)
plt.legend(['False', 'True'], loc='best', fontsize=12)
plt.show()


# In[ ]:


plt.figure(figsize=(15,10))
sns.heatmap(data[numerical_col].corr(),annot=True,cmap='Greens')


# In[ ]:


plt.figure(figsize=(8,5))
sns.heatmap(data.isna(),cbar=False, cmap='viridis')
plt.show()


# # **data preprocessing**

# In[ ]:


data.drop(['PassengerId','Name'],axis=1,inplace=True)
categorical_col.remove('PassengerId')
categorical_col.remove('Name')


# In[ ]:


categorical_col.remove('Transported')
for col in categorical_col:
  data[col].fillna(data[col].mode()[0],inplace=True)
for col in numerical_col:
  data[col].fillna(data[col].median(),inplace=True)


# In[ ]:


data[['deck','num','side']]=data['Cabin'].str.split('/',expand=True)


# In[ ]:


data.drop('Cabin',axis=1,inplace=True)
categorical_col.remove('Cabin')
categorical_col.extend(['deck','num','side'])


# In[ ]:


data.isnull().sum()


# In[ ]:


from operator import le
le=LabelEncoder()
for col in categorical_col:
  data[col]=le.fit_transform(data[col])
data['Transported']=le.fit_transform(data['Transported'])
data


# In[ ]:


plt.figure(figsize=(30,15))
sns.heatmap(data.corr(),annot=True,cmap='Greens')
plt.show()


# In[ ]:


x=data.drop('Transported',axis=1)
y=data['Transported']


# In[ ]:


x_train=x.iloc[:8693,:]
y_train=y.iloc[:8693]
x_test=x.iloc[8693:,:]


# In[ ]:


x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.15,random_state=42)


# # **modeling**

# # *logestic regression*

# In[ ]:


lr=LogisticRegression()
lr.fit(x_train,y_train)
cv=cross_val_score(lr,x_train,y_train,cv=5)


# In[ ]:


param_lr={
    'penalty':['l1','l2','elasticnet'],
    'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]
}


# In[ ]:


rs=RandomizedSearchCV(
    estimator=lr,
    param_distributions=param_lr,
    n_iter=10,
    cv=5,
    verbose=2,
    n_jobs=-1,
    random_state=42,
    scoring='accuracy',
)
rs.fit(x_train,y_train)


# In[ ]:


rs.best_estimator_


# In[ ]:


lr=LogisticRegression(C=10,  solver='newton-cg')
lr.fit(x_train,y_train)
lr_pred=lr.predict(x_val)
sns.heatmap(confusion_matrix(y_val,lr_pred),fmt="g",annot=True,cmap='Greens')


# # *Random Forest*

# In[ ]:


rf=RandomForestClassifier()
rf.fit(x_train,y_train)
cv=cross_val_score(rf,x_train,y_train,cv=5)


# In[ ]:


param_rf = {
    'n_estimators': [100, 200, 300,500,700,900],  # Number of trees in the forest
    'max_depth': [None, 5, 10, 20],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 7,10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider at each split
    'bootstrap': [True, False],  # Whether to use bootstrapping (sampling with replacement)
    'criterion': ['gini', 'entropy'] , # Splitting criterion
    'class_weight':['balanced',None],
    'random_state':[42]
}


# In[ ]:


rs=RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_rf,
    n_iter=50,
    cv=5,
    verbose=2,
    n_jobs=-1,
    random_state=42,
    scoring='accuracy',
)
rs.fit(x_train,y_train)


# In[ ]:


rs.best_estimator_


# In[ ]:


rf=RandomForestClassifier(class_weight='balanced', criterion='entropy',
                       min_samples_leaf=2, min_samples_split=7,
                       n_estimators=500, random_state=42)
rf.fit(x_train,y_train)
rf_pred=rf.predict(x_val)
sns.heatmap(confusion_matrix(y_val,rf_pred),fmt="g",annot=True,cmap='Greens')


# # *XGB*

# In[ ]:


xgb=XGBClassifier()
xgb.fit(x_train,y_train)
cv=cross_val_score(xgb,x_train,y_train,cv=5)


# In[ ]:


xgb.score(x_val,y_val)


# In[ ]:


param_xgb = {
    "n_estimators":[50,100,150],
             "random_state":[0,42,50],
             "learning_rate":[0.1,0.3,0.5,1.0],
}

gs = GridSearchCV(estimator=xgb, param_grid=param_xgb, cv=3, n_jobs=-1)

gs.fit(x_train, y_train)
best_param= gs.best_params_


# In[ ]:


xgb=XGBClassifier(**best_param)
xgb.fit(x_train,y_train)
xgb_pred=xgb.predict(x_val)
sns.heatmap(confusion_matrix(y_val,xgb_pred),fmt="g",annot=True,cmap='Greens')


# # *CatBoost*

# In[ ]:


#get_ipython().system('pip install catboost')


# In[ ]:


from catboost import CatBoostClassifier
cat=CatBoostClassifier()
cat.fit(x_train,y_train)
cv=cross_val_score(cat,x_train,y_train,cv=5)


# In[ ]:


cat_pred=cat.predict(x_val)
sns.heatmap(confusion_matrix(y_val,cat_pred),annot=True,fmt="g",cmap='Greens')


# In[ ]:


cat.score(x_val,y_val)


# In[ ]:


# param_cat={
#     'iterations':[500,600,700,900,1000],
#     'learning_rate':[0.03,0.05,0.07,0.1,0.15,0.2],
#     'depth':[5,6,7,8],
#     'l2_leaf_reg':[1,3,5,7,9],
#     'l2_leaf_reg':[3,5,7],
#     'random_seed':[42],
#     'bagging_temperature':[0.5,1],
#     'eval_metric':['AUC','Accuracy','Logloss'],
# }


# In[ ]:


# rs=RandomizedSearchCV(
#     estimator=cat,
#     param_distributions=param_cat,
#     n_iter=50,
#     cv=5,
#     verbose=0,
#     n_jobs=-1,
#     random_state=42,
#     scoring='accuracy',
# )
# rs.fit(x_train,y_train)


# In[ ]:


# rs.best_params_


# In[ ]:


# cat=CatBoostClassifier(
#     iterations=700,
#     learning_rate=0.03,
#     depth=8,
#     l2_leaf_reg=5,
#     random_seed=42,
#     bagging_temperature=1,
#     eval_metric='Accuracy',
# )
# cat.fit(x_train,y_train)
# cat_pred=cat.predict(x_val)
# print(accuracy_score(y_val,cat_pred))
# sns.heatmap(confusion_matrix(y_val,cat_pred),annot=True,fmt="g",cmap='Greens')
# print(classification_report(y_val,cat_pred))


# In[ ]:


prediction=cat.predict(x_test)
sub=pd.DataFrame({'PassengerId':test_df['PassengerId'],'Transported':prediction})
sub


# In[ ]:


sub['Transported'].replace([1,0],['True','False'],inplace=True)


# In[ ]:


sub.shape


# In[ ]:


sub


# In[ ]:


sub.to_csv('titanic_submission.csv',index=False)


# In[ ]:




