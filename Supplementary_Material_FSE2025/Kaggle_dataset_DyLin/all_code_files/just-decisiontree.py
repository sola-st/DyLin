#!/usr/bin/env python
# coding: utf-8

# # **Spaceship-titanic competition**

# ## **Import modules**

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np


# ## **Check dataset**

# In[ ]:


df_train = pd.read_csv("/kaggle/input/spaceship-titanic/train.csv", index_col=0)
df_test = pd.read_csv("/kaggle/input/spaceship-titanic/test.csv", index_col=0)

df_train.head()


# ### **Transform and clear data**

# In[ ]:


df_train['Transported'] = df_train['Transported'].astype(int)
df_train.head()


# In[ ]:


df_train.isnull().sum()


# In[ ]:


df_train.shape


# So, we see that Nan values not so much. We can just fill them with median value for numeric and moda value for categorical

# In[ ]:


df_train['HomePlanet'].value_counts()


# In[ ]:


df_train['HomePlanet'] = df_train['HomePlanet'].fillna('Earth')


# In[ ]:


df_train['CryoSleep'].value_counts()


# In[ ]:


df_train['CryoSleep'] = df_train['CryoSleep'].fillna(0)
df_train['CryoSleep'] = df_train['CryoSleep'].astype(int)


# In[ ]:


df_train['VIP'].value_counts()


# In[ ]:


df_train['VIP'] = df_train['VIP'].fillna(0)
df_train['VIP']= df_train['VIP'].astype(int)


# In[ ]:


df_train['Destination'].value_counts()


# In[ ]:


df_train['Destination'] = df_train['Destination'].fillna('TRAPPIST-1e')


# In[ ]:


df_train['Cabin'].nunique()


# **Features like this we just delete**
# 
# They can't give us some useful information

# In[ ]:


df_train.drop('Cabin', axis=1, inplace=True) # Dont forget about 'inplace'. This change our dataset
df_train.drop('Name', axis=1, inplace=True) # Name as well as Cabin


# ## **Go on fill numrecic feature**

# In[ ]:


df_train['Age'] = df_train['Age'].fillna(np.round(df_train['Age'].mean()))
df_train['RoomService'] = df_train['RoomService'].fillna(np.round(df_train['RoomService'].mean()))
df_train['FoodCourt'] = df_train['FoodCourt'].fillna(np.round(df_train['FoodCourt'].mean()))
df_train['ShoppingMall'] = df_train['ShoppingMall'].fillna(np.round(df_train['ShoppingMall'].mean()))
df_train['Spa'] = df_train['Spa'].fillna(np.round(df_train['Spa'].mean()))
df_train['VRDeck'] = df_train['VRDeck'].fillna(np.round(df_train['VRDeck'].mean()))


# In[ ]:


df_train.isnull().sum()


# # **Let's do some plots**

# In[ ]:


sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10,8)
sns.countplot(data=df_train, x='VIP', hue='Transported')
plt.title("VIP counts");


# In[ ]:


df_train.hist();


# In[ ]:


sns.countplot(data=df_train, x='HomePlanet', hue='Transported')
plt.title("Home planet - chance to survive");


# ### **Transform string feature to numeric**

# In[ ]:


def string_to_numeric(df: pd.DataFrame, name: str):
    temp = {}
    feature = df[name].unique()
    for num in range(len(df[name].unique())):
        temp[feature[num]] = num
    df[name] = df[name].map(temp)

string_to_numeric(df_train, 'HomePlanet')
string_to_numeric(df_train, 'Destination')

df_train.head()


# In[ ]:


sns.heatmap(df_train.corr());


# ### **Make T-SNE visualisation of this dataset**

# In[ ]:


from sklearn.manifold import TSNE

tsne = TSNE(random_state=7)
X_repres = tsne.fit_transform(df_train)


# In[ ]:


plt.scatter(X_repres[df_train['Transported'] == 1, 0], X_repres[df_train['Transported'] == 1, 1], c='red', label='Transpoted')
plt.scatter(X_repres[df_train['Transported'] == 0, 0], X_repres[df_train['Transported'] == 0, 1], c='blue', label='Not Transpoerted', alpha=0.3)

plt.legend()
plt.title("T-SNE visualisation");


# ## **Now we're ready to make predictions**

# In[ ]:


from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(df_train.drop('Transported', axis=1), df_train['Transported'], test_size=0.3, shuffle=True)


# In[ ]:


tree = DecisionTreeClassifier(random_state=7)

tree.fit(x_train, y_train)


# In[ ]:


pred_test = tree.predict(x_test)

accuracy_score(y_test, pred_test)


# **Without tunning we have 0.73. Need to add accuracy**

# ## Try to improve our model

# In[ ]:


params = {'max_depth': np.arange(8, 15), 'min_samples_leaf': np.arange(8, 30)}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

best_tree = GridSearchCV(estimator=tree, param_grid=params, cv=skf, n_jobs=-1, verbose=True)


# In[ ]:


best_tree.fit(x_train, y_train)


# In[ ]:


best_tree.best_params_


# In[ ]:


tree = best_tree.best_estimator_


# In[ ]:


pred_test = tree.predict(x_test)

accuracy_score(y_test, pred_test)


# ### **We achive accuracy 0.78**
# 
# Let's transform data_test and make prediction

# In[ ]:


df_test.head()


# In[ ]:


df_test.isnull().sum()


# In[ ]:


df_test['HomePlanet'] = df_test['HomePlanet'].fillna('Earth')

df_test['CryoSleep'] = df_test['CryoSleep'].fillna(0)
df_test['CryoSleep'] = df_test['CryoSleep'].astype(int)

df_test['VIP'] = df_test['VIP'].fillna(0)
df_test['VIP']= df_test['VIP'].astype(int)

df_test['Destination'] = df_test['Destination'].fillna('TRAPPIST-1e')

df_test.drop('Cabin', axis=1, inplace=True) # Dont forget about 'inplace'. This change our dataset
df_test.drop('Name', axis=1, inplace=True) # Name as well as Cabin

df_test['Age'] = df_test['Age'].fillna(np.round(df_train['Age'].mean()))
df_test['RoomService'] = df_test['RoomService'].fillna(np.round(df_train['RoomService'].mean()))
df_test['FoodCourt'] = df_test['FoodCourt'].fillna(np.round(df_train['FoodCourt'].mean()))
df_test['ShoppingMall'] = df_test['ShoppingMall'].fillna(np.round(df_train['ShoppingMall'].mean()))
df_test['Spa'] = df_test['Spa'].fillna(np.round(df_train['Spa'].mean()))
df_test['VRDeck'] = df_test['VRDeck'].fillna(np.round(df_train['VRDeck'].mean()))


# In[ ]:


string_to_numeric(df_test, 'HomePlanet')
string_to_numeric(df_test, 'Destination')


# In[ ]:


df_test.isnull().sum()


# ## **Make csv file to submit our prediction**

# In[ ]:


submission = pd.read_csv('/kaggle/input/spaceship-titanic/sample_submission.csv')


# In[ ]:


submission['Transported'] = tree.predict(df_test)
submission['Transported'] = submission['Transported'].astype(bool)
submission.head()


# In[ ]:


submission.to_csv('submission.csv', index=False)

