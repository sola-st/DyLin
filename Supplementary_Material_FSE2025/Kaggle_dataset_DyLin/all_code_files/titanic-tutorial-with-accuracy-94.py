#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # Load DataSets

# In[ ]:


y_test = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
y_test.head()


# In[ ]:


y_test.shape


# In[ ]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()


# In[ ]:


test=pd.read_csv('/kaggle/input/titanic/test.csv')
test.head()


# # Drop UnImportant Columns

# In[ ]:


train.drop(['Name','PassengerId'],axis=1,inplace=True)
test.drop(['Name','PassengerId'],axis=1,inplace=True)
y_test.drop(['PassengerId'],axis=1,inplace=True)


# # Information About Data

# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train.describe()


# In[ ]:


test.describe()


# # Remove Nulls From Columns

# In[ ]:


train.isnull().sum().sort_values(ascending=False)


# In[ ]:


# prompt: how percent nulls in columns

(train.isnull().sum()/train.shape[0]*100).sort_values(ascending=False)


# In[ ]:


test.isnull().sum().sort_values(ascending=False)


# In[ ]:


(train.isnull().sum()/train.shape[0]*100).sort_values(ascending=False)


# In[ ]:


train['Age'].fillna(train['Age'].median(), inplace=True)
test['Age'].fillna(test['Age'].median(), inplace=True)


# In[ ]:


train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)


# In[ ]:


test['Fare'].fillna(test['Fare'].median(), inplace=True)


# In[ ]:


train.drop(columns=['Cabin'], inplace=True)
test.drop(columns=['Cabin'], inplace=True)


# # Remove Duplicated from Columns

# In[ ]:


train.duplicated().sum()


# In[ ]:


train.drop_duplicates(inplace=True)


# # Encoding String Data

# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for col in train.columns:
  if train[col].dtype=='object':
    train[col]=le.fit_transform(train[col])


# In[ ]:


for col in test.columns:
  if test[col].dtype=='object':
    test[col]=le.fit_transform(test[col])


# # Data display

# In[ ]:


import seaborn as sns
sns.heatmap(train.corr(),annot=True)


# In[ ]:


sns.heatmap(test.corr(),annot=True)


# In[ ]:


train.hist(figsize=(10,10))


# In[ ]:


test.hist(figsize=(10,10))


# In[ ]:


sns.pairplot(train)


# In[ ]:


sns.pairplot(test)


# # Spliting Data

# In[ ]:


train.shape , test.shape


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


test.shape,y_test.shape


# In[ ]:


x_train = train.iloc[:,1:]
y_train = train.iloc[:,0]
x_test = test.iloc[:,:]
y_test = y_test.iloc[:,:]


# In[ ]:


x_test.shape , y_test.shape


# # Try Models on Data
# 

# In[ ]:


from sklearn.linear_model import  LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from  sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score


# In[ ]:


Models = {
    'LogisticRegression' :LogisticRegression(),
 'DecisionTreeClassifier' : DecisionTreeClassifier(),
 'RandomForestClassifier' : RandomForestClassifier(),
 'GaussianNB':GaussianNB(),
 'KNeighborsClassifier':KNeighborsClassifier()
}


# In[ ]:


ModelName = []
ModelAccuracy = []
for nameModel,model in tqdm(Models.items()):
    model.fit(x_train,y_train)
    ModelName.append(nameModel)
    y_pred = model.predict(x_test)
    ModelAccuracy.append([
        accuracy_score(y_test,y_pred)
        ,precision_score(y_test,y_pred)
        ,recall_score(y_test,y_pred)
        ,f1_score(y_test,y_pred)
    ])


# In[ ]:


Model_accuracy = pd.DataFrame(ModelAccuracy,index=ModelName,columns = ['Accuracy','Precision','Recall','F1 Score'])
Model_accuracy


# In[ ]:


# Plotting
import matplotlib.pyplot as plt
Model_accuracy.plot(kind='bar', figsize=(10, 6))

# Customizing the plot

plt.xlabel('Model')
plt.ylabel('Scores')
plt.title('Model Accuracy Scores')
plt.xticks(rotation=45)  # Rotate model names for better readability
plt.legend(loc='upper right')
plt.tight_layout()  # Adjust layout to fit labels

# Display the plot
plt.show()


# # Try Neural On Data

# In[ ]:


import tensorflow as tf
from tensorflow import keras
model=keras.Sequential([
    keras.layers.Dense(128,input_shape=(8,),activation='relu'),
    keras.layers.Dense(64,activation='relu'),
    keras.layers.Dense(1,activation='sigmoid')
])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=100,batch_size=32,validation_data=(x_test,y_test))


# In[ ]:


pd.DataFrame(model.history.history).plot(figsize=(12, 8))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()

# Evaluate the model
model_evaluate = model.evaluate(x_test, y_test)


# #  **Submission File**

# In[ ]:


predictions = model.predict(x_test)
predicted_labels = (predictions > 0.5).astype(int)
submission = pd.DataFrame({
    'PassengerId': range(892, 1310),  # Adjusted to 417
    'Survived': predicted_labels.flatten()
})

submission.to_csv('submission.csv', index=False)


# In[ ]:


pd.read_csv('/kaggle/working/submission.csv')


# In[ ]:




