#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


train=pd.read_csv("/kaggle/input/playground-series-s4e8/train.csv")


# In[ ]:


train.shape


# In[ ]:


train.head()


# In[ ]:


(train.isnull().sum()/train.shape[0])*100


# In[ ]:


for i in train:
    print(i,"--"*20,train[i].nunique())


# > ### Droping NULL VALUE COL WITH MORE THAN 40% data Present in it is NULL VALUE 

# In[ ]:


for i in train:
    if (train[i].isnull().sum()/train.shape[0])*100 >= 40:
        train.drop(i,axis=1,inplace=True)


# In[ ]:


train.columns


# In[ ]:


train.info()


# In[ ]:


train.describe()


# In[ ]:


train.hist(bins=40)


# In[ ]:


corr=train.corr(numeric_only=True)


# In[ ]:


corr


# In[ ]:


for i in train.select_dtypes(include=["number"]):
    sns.histplot(data=train,x=i,kde=True)
    plt.show()


# In[ ]:


train.skew(numeric_only=True).round()


# In[ ]:


features=train.drop(["id","class"],axis=1).copy()
labels=train["class"]


# In[ ]:


from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.compose import ColumnTransformer,make_column_transformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,FunctionTransformer,OrdinalEncoder
from sklearn.impute import SimpleImputer


# In[ ]:


features.columns


# In[ ]:


features.info()


# In[ ]:


num=["stem-height","stem-width","cap-diameter"]
cat=['cap-shape', 'cap-surface', 'cap-color','does-bruise-or-bleed', 'gill-attachment', 'gill-color','stem-color', 'has-ring', 'ring-type', 'habitat','season']


# In[ ]:


pipe_num_val=make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)

pipe_cat_val=make_pipeline(
    OneHotEncoder(handle_unknown='ignore')
)


# In[ ]:


inner=ColumnTransformer([
    ("Num_val",pipe_num_val,num),
    ("cat_val",pipe_cat_val,cat),
])


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features,labels, test_size=0.2, random_state=42)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
import xgboost as xgb
from sklearn.svm import SVC


# In[ ]:


pipeline=Pipeline([
    ("Pipeline",inner),
    ("model",SVC())
])
pipeline


# In[ ]:


pipeline.fit(X_train,y_train)


# In[ ]:


y_hat=pipeline.predict(X_test)
y_hat


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, accuracy_score,roc_curve,roc_auc_score


# In[ ]:




# In[ ]:




# In[ ]:


test=pd.read_csv("/kaggle/input/playground-series-s4e8/test.csv")


# In[ ]:


new_df=test.copy()


# In[ ]:


temp=test["id"]


# In[ ]:


test=test.drop("id",axis=1)


# In[ ]:


test.isnull().sum()


# In[ ]:


for i in test:
    if (test[i].isnull().sum()/test.shape[0])*100 >= 40:
        test.drop(i,axis=1,inplace=True)


# In[ ]:


y_hat=test_data=pipeline.predict(test)


# In[ ]:


# submitting file
rfc_prediction = pipeline.predict(test)

output = pd.DataFrame({'id': new_df.id, 'NObeyesdad':rfc_prediction})
output.to_csv('Multi-Class Prediction of Obesity Risk.csv', index=False)


# In[ ]:




