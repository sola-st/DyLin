#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import warnings
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,cross_val_score
warnings.filterwarnings(action='ignore')


# # Reading the dataset in a dataframe

# In[ ]:


df=pd.read_csv('/kaggle/input/playground-series-s4e8/train.csv')
df.drop('id',axis=1,inplace=True)


# In[ ]:


df.head()


# # Finding basic insights

# In[ ]:


df.describe()


# In[ ]:


df['class']=df['class'].replace(['e','p'],value=[1,0])


# # Performing EDA

# In[ ]:


fig,ax = plt.subplots(1,3,figsize=(15,4))
sns.boxplot(x=df['cap-diameter'],ax=ax[0])
sns.boxplot(x=df['stem-height'],ax=ax[1])
sns.boxplot(x=df['stem-width'],ax=ax[2])
plt.tight_layout()
plt.show()


# In[ ]:


fig,ax = plt.subplots(1,3,figsize=(15,4))
sns.histplot(df['cap-diameter'],ax=ax[0])
sns.histplot(df['stem-height'],ax=ax[1])
sns.histplot(df['stem-width'],ax=ax[2])
plt.tight_layout()
plt.show()


# # Creating classes for data wrangling

# In[ ]:


class drop_cols(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        x_cols_drop=['gill-spacing','stem-root','stem-surface','veil-type','veil-color','spore-print-color']
        X=X.drop(x_cols_drop,axis=1)
        return X
class data_cleaning(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        for f in X.columns:
            if X[f].dtype == 'float64':
                X[f]=X[f].fillna(value=X[f].median())
            else:
                X[f]=X[f].fillna(value=X[f].mode()[0])
        return X
class Outlier_rem(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        numcol=['cap-diameter','stem-height','stem-width']
        for col in numcol:
            q1=X[col].quantile(0.25)
            q3=X[col].quantile(0.75)
            iqr=q3-q1
            lwr=q1-(1.5*iqr)
            upr=q3+(1.5*iqr)
            X[col]=X[col].apply(lambda x:lwr if x < lwr else x)
            X[col]=X[col].apply(lambda x:upr if x > upr else x)
        return X
class encoding(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        x_cols_to_encode=['cap-shape','cap-surface','cap-color','does-bruise-or-bleed','gill-attachment','gill-color','stem-color','has-ring','ring-type','habitat','season']
        le=LabelEncoder()
        for col in x_cols_to_encode:
            X[col]=le.fit_transform(X[col])
        return X
class Scaling(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        sc=StandardScaler()
        if 'class' in X.columns:
            for col in X.columns[1:]:
                X[col]=sc.fit_transform(X[[col]])
        else:
            for col in X.columns:
                X[col]=sc.fit_transform(X[[col]])
        return X


# # Creating pipeline to run each step in preprocessing

# In[ ]:


pipeline=Pipeline([
    ('drop',drop_cols()),
    ('data_cleanup',data_cleaning()),
    ('Outlier',Outlier_rem()),
    ('encoding',encoding()),
    ('scaling',Scaling())
])
df=pipeline.fit_transform(df)


# # Finding correlation matrix

# In[ ]:


plt.figure(figsize=(15,7))
cor=df.corr()
mask=np.triu(cor)
sns.heatmap(cor,mask=mask,linewidth=0.1,annot=True,cmap='coolwarm',fmt='.2f')


# # Separating train data and target variable

# In[ ]:


X=df.drop('class',axis=1)
y=df['class']


# In[ ]:


X.head()


# In[ ]:


X.isna().sum()


# In[ ]:


y.head()


# # Splitting data into train and test set

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)


# # Creating Random Forest classifier

# In[ ]:


model=RandomForestClassifier(n_estimators=100,max_depth=30,min_samples_split=5,random_state=42,n_jobs=-1)
model.fit(X_train,y_train)


# # Making predictions and finding the MCC score

# In[ ]:


from sklearn.metrics import confusion_matrix,matthews_corrcoef
y_pred= model.predict(X_test)


# # Reading test data

# In[ ]:


test_df=pd.read_csv('/kaggle/input/playground-series-s4e8/test.csv')
test_df.head()


# In[ ]:


ids=test_df['id']
test_df=test_df.drop('id',axis=1)


# # Running the test data through data preprocessing steps

# In[ ]:


pipeline_t=Pipeline([
    ('drop',drop_cols()),
    ('data_cleanup',data_cleaning()),
    ('encoding',encoding()),
    ('scaling',Scaling())
])
test_df=pipeline_t.fit_transform(test_df)


# # Using the model to make predictions

# In[ ]:


y_hats=model.predict(test_df)
y_hats=(y_hats > 0.5).astype(int)


# # Converting predictions to labels

# In[ ]:


y_hats=['p' if pred == 0 else 'e' for pred in y_hats]


# In[ ]:


y_hats[:10]


# In[ ]:


predict=pd.DataFrame({'id':ids,'class':y_hats})


# In[ ]:


predict.head()


# # Creating csv file with final results

# In[ ]:


predict.to_csv('submission.csv',index=False)

