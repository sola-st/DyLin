#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import lightgbm as lgb
import gc


# In[ ]:


df_sub=pd.read_csv("/kaggle/input/playground-series-s4e8/sample_submission.csv")
df=pd.read_csv("/kaggle/input/playground-series-s4e8/train.csv")
df_test=pd.read_csv("/kaggle/input/playground-series-s4e8/test.csv")


# In[ ]:


df_train = df.copy()


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


df_test.shape,df_train.shape


# 
# 

# In[ ]:


df_train = df_train.drop(columns=['id'])
df_test = df_test.drop(columns=['id'])


# In[ ]:


df_train.info()


# In[ ]:


categorical_columns = df_train.select_dtypes(include=['object']).columns
unique_values = {col: df_train[col].nunique() for col in categorical_columns}
for col, unique_count in unique_values.items():
    print(f"{col}: {unique_count} ")
    
gc.collect()


# In[ ]:


categorical_columns = df_test.select_dtypes(include=['object']).columns
unique_values = {col: df_test[col].nunique() for col in categorical_columns}
for col, unique_count in unique_values.items():
    print(f"{col}: {unique_count}")
    
gc.collect()


# In[ ]:




# In[ ]:


import missingno as msno
import matplotlib.pyplot as plt

# Bar plot to visualize missing values
msno.bar(df_train)


# In[ ]:


msno.bar(df_test)


# In[ ]:


missing_train = df_train.isna().mean() * 100
missing_test = df_test.isna().mean() * 100




# In[ ]:


import seaborn as sns
missing_values = df_train.isnull().mean() * 100
missing_values = missing_values[missing_values >0]
missing_values = missing_values.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=missing_values.index, y=missing_values.values, palette='viridis')
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Percentage of Missing Values')
plt.title('Missing Values Distribution in df_train')
plt.show()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
import category_encoders as ce

missing_threshold = 0.95

high_missing_columns = df_train.columns[df_train.isnull().mean() > missing_threshold]

df_train = df_train.drop(columns=high_missing_columns)
df_test = df_test.drop(columns=high_missing_columns)
target = 'class'

for column in df_train.columns:
    if df_train[column].isnull().any():      
        if df_train[column].dtype == 'object':
            mode_value = df_train[column].mode()[0]
            df_train[column].fillna(mode_value, inplace=True)
            df_test[column].fillna(mode_value, inplace=True)     
        else:
            median_value = df_train[column].median()
            df_train[column].fillna(median_value, inplace=True)
            df_test[column].fillna(median_value, inplace=True)


# In[ ]:


cols_to_drop_train = missing_train[missing_train > 95].index
cols_to_drop_test = missing_test[missing_test > 95].index

df_train = df_train.drop(columns=cols_to_drop_train)
df_test = df_test.drop(columns=cols_to_drop_test)
gc.collect()


# In[ ]:


from sklearn.impute import KNNImputer
import pandas as pd

def knn_impute(df, n_neighbors=5):   
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include='object').columns:
        df_encoded[col] = df_encoded[col].astype('category').cat.codes
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed = pd.DataFrame(knn_imputer.fit_transform(df_encoded), columns=df_encoded.columns)
    for col in df.select_dtypes(include='object').columns:
        df_imputed[col] = df_imputed[col].round().astype(int).map(
            dict(enumerate(df[col].astype('category').cat.categories)))
    return df_imputed


# In[ ]:


df_train_imputed = knn_impute(df_train, n_neighbors=5)
df_test_imputed = knn_impute(df_test, n_neighbors=5)


# In[ ]:


cat_cols_train = df_train_imputed.select_dtypes(include=['object']).columns
cat_cols_train = cat_cols_train[cat_cols_train != 'class']
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

df_train_imputed[cat_cols_train] = ordinal_encoder.fit_transform(df_train_imputed[cat_cols_train].astype(str))
df_test_imputed[cat_cols_train] = ordinal_encoder.transform(df_test_imputed[cat_cols_train].astype(str))


# In[ ]:


df_train_imputed.head()


# In[ ]:


df_test_imputed.head()


# In[ ]:


df_train = df_train_imputed
df_test = df_test_imputed


# In[ ]:


df_test.head()


# In[ ]:


le = LabelEncoder()
df_train['class'] = le.fit_transform(df_train['class'])


# In[ ]:


y = df_train['class'] 
X = df_train.drop(['class'],axis=1)


# In[ ]:


train_X, test_X, train_y, test_y = train_test_split(X, y,test_size = 0.2, random_state =42,stratify=y)


# In[ ]:


from sklearn.metrics import matthews_corrcoef


# In[ ]:


def mcc_metric(y_pred, dmatrix):
    y_true = dmatrix.get_label()
    y_pred = (y_pred > 0.5).astype(int) 
    mcc = matthews_corrcoef(y_true, y_pred)
    return 'mcc', mcc


# In[ ]:


from sklearn.metrics import matthews_corrcoef
from xgboost import XGBClassifier

model = XGBClassifier(                    
    alpha=0.1,                   
    subsample=0.8,     
    colsample_bytree=0.6,  
    objective='binary:logistic',
    max_depth=14,             
    min_child_weight=7,         
    gamma=1e-6,                
    #random_state=42,                 
    n_estimators=100
    )

XGB = model.fit(
    train_X, 
    train_y, 
    eval_set=[(test_X, test_y)],
    eval_metric=mcc_metric)


# In[ ]:


y_pred = XGB.predict(test_X)


# In[ ]:


import lime
import lime.lime_tabular


# In[ ]:


redict_fn_xgb = lambda x: XGB.predict_proba(x).astype(float)
X = train_X.values
explainer = lime.lime_tabular.LimeTabularExplainer(X,feature_names = train_X.columns,class_names=['Poisnous','edible'],kernel_width=5)


# In[ ]:


df_test.head(4)


# In[ ]:


df_test.loc[[3]]


# In[ ]:


test_X


# In[ ]:


score = matthews_corrcoef(test_y, y_pred)


# In[ ]:


test_pred_prob = XGB.predict(df_test)


# In[ ]:


test_pred_prob


# In[ ]:


#test_pred_binary = (test_pred_prob > 0.5).astype(int)
test_pred_class = le.inverse_transform(test_pred_prob)


# In[ ]:


df_sub['class']= test_pred_class


# In[ ]:


df_sub.to_csv('submission.csv', index = False)
pd.read_csv('submission.csv')

