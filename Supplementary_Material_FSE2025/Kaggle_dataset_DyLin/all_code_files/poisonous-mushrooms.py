#!/usr/bin/env python
# coding: utf-8

# ### **Libraries**

# In[ ]:


#get_ipython().system('pip install scikit-optimize')
#get_ipython().system('pip install catboost')
#get_ipython().system('pip install prince')


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import copy
# import prince


# 
# ## Data

# In[ ]:


pd.set_option('display.max_columns', None)


# In[ ]:


# df=pd.read_csv('/content/sample_submission.csv')
tr=pd.read_csv("/kaggle/input/playground-series-s4e8/train.csv")


# In[ ]:


tr.shape


# In[ ]:


tr.head()


# In[ ]:


tr.info()


# In[ ]:


df_train = copy.deepcopy(tr)


# # **Category**

# In[ ]:


def subclass_details (df):
  df=df.select_dtypes(include='category')
  for col in df.columns:
    print('Number of unique classes:',df[col].nunique())
    print(df[col].value_counts().head(10))
    print('############################')


# In[ ]:


def remove_att(df,threshold=200):
  cat_coln=df.select_dtypes(include='object')
  for col in cat_coln:
    attrib_drop=[]
    for att , count in df[col].value_counts().items() :
      if count <threshold:
        attrib_drop.append(att)
    mask = df[col].isin(attrib_drop)
    df.loc[mask,col] = 'UNK'
  return df


# In[ ]:


def convert_cate (df):
  for clas in df.select_dtypes(include='object'):
    df[clas] =   df[clas].astype('category')
  return df


# In[ ]:


df_train = remove_att(df_train)
df_train = convert_cate(df_train)
subclass_details(df_train)


# In[ ]:


df_train.info()


# # **Missing Value**

# In[ ]:


def plot_missing_feature(df):
  null_df=(df.isna().sum()*100/df.shape[0]).sort_values(ascending=False)
  sns.barplot(x=null_df.index,y=null_df.values,palette='plasma')
  plt.xticks(rotation=90)
  plt.xlabel('Feature')
  plt.ylabel('Percent(%)')
  plt.title('Missing Values')
  plt.show()


# In[ ]:


def missing_feature (df):
  null_df=(df.isna().sum()*100/df.shape[0]).sort_values(ascending=False)
  return null_df


# In[ ]:


null_df_train = missing_feature(df_train)
null_df_train


# In[ ]:


plot_missing_feature(df_train)


# In[ ]:


def columns_drop(df):
  column_drop=[]
  null_df=missing_feature(df)
  for col,val in null_df.items():
    if val >4:
      column_drop.append(col)
  return column_drop


# In[ ]:


column_drop_train = columns_drop(df_train)


# In[ ]:


df_train.drop(column_drop_train,axis=1,inplace=True)
df_train.drop('id',axis=1,inplace=True)


# In[ ]:


df_train.isna().sum()


# In[ ]:


df_train.plot(kind='hist',subplots=True,sharex=True,figsize=(15,15),bins=100)


# # **Splitting Data**

# In[ ]:


x=df_train.drop('class',axis=1)
y=df_train['class']
y=np.array([0 if i =='e' else 1 for i in y])
y.reshape(-1,1)


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=.7,stratify=y)


# # **Pipeline**

# In[ ]:


num_data_train_columns = x.select_dtypes(include='number').columns
cat_data_train_columns = x.select_dtypes(include='category').columns
cat_data_train_columns


# In[ ]:


num_pipe=Pipeline (steps=[
    ('imputer',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler())
])
cat_pipe=Pipeline (steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('encoder',OneHotEncoder(sparse=False, handle_unknown='ignore'))
    # ('encoder',OrdinalEncoder())

])


# In[ ]:


df_preprocessing=ColumnTransformer(
     transformers=[
        ('num', num_pipe, num_data_train_columns),
        ('cat', cat_pipe, cat_data_train_columns)
    ]
)


# In[ ]:


final_pipe = Pipeline(steps=[
    ('preprocessor',df_preprocessing),
    ('PCA',PCA(n_components=.95)),
#     ('MCA',prince.MCA( n_components=2,  n_iter=3,       check_input=True, engine='auto',   random_state=42))

])


# In[ ]:


x_train=final_pipe.fit_transform(x_train)
x_test=final_pipe.transform(x_test)


# # **Modeling**

# In[ ]:


def Bayesian_Optimization (model,search_space):
  bayes = BayesSearchCV(model,
                        search_space,
                         n_iter= 10,
                        n_jobs=-1,
                        scoring='accuracy',
                        random_state=42)
  return bayes


# In[ ]:


xgb_space = {
    'n_estimators': Integer(50, 150),
    'max_depth': Integer(2, 8),
    'learning_rate': Real(0.01, .4, 'log-uniform'),
    'subsample': Real(0.5, 1.0, 'uniform'),
    'colsample_bytree': Real(0.5, 1.0, 'uniform'),
    'alpha':Real(.1,.5,'uniform'),
    'min_child_weight':Integer(5,10)
}
lgb_space={'num_leaves': Integer(24, 45),
          'feature_fraction': Real(0.1, 0.9),
          'bagging_fraction': Real(0.5, 1),
          'max_depth':Integer (5, 9),
          'lambda_l1':Real (0, 5),
          'lambda_l2':Real (0, 3),
          'min_split_gain':Real (0.001, 0.1),
          'min_child_weight': Integer(5, 60)
}
cat_space={
    'iterations': Integer(10, 100),
    'depth': Integer(1, 8),
    'learning_rate': Real(0.01, 1.0, 'log-uniform'),
    'l2_leaf_reg': Real(1e-3, 10, 'log-uniform'),
    'border_count': Integer(32, 128)
}


# In[ ]:


clsss=len(y[y==0]) / len(y[y==1])
clsss


# In[ ]:


xgb = XGBClassifier(random_state=42,scale_pos_weight =clsss)
# lgb = LGBMClassifier(random_state=42)
cat = CatBoostClassifier(random_state=42)
xgb_optimzied = Bayesian_Optimization(xgb,xgb_space)
# lgb_optimzied = Bayesian_Optimization(lgb,lgb_space)
cat_optimzied = Bayesian_Optimization(cat,cat_space)


# In[ ]:


xgb_optimzied.fit(x_train,y_train)
# lgb_optimzied.fit(x_train,y_train)
cat_optimzied.fit(x_train,y_train)


# In[ ]:


xgb_optimzied.best_params_


# In[ ]:


xgb=xgb_optimzied.best_estimator_
# lgb=lgb_optimzied.best_estimator_
cat=cat_optimzied.best_estimator_


# In[ ]:


y_pred_xgb=xgb.predict(x_test)
# y_pred_lgm=lgb.predict(x_test)
y_pred_cat=cat.predict(x_test)
# print('LGM --> ',accuracy_score(y_test,y_pred_lgm))


# In[ ]:


voting_clf=VotingClassifier(estimators=[
        ('xgb', xgb),
        ('catboost', cat)
#         ,('lightbgm',lbm)
    ],
    voting='soft'
)
voting_clf.fit(x_train,y_train)
y_pred_voting=voting_clf.predict(x_test)


# In[ ]:


import gc
gc.collect()


# > ***Submession***

# In[ ]:


ts = pd.read_csv('/kaggle/input/playground-series-s4e8/test.csv')


# In[ ]:


df_test = copy.deepcopy(ts)
df_test.drop('id',axis=1,inplace=True)


# In[ ]:


columns_drop(ts)


# In[ ]:


df_test=final_pipe.transform(df_test)


# In[ ]:


final_pred = voting_clf.predict(df_test)


# In[ ]:


final_pred_trans=['e' if i==0 else 'p' for i in final_pred]


# In[ ]:


submession= pd.DataFrame({'id':ts['id'].values,
                          'class':final_pred_trans
                         }
                        )


# In[ ]:


submession.to_csv('submession.csv',index=False)


# In[ ]:




