#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "/kaggle/input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Important Notes
# * Only 4 columns contain int/float values including ID
# 
# * a lot of null values

# # Loading Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler,OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedShuffleSplit
from xgboost import XGBClassifier
import optuna
from sklearn.metrics import roc_auc_score, roc_curve, auc, classification_report, confusion_matrix, accuracy_score,matthews_corrcoef
import scipy
import warnings
warnings.filterwarnings('ignore')


# # Loading DataSet

# In[ ]:


#Import DataSet

train=pd.read_csv("/kaggle/input/playground-series-s4e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s4e8/test.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s4e8/sample_submission.csv")


# In[ ]:


sub.head()


# In[ ]:



train.head()


# In[ ]:


train.columns


# In[ ]:


train.info()


# ## Checking Null Values

# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# ## Basic Info about Target

# In[ ]:


train["class"].value_counts()


# In[ ]:


sns.countplot(x="class",data = train)


# ## Correlation Metrics

# In[ ]:


plt.figure(figsize=(16,12))
temp = train.drop("id", axis = 1)
temp= temp.apply(lambda x : pd.factorize(x)[0] if x.dtype=="object" else x)
sns.heatmap(temp.corr(),annot= True, cmap="coolwarm",annot_kws={"size":8})
plt.show()


# ## Checking unique categories in every feature

# For 1 column

# In[ ]:


count=train["stem-root"].value_counts()
plt.figure(figsize=(12,4))
sns.barplot(x=count.index,y=count.values)
    


# for all columns
# 

# In[ ]:


def count_cat(df):
    cat_col=df.select_dtypes(include=['object']).columns
    uni_col={col: df[col].value_counts() for col in cat_col}
    plt.figure(figsize=(15,len(cat_col)*5))
    for i ,(col,count) in enumerate(uni_col.items(),1):
        plt.subplot(len(cat_col),1,i)
        sns.barplot(x=count.index,y=count.values)
        plt.title(f"Count of unique categories in column '{col}'")
    plt.show()
count_cat(train)


# ## Checking Percentage of Null values

# In[ ]:


def check_null(df):
    percentage = (df.isnull().sum()/len(df))*100
    return percentage


    


# NAN Values in Train Data

# In[ ]:


check_null(train)


# NAN values in Test Data

# In[ ]:


check_null(test)


# ## Checking Feature Importance and dropping useless column

# In[ ]:


#using chi square test to check the importance of features 
#chi sq test is used for the categorical features
from scipy.stats import chi2_contingency

alpha =0.05
df = train
for col in df.columns:
    if col=="class":
        continue
    a , b = df[col], df["class"]
    obs = pd.crosstab(a,b)
    
    #now chi 2 test
    res= chi2_contingency(obs)
    if res.pvalue >= alpha:
        print("{} is NOT important: (p = {}) ".format(col,res.pvalue))
    else:
        print("{} is important: (p = {}) ".format(col,res.pvalue))
    


# ## Dropping the not important features

# In[ ]:


train = train.drop(["id","veil-type","veil-color"],axis = 1)
test = test.drop(["id","veil-type","veil-color"],axis = 1)


# In[ ]:


train.columns


# ## Handing NAN values and less frequent Categories

# Checking Numerical Cols

# In[ ]:


numerical_cols = train.select_dtypes(include=np.number)
numerical_cols


# ## Filling the missing values using mode (most frequent values in the dataset)

# In[ ]:


numerical_cols.columns


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


for col in numerical_cols:
    median_val = train[col].mode()[0]
    train[col].fillna(median_val,inplace = True)
    test[col].fillna(median_val,inplace = True)

    


# In[ ]:


train.select_dtypes(include=np.number).isnull().sum()


# In[ ]:


test.select_dtypes(include=np.number).isnull().sum()


# In[ ]:


numeric_feats = ["cap-diameter", "stem-height", "stem-width"]
plt.figure(figsize=(15, 5))

for i , column in enumerate(numeric_feats,1):
    plt.subplot(1,len(numeric_feats),i)
    sns.boxplot(x=train[column])
    plt.title("BoxPlot of {}".format(column))

plt.show()
    


# # Handling Outliers

# ## Detecting Outlier using IQR (Interquartile Range)
# 

# In[ ]:


numeric_feats = ["cap-diameter", "stem-height", "stem-width"]
def detect_outliers_iqr(data, column , r1 , r2 ):
    
    Q1 = data[column].quantile(r1)
    Q3 = data[column].quantile(r2)
    iqr = Q3 - Q1
    lower_bound = Q1 - (1.5 * iqr)
    upper_bound = Q3 + (1.5 * iqr)
    
    return data[(data[column]<lower_bound)| (data[column]>upper_bound)]  


# ## Checking Outliers for (0.25-0.75)

# In[ ]:


for col in numeric_feats:
    
    outliers = detect_outliers_iqr(train,col,0.25,0.75)
    print(f"Number of outliers in {col} : {outliers.shape[0]}")


# ## Checking Outliers for (0.05-0.95)

# In[ ]:


for col in numeric_feats:
    
    outliers = detect_outliers_iqr(train,col,0.05,0.95)
    print(f"Number of outliers in {col} : {outliers.shape[0]}")


# ## Checking Outliers for (0.01-0.99)

# In[ ]:


for col in numeric_feats:
    
    outliers = detect_outliers_iqr(train,col,0.01,0.99)
    print(f"Number of outliers in {col} : {outliers.shape[0]}")


# ## Checking Outliers Using Z-Score

# In[ ]:


from scipy import stats

for col in numeric_feats:
    z_scores = np.abs(stats.zscore(train[col]))
    outliers = train[col][z_scores > 3]
    print(f"Number of outliers in {col} : {outliers.shape[0]}")


# In[ ]:


plt.figure(figsize=(15, 5))

for i , column in enumerate(numeric_feats,1):
    plt.subplot(1,len(numeric_feats),i)
    plt.plot(train[column])
    plt.title("Plot of {}".format(column))

plt.show()


# ## Removing Outliers

# In[ ]:


def cap_outliers(data,col):
    Q1 = data[col].quantile(0.01)
    Q3 =data[col].quantile(0.99)
    iqr = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * iqr
    upper_bound = Q3 + 1.5 * iqr
    
    data[col] = np.where(data[col]<lower_bound,lower_bound,data[col])
    data[col] = np.where(data[col]>upper_bound,upper_bound,data[col])
    
    


# In[ ]:


for column in numeric_feats:
    cap_outliers(train, column)
    cap_outliers(test, column)


# In[ ]:


train.isnull().sum()


# # Cleaning DataSet

# In[ ]:


categorical_cols = train.select_dtypes(include=['object']).columns
categorical_cols


# In[ ]:


def cleaner(df,threshold = 100):
    
    #threshold = 100  #for infrequent categories
    
    cat_features=['cap-shape', 'cap-surface', 'cap-color',
       'does-bruise-or-bleed', 'gill-attachment', 'gill-spacing', 'gill-color',
       'stem-root', 'stem-surface', 'stem-color', 'has-ring',
       'ring-type', 'spore-print-color', 'habitat', 'season']
    
    for feat in cat_features:
        if df[feat].dtype.name == "category":
            # ADD missing and noise , if not present 
            if "missing" not in df[feat].cat.categories:
                df[feat] = df[feat].cat.add_categories("missing")
            if "noise"  not in df[feat].cat.categories:
                df[feat] = df[feat].cat.add_categoies("noise")
        
        else:
            #convert to category and add new Categories
            df[feat] = df[feat].astype("category")
            
            #.cat is attribute that provides methods for working with categorical data
            
            #adding (missing and noise) category to the features
            df[feat] = df[feat].cat.add_categories(["missing","noise"])
        
        #fill missing values with missing
        df[feat] = df[feat].fillna("missing")
        
        # replace infrequent category with noise
        count = df[feat].value_counts(dropna=False)
        infrequent_categories = count[count<threshold].index
        df[feat] = df[feat].apply(lambda x : "noise" if x in infrequent_categories else x)
        
    
    return df


# In[ ]:


train_data = cleaner(train)
test_data = cleaner(test)


# In[ ]:




# Converting features into category Features

# In[ ]:


cat_feats = ["cap-shape", "cap-surface", "cap-color", "does-bruise-or-bleed", "gill-attachment",
             "gill-spacing", "gill-color", "stem-root", "stem-surface", "stem-color",
              "has-ring", "ring-type", "spore-print-color", "habitat", "season"]
for feat in cat_feats:
    train_data[feat] = train_data[feat].astype('category')
for feat in cat_feats:
    test_data[feat] = test_data[feat].astype('category')


# In[ ]:


train_data.info()


# In[ ]:


test_data.info()


# ## Splitting Data

# In[ ]:


y=train_data["class"]
X=train_data.drop(["class"],axis=1)
X.shape,y.shape


# In[ ]:


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=42)


# # Model Training

# 

# ## Using XGBoost with OPTUNA

# In[ ]:


import optuna


# In[ ]:


# # Defining objective function for optuna 

# def objective(trial):
    
#     params = {
#         "n_estimators" : trial.suggest_int("n_estimators",200,5000),
#         "max_depth" : trial.suggest_int("max_depth",10,50),
#         "learning_rate" : trial.suggest_loguniform("learning_rate",0.01,0.3),
#         "sub_sample" : trial.suggest_uniform("sub_sample",0.4,1.0),
#         "colsample_bytree" : trial.suggest_uniform("colsample_bytree",0.4,1.0),
#         "gamma" : trial.suggest_loguniform("gamma",1e-8,1.0),
#         "lambda" : trial.suggest_loguniform("lambda",1e-8,10.0),
#         "alpha" : trial.suggest_loguniform("alpha",1e-8,10.0),
#         "scale_pos_weight" : trial.suggest_uniform('scale_pos_weight', 1.0, 10.0),
#     }
    
#     model = XGBClassifier(**params,use_label_encoder=False,eval_metric="logloss",
#                          enable_categorical=True,tree_method ="hist",device = "cuda",
#                          objective="multi:softmax",num_class=2)
    
#     model.fit(X_train,y_train)
#     y_pred = model.predict(X_test)
#     mcc = matthews_corrcoef(y_test, y_pred)
#     trial.set_user_attr("mcc", mcc)
#     return mcc

# # Callback to print the MCC score for each trial
# def print_mcc_score(study,trial):
#     mcc = trial.user_attrs["mcc"]
#     print(f"Trial {trial.number}: MCC = {mcc}")

# # Optimize hyperparameters with Optuna
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=2, callbacks=[print_mcc_score])

# # Get the best parameters
# best_params = study.best_params
# print(f"Best parameters: {best_params}")


# 1. study = optuna.create_study(direction='maximize')
# 
# optuna.create_study() is a function that creates a new study (i.e., an optimization process) in Optuna.
# direction='maximize' is an argument that specifies the direction of the optimization. In this case, we want to maximize the Matthews correlation coefficient (MCC), so we set the direction to 'maximize'
# 
# 2. study.optimize(objective, n_trials=2, callbacks=[print_mcc_score])
# 
# study.optimize() is a method that starts the optimization process.
# objective is the function that we want to optimize. In this case, it's the objective function we defined earlier, which calculates the MCC value for a given set of hyperparameters.
# n_trials=2 is an argument that specifies the number of trials (i.e., evaluations of the objective function) to perform. In this case, we're doing only 2 trials, which means Optuna will try 2 different sets of hyperparameters.
# callbacks=[print_mcc_score] is an argument that specifies a list of callback functions to call after each trial. In this case, we're using the print_mcc_score function we defined earlier, which prints the MCC value for each trial.

# # Best Parameters

# In[ ]:


best_params=  {'n_estimators': 288, 
               'max_depth': 35, 
               'learning_rate': 0.04964969702056722, 
               'subsample': 0.9976503206175568, 
               'colsample_bytree': 0.451300957115364, 
               'gamma': 0.8965084856869137, 
               'lambda': 0.03608752969236549, 
               'alpha': 6.33187159990702e-05, 
               'scale_pos_weight': 4.575979634910302}


# ## Training Model with Best Parameters

# In[ ]:


model = XGBClassifier(**best_params,enable_categorical=True,tree_method='hist',device= 'cuda',objective='multi:softmax',num_class=2)
model = model.fit(X, y)


# In[ ]:


id_column = sub.pop('id')

# Make predictions on the test data
y_test_pred = model.predict(test_data)
y_test_pred_binary = (y_test_pred > 0.5).astype(int)  

# Create the submission DataFrame
submission_df = pd.DataFrame({
    'id': id_column,
    'class': y_test_pred_binary
})

# Map the binary predictions to 'e' and 'p'
submission_df['class'] = np.where(submission_df['class'] == 1, 'p', 'e')

# Save the submission DataFrame to a CSV file
submission_df.to_csv('submission_xgb65.csv', index=False)


# In[ ]:




