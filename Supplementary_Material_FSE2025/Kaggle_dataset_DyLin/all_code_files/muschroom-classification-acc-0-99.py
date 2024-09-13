#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['font.size'] = 14

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
# hyperparameter tunning
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
# model evaluation
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import matthews_corrcoef

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier, StackingClassifier


# ## <a id="1"></a>
# <div style="
#            border-radius:50px;
#            background-color:#7ca4cd;
#            font-size:200%;
#            font-family:Arial;
#            letter-spacing:0.10px">
# <p style="padding: 10px;
#           color:white;
#           text-align:center;">1: Exploratory Data Analysis
# </p>
# </div>

# ## <a id="1.1"></a>
# ## 1.1 -Basic info

# In[ ]:


df_train = pd.read_csv("/kaggle/input/playground-series-s4e8/train.csv", index_col='id')
df_exam  = pd.read_csv("/kaggle/input/playground-series-s4e8/test.csv", index_col='id')
# df_sample  = pd.read_csv("/kaggle/input/playground-series-s4e8/sample_submission.csv", index_col='id')


# In[ ]:


df_train.head(2)


# In[ ]:


df_exam.head(2)


# In[ ]:


# check missing values


# In[ ]:


# check data type


# ## <a id="1.2"></a>
# ## 1.2 -Visualize missing values

# In[ ]:


def get_percentage_nan_values(data, title,thresh=20, color='black', edgecolor='black', width=15, height=3):
    """
    visualize the percentage of missing values in each columns
    SOURCE: https://www.kaggle.com/amiiiney/price-prediction-regularization-stacking
    """
    
    plt.figure(figsize=(width,height))
    percentage=(data.isnull().mean())*100
    percentage.sort_values(ascending=False).plot.bar(color=color, edgecolor=edgecolor)
    plt.axhline(y=thresh, color='r', linestyle='-')
    plt.title('Missing values percentage per column in {}'.format(title), fontsize=20, weight='bold' )
    plt.text(len(data.isnull().sum()/len(data))/1.7, thresh+12.5, f'Columns with more than {thresh}% missing values', fontsize=12, color='crimson',
         ha='left' ,va='top')
    plt.text(len(data.isnull().sum()/len(data))/1.7, thresh - 5, f'Columns with less than {thresh} missing values', fontsize=12, color='green',
         ha='left' ,va='top')
    plt.xlabel('Columns', size=15, weight='bold')
    plt.ylabel('Missing values percentage')
    plt.yticks(weight ='bold')
    
    return plt.show()


# In[ ]:


# show percentage and distribution missing values per columns
get_percentage_nan_values(df_train ,'df_train', 20, color=sns.color_palette('Reds',15))
get_percentage_nan_values(df_exam,'df_exam', 20, color=sns.color_palette('Blues',15))


# ## <a id="1.3"></a>
# ## 1.3 -Check and drop duplicates

# In[ ]:



df_train.drop_duplicates(inplace=True)


# ## <a id="1.4"></a>
# ## 1.4 -Check check the uniqueness of categorical variables

# In[ ]:


# create df with all data
df_train2 = df_train.drop('class', axis=1)
df_all_data = pd.concat([df_train2,df_exam])
# select only categorical values
df_all_data_cat = df_all_data.select_dtypes(include=['object'])


# In[ ]:


def delete_rows_not_in_list(df, column_name):
    # Define the range of allowed values
    allowed_range = set(chr(i) for i in range(ord('a'), ord('z') + 1))

    # Create a mask to filter rows
    mask = df[column_name].isin(allowed_range) | df[column_name].isna()
    
    # Filter the DataFrame based on the mask
    filtered_df = df[mask]
    
    return filtered_df

def clean_cat(df_train,df_exam,col):
    shape_train_ini = df_train.shape[0]
    shape_exam_ini = df_exam.shape[0]
    df_train = delete_rows_not_in_list(df_train, col)
    df_exam  = delete_rows_not_in_list(df_exam, col)
    shape_train_final = df_train.shape[0]
    shape_exam_final  = df_exam.shape[0]
    print('nb of line delete in df_train = ',shape_train_ini-shape_train_final)
    print('nb of line delete in df_exam  = ',shape_exam_ini-shape_exam_final)
    print(df_train[col].unique())
    print(df_exam[col].unique())


# In[ ]:


df_all_data_cat['cap-shape'].unique()


# In[ ]:


def clean_var1(df,var):
    df[var] = df[var].replace(to_replace='5 f',value="f")
    df[var] = df[var].replace(to_replace='7 x',value="x")
    df[var] = df[var].replace(to_replace='6 x',value="x")
    df[var] = df[var].replace(to_replace='is s',value="s")
    df[var] = df[var].replace(to_replace='is f',value="f")
    df[var] = df[var].replace(to_replace='is p',value="p")
    df[var] = df[var].replace(to_replace='3 x',value="x")
    
clean_var1(df_train,'cap-shape')
clean_var1(df_exam,'cap-shape')
clean_cat(df_train,df_exam,'cap-shape')


# In[ ]:


df_all_data_cat['cap-surface'].unique()


# In[ ]:


def clean_var2(df,var):
    df[var] = df[var].replace(to_replace='does l',value="l")
    df[var] = df[var].replace(to_replace='is h',value="h")
    df[var] = df[var].replace(to_replace='does h',value="h")
    df[var] = df[var].replace(to_replace='does t',value="t")
    df[var] = df[var].replace(to_replace='has h',value="h")
    df[var] = df[var].replace(to_replace='does None',value="None")
    df[var] = df[var].replace(to_replace='is None',value="None")
    df[var] = df[var].replace(to_replace='is y',value="y")
    df[var] = df[var].replace(to_replace='is k',value="k")

clean_var2(df_train,'cap-surface')
clean_var2(df_exam,'cap-surface')
clean_cat(df_train,df_exam,'cap-surface')


# In[ ]:


df_all_data['cap-color'].unique()


# In[ ]:


def clean_var3(df,var):
    df[var] = df[var].replace(to_replace='does n',value="n")
    df[var] = df[var].replace(to_replace='4. n',value="n")

clean_var3(df_train,'cap-color')
clean_var3(df_exam,'cap-color')
clean_cat(df_train,df_exam,'cap-color')


# In[ ]:


df_all_data['gill-attachment'].unique()


# In[ ]:


df_all_data['gill-color'].unique()


# In[ ]:


def clean_var4(df,var):
    df[var] = df[var].replace(to_replace='p p',value="p")
    df[var] = df[var].replace(to_replace='4. n',value="n")
    df[var] = df[var].replace(to_replace='p p',value="p")
    df[var] = df[var].replace(to_replace='does None',value="None")
    df[var] = df[var].replace(to_replace='has f',value="f")
    df[var] = df[var].replace(to_replace='is a',value="a")
    df[var] = df[var].replace(to_replace='has d',value="d")
    df[var] = df[var].replace(to_replace='does f',value="f")
    df[var] = df[var].replace(to_replace='is None',value="None")

clean_var4(df_train,'gill-attachment')
clean_var4(df_exam,'gill-attachment')
clean_cat(df_train,df_exam,'gill-attachment')


# In[ ]:


def clean_var6(df,var):
    df[var] = df[var].replace(to_replace='does w',value="w")
    df[var] = df[var].replace(to_replace='is w',value="w")
    df[var] = df[var].replace(to_replace='does n',value="n")
    df[var] = df[var].replace(to_replace='does f',value="f")
    df[var] = df[var].replace(to_replace='is y',value="y")
    df[var] = df[var].replace(to_replace='has g',value="g")
    df[var] = df[var].replace(to_replace='does None',value="None")

clean_var6(df_train,'gill-color')
clean_var6(df_exam,'gill-color')
clean_cat(df_train,df_exam,'gill-color')


# In[ ]:


df_all_data['stem-color'].unique()


# In[ ]:


def clean_var8(df,var):
    df[var] = df[var].replace(to_replace='is w',value="w")
    df[var] = df[var].replace(to_replace='is n',value="n")

clean_var8(df_train,'stem-color')
clean_var8(df_exam,'stem-color')
clean_cat(df_train,df_exam,'stem-color')


# In[ ]:


df_all_data['has-ring'].unique()


# In[ ]:


def clean_var9(df,var):
    df[var] = df[var].replace(to_replace='f has-ring',value="f")

clean_var9(df_train,'has-ring')
clean_var9(df_exam,'has-ring')
clean_cat(df_train,df_exam,'has-ring')


# In[ ]:


df_all_data['ring-type'].unique()


# In[ ]:


def clean_var10(df,var):
    df[var] = df[var].replace(to_replace='does f',value="f")
    df[var] = df[var].replace(to_replace='is p',value="p")

clean_var10(df_train,'ring-type')
clean_var10(df_exam,'ring-type')
clean_cat(df_train,df_exam,'ring-type')


# In[ ]:


df_all_data['habitat'].unique()


# In[ ]:


def clean_var11(df,var):
    df[var] = df[var].replace(to_replace='is w',value="w")
    df[var] = df[var].replace(to_replace='is h',value="h")

clean_var11(df_train,'habitat')
clean_var11(df_exam,'habitat')
clean_cat(df_train,df_exam,'habitat')


# In[ ]:


df_all_data['season'].unique()


# ## <a id="1.5"></a>
# ## 1.5 -Handling missing values in categorical data

# In[ ]:


# check missing values


# **There are still a lot of missing values.**
# 
# **Here, we will replace them by a new category 'None'.**
# 
# **The category 'None' already exist, so we won't add a new one.**  

# In[ ]:


list_col_object = df_train.select_dtypes(include=['object']).columns.tolist()
list_col_object.remove('class')

for col in list_col_object:
    df_train[col].fillna('Missing', inplace=True)
    df_exam[col].fillna('Missing', inplace=True)
#     df_train[col].fillna('None', inplace=True)
#     df_exam[col].fillna('None', inplace=True)


# In[ ]:


# Detect number of columns
num_cols = len(list_col_object)

# Calculate the number of rows and columns for subplots
num_rows = int(np.ceil(num_cols / 3))  # 3 columns per row

plt.suptitle("Distribution of Categorical Features", y=1.005, size=24)
plt.subplots_adjust(hspace = 0.7, wspace=0.6)

fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 3))
axes = axes.flatten()

# Plot histograms for each column
for i, col in enumerate(list_col_object):
    sns.countplot(x=col, hue='class', data=df_train,ax=axes[i])
    axes[i].set_title(f'count Plot of {col}')
    axes[i].set_ylabel('')

# Turn off any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.show()


# ## <a id="1.5"></a>
# ## 1.5 -Numerical data
# ### 1.5.1 - Distribution

# In[ ]:


list_col_numeric = list(df_train.select_dtypes(include=[np.number]).columns)
list_col_numeric


# In[ ]:


# Create a figure with a specific size
fig = plt.figure(figsize=(12, 6))
fig.subplots_adjust(hspace=0.4, wspace=0.3)

# Figure title
fig.suptitle("Distribution of Numerical Variables", ha='center', fontsize=18, fontweight='bold')

# Subplot 1:
ax1 = fig.add_subplot(3,2, 1)
sns.histplot(data = df_train,x='cap-diameter',hue='class', kde=True, color='blue', bins=20, stat='density', alpha=0.6,ax=ax1)
ax1.set_title('cap-diameter df_train')
ax1.set_xlabel('')

# Subplot 2:
ax2 = fig.add_subplot(3,2, 2)
sns.histplot(data = df_exam,x='cap-diameter', kde=True, color='green', bins=20, stat='density', alpha=0.6,ax=ax2)
ax2.set_title('cap-diameter df_test')
ax2.set_xlabel('')

# Subplot 3: 
ax3 = fig.add_subplot(3,2, 3)
sns.histplot(data = df_train,x='stem-height',hue='class', kde=True, color='blue', bins=20, stat='density', alpha=0.6,ax=ax3)
ax3.set_title('stem-height df_train')
ax3.set_xlabel('')

# Subplot 4: 
ax4 = fig.add_subplot(3,2, 4)
sns.histplot(data = df_exam,x='stem-height', kde=True, color='green', bins=20, stat='density', alpha=0.6,ax=ax4)
ax4.set_title('stem-height df_test')
ax4.set_xlabel('')

# Subplot 3: 
ax5 = fig.add_subplot(3,2, 5)
sns.histplot(data = df_train,x='stem-width',hue='class', kde=True, color='blue', bins=20, stat='density', alpha=0.6,ax=ax5)
ax5.set_title('stem-height df_train')
ax5.set_xlabel('')

# Subplot 4: 
ax6 = fig.add_subplot(3,2, 6)
sns.histplot(data = df_exam,x='stem-width', kde=True, color='green', bins=20, stat='density', alpha=0.6,ax=ax6)
ax6.set_title('stem-height df_test')
ax6.set_xlabel('')

# Show the plot
plt.show()


# In[ ]:


df_train[list_col_numeric].isnull().sum()


# In[ ]:


df_exam[list_col_numeric].isnull().sum()


# Only 4 missing values for numerical data... let's see where there are.

# In[ ]:


rows_with_missing = df_train[df_train.isna().any(axis=1)]
rows_with_missing


# May be do that later to improve model performance:
# 
# Base on the plot showing 'cap-diameter' distribution:
# - The one with 'class = e' have a normal distribution, so the missing value will be replace with the mean of this group.
# - The one with 'class = p' have a positive skew distribution, so the missing value will be replace with the median of this group

# In[ ]:


df_train['cap-diameter'].fillna(df_train['cap-diameter'].mean(), inplace=True)
df_exam['cap-diameter'].fillna(df_exam['cap-diameter'].mean(), inplace=True)
df_exam['stem-height'].fillna(df_exam['stem-height'].mean(), inplace=True)


# ## <a id="1.6"></a>
# ## 1.6 -TARGT DATA
# 

# In[ ]:


# Map the target 'class' column to 0 and 1
df_train['class'] = df_train['class'].map({'e': 0, 'p': 1})


# In[ ]:


target = df_train['class']
plt.figure(figsize=(5,3))
sns.countplot(x=target, data=df_train)
plt.title("Target Distribution", size=18)
plt.plot()


# #### The target is imbalanced.

# ## <a id="2"></a>
# <div style="
#            border-radius:50px;
#            background-color:#7ca4cd;
#            font-size:200%;
#            font-family:Arial;
#            letter-spacing:0.10px">
# <p style="padding: 10px;
#           color:white;
#           text-align:center;">2: Prepare data for MACHINE LEARNING
# </p>
# </div>

# In[ ]:


X = df_train.drop(columns=['class'])
X_exam  = df_exam.copy()


# In[ ]:


from imblearn.over_sampling import RandomOverSampler

# Assuming X is your features and y is your target
ros = RandomOverSampler(random_state=42)
X_resampled, target_resampled = ros.fit_resample(X, target)


# In[ ]:


# Look at new distributuion
counter = Counter(target_resampled)

# Plotting
plt.bar(counter.keys(), counter.values())
plt.ylabel('Count')
plt.title('Category Counts')
plt.show()


# ## 2.2 -Split the data

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_resampled, target_resampled, test_size=0.3, random_state=42)


# In[ ]:


X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


# ## 2.3 -Scale the data

# In[ ]:


# Separate numerical and categorical columns in train dataset
X_train_numerical = X_train[list_col_numeric]
X_train_categorical = X_train[list_col_object]

# Separate numerical and categorical columns in test dataset
X_test_numerical = X_test[list_col_numeric]
X_test_categorical = X_test[list_col_object]

# Separate numerical and categorical columns in exam dataset
X_exam_numerical = X_exam[list_col_numeric]
X_exam_categorical = X_exam[list_col_object]

#### SCALE NUMERICAL DATA
# scaler = StandardScaler()
# X_train_numerical_scaled = scaler.fit_transform(X_train_numerical)
# X_test_numerical_scaled = scaler.transform(X_test_numerical)

# Apply Box-Cox transformation only to positive features in the training set
# Initialize empty DataFrames to hold the scaled values
X_train_numerical_scaled = pd.DataFrame(index=X_train_numerical.index, columns=X_train_numerical.columns)
X_test_numerical_scaled = pd.DataFrame(index=X_test_numerical.index, columns=X_test_numerical.columns)
X_exam_numerical_scaled = pd.DataFrame(index=X_exam_numerical.index, columns=X_exam_numerical.columns)

for column in X_train_numerical.columns:
    # Fit the transformation on the training data and apply the transformation
    X_train_numerical_scaled[column], fitted_lambda = boxcox(X_train_numerical[column] + 1)
    
    # Apply the transformation to the test data using the same lambda
    X_test_numerical_scaled[column] = boxcox(X_test_numerical[column] + 1, fitted_lambda)
    X_exam_numerical_scaled[column] = boxcox(X_exam_numerical[column] + 1, fitted_lambda)

# Convert scaled numerical features back to DataFrame with original indices
df_X_train_numerical_scaled = pd.DataFrame(X_train_numerical_scaled, columns=X_train_numerical.columns, index=X_train.index)
df_X_test_numerical_scaled = pd.DataFrame(X_test_numerical_scaled, columns=X_test_numerical.columns, index=X_test.index)
df_X_exam_numerical_scaled = pd.DataFrame(X_exam_numerical_scaled, columns=X_exam_numerical.columns, index=X_exam.index)

# Concatenate scaled numerical data with categorical data
X_train_scaled = pd.concat([df_X_train_numerical_scaled, X_train_categorical], axis=1)
X_test_scaled = pd.concat([df_X_test_numerical_scaled, X_test_categorical], axis=1)
X_exam_scaled = pd.concat([df_X_exam_numerical_scaled, X_exam_categorical], axis=1)


# ## 2.4 Model selection
# 
# we will test machine learning classifiers that can handle categorical features directly without requiring explicit encoding into numerical formats.
# 
# In this case, 'object' type should be 'category' type.

# In[ ]:


for col in list_col_object:
    X_train_scaled[col] = X_train_scaled[col].astype('category')
    X_test_scaled[col] = X_test_scaled[col].astype('category')
    X_exam_scaled[col] = X_exam_scaled[col].astype('category')


# In[ ]:


def model_report(model, X_train, y_train,X_test, y_test):
    print("="*80)
    print(f"    Model: {model.__class__.__name__}")
    print("="*80)
     
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    
    accuracy = accuracy_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    
    print('accuracy: ', accuracy)
    print('recall: ',recall)
    print('precision: ', precision)
    print('f1: ', f1)
    plt.show()


# ### 2.4.1: Model selections

# In[ ]:


def model_report2(model, X_train, y_train,X_test, y_test,cat_features):
    print("="*80)
    print(f"    Model: {model.__class__.__name__}")
    print("="*80)
     
    model.fit(X_train, y_train,cat_features=cat_features)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    
    accuracy = accuracy_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    
    print('accuracy: ', accuracy)
    print('recall: ',recall)
    print('precision: ', precision)
    print('f1: ', f1)
    plt.show()


# In[ ]:


# from catboost import CatBoostClassifier
cat_features= [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
# clf = CatBoostClassifier()
# model_report2(clf, X_train_scaled, y_train,X_test_scaled, y_test,cat_features)


# In[ ]:


# Initialize the CatBoostClassifier
model = CatBoostClassifier(
    verbose=0,  # Silent mode
    random_state=42  # For reproducibility
)

# Step 4: Set up the hyperparameter grid for GridSearchCV
param = {
    'iterations': [100, 200, 500],
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'l2_leaf_reg': [1, 3, 5, 7, 9]
}

# Step 5: Initialize GridSearchCV
random_search = RandomizedSearchCV(
    model,
    param_distributions=param,
    n_iter=40,  # Number of parameter settings to sample
    scoring='roc_auc',  # Choose your metric, e.g., 'accuracy', 'f1', 'roc_auc'
    cv=3,  # Number of cross-validation folds
    verbose=1,
    n_jobs=-1,
    random_state=42
)
# Fit the model
random_search.fit(X_train_scaled, y_train,cat_features=cat_features)

# Best hyperparameters


# In[ ]:





# In[ ]:


#  y_pred_exam = clf.predict(X_exam_scaled)


# In[ ]:


# param_grid = {
#     'depth': [4, 8],
#     'learning_rate': [0.01, 0.05, 0.1],
# }

# model = CatBoostClassifier(eval_metric='AUC', random_seed=42)
# grid_search = GridSearchCV(model, param_grid, scoring='roc_auc', cv=3, verbose=2)
# grid_search.fit(X_train_scaled, y_train,cat_features=cat_features)

# print(f"Best parameters found: {grid_search.best_params_}")


# In[ ]:


# import lightgbm as lgb

# # Convert to LightGBM Dataset
# train_data = lgb.Dataset(X, label=y, categorical_feature=cat_features)

# # Parameters
# params = {
#     'objective': 'binary',
#     'metric': 'binary_logloss'
# }

# # Train the model
# clf = lgb.train(params, train_data, num_boost_round=100)


# In[ ]:





# In[ ]:


# import lightgbm as lgb

# lgb_model = lgb.LGBMClassifier(objective='binary', random_state=42)  # For binary classification; use 'multiclass' for multiclass problems


# param_dist = {
#     'num_leaves': [50, 70, 100],
#     'max_depth': [-1, 10, 20, 30, 40],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'n_estimators': [50, 100, 200],
#     'min_child_samples': [10, 20, 30, 50, 100],
#     'subsample': [0.7, 0.8, 1.0],
#     'colsample_bytree': [0.6,  0.8, 1.0],
# }

# # Initialize RandomizedSearchCV
# random_search = RandomizedSearchCV(
#     lgb_model,
#     param_distributions=param_dist,
#     n_iter=80,  # Number of parameter settings to sample
#     scoring='roc_auc',  # Choose your metric, e.g., 'accuracy', 'f1', 'roc_auc'
#     cv=3,  # Number of cross-validation folds
#     verbose=1,
#     n_jobs=-1,
#     random_state=42
# )

# # Fit the model
# random_search.fit(X_train_scaled, y_train)

# # model = CatBoostClassifier(eval_metric='AUC', random_seed=42)
# # grid_search = GridSearchCV(model, param_grid, scoring='roc_auc', cv=3, verbose=2)
# # grid_search.fit(X_train_scaled, y_train,cat_features=cat_features)

# # Best hyperparameters
# print(f"Best hyperparameters: {random_search.best_params_}")


# In[ ]:


# # best parameters
# params = {
#     'subsample': 0.7,
#     'num_leaves': 100,
#     'n_estimators': 200,
#     'min_child_samples': 20,
#     'max_depth': 30,
#     'learning_rate': 0.1,
#     'colsample_bytree': 0.6}
    
# lgb_model = lgb.LGBMClassifier(objective='binary', random_state=42,**params)
# lgb_model.fit(X_train_scaled, y_train)
# y_pred_exam = lgb_model.predict(X_exam_scaled)

# # create output
# list_id = df_exam.index
# output = pd.DataFrame({'id': list_id,
#                        'class': y_pred_exam})

# # Map the target 'class' column to 0 and 1
# output['class'] = output['class'].map({0:'e', 1:'p'})

# output.to_csv('submission4.csv', index=False)


# In[ ]:


# xgb_clf = XGBClassifier(enable_categorical=True, device="cuda", tree_method="hist")
# model_report(xgb_clf, X_train_scaled, y_train,X_test_scaled, y_test)


# In[ ]:


# lgb_clf = LGBMClassifier(device='CPU', verbosity=-1)

# model_report(lgb_clf, X_train_scaled, y_train,X_test_scaled, y_test)


# In[ ]:


# from catboost import CatBoostClassifier

# # Example parameters
# params = {
#     'iterations': 1000,          # Number of boosting rounds
#     'learning_rate': 0.1,        # Learning rate for gradient boosting
#     'depth': 6,                  # Depth of each tree
#     'loss_function': 'Logloss',  # Loss function to optimize
#     'eval_metric': 'AUC',        # Evaluation metric
#     'random_seed': 42,           # Random seed for reproducibility
#     'verbose': 100,              # Verbosity level, printing every 100 iterations
#     'cat_features': [0, 3, 5]    # Indexes of categorical features in your dataset
# }

# # Initialize CatBoostClassifier with the specified parameters
# model = CatBoostClassifier(**params)


# In[ ]:


# # model 1
# lgb_clf = LGBMClassifier(device='CPU', verbosity=-1)
# lgb_clf.fit(X_train_scaled, y_train)
# y_pred_exam = lgb_clf.predict(X_exam_scaled)


# In[ ]:


# # Define the model
# lgb_clf = LGBMClassifier(device='CPU', verbosity=-1)

# # Define the hyperparameters to tune
# param_grid = {
#     'num_leaves': [31, 50, 70],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'n_estimators': [100, 200, 500],
#     'max_depth': [3, 5, 7],
# #     'min_child_samples': [20, 30, 40]
# }

# # Set up the GridSearchCV
# grid_search = GridSearchCV(estimator=lgb_clf, param_grid=param_grid,
#                            cv=3, n_jobs=-1, verbose=1, scoring='accuracy')

# # Fit the grid search
# grid_search.fit(X_train_scaled, y_train)

# # Best parameters
# print("Best parameters found: ", grid_search.best_params_)

# # Best model
# best_model = grid_search.best_estimator_

# # Evaluate the best model on the test set
# lgb_clf = LGBMClassifier(device='CPU', verbosity=-1)
# lgb_clf.fit(X_test_scaled)

# y_pred = best_model.predict(X_test_scaled)
# accuracy = accuracy_score(y_test, y_pred)
# print("Test set accuracy: ", accuracy)


# Fitting 3 folds for each of 27 candidates, totalling 81 fits
# 
# Best parameters found:  {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 500}

# In[ ]:


# # Parameters
# params = {
#     'learning_rate': 0.1,
#     'max_depth': 5,
#     'n_estimators': 500,
#     'device': 'CPU',        # Ensures the model runs on the CPU
#     'verbosity': -1         # Suppresses output
# }

# # Initialize the LGBMClassifier with the specified parameters
# best_model = LGBMClassifier(**params)
# best_model.fit(X_train_scaled, y_train)
# y_pred_exam = best_model.predict(X_exam_scaled)


# In[ ]:


# model_report(best_model, X_train_scaled, y_train,X_test_scaled, y_test)


# In[ ]:





# In[ ]:


best_model.fit(X_train_scaled, y_train)
y_pred_exam = best_model.predict(X_exam_scaled)


# In[ ]:


list_id = df_exam.index
output = pd.DataFrame({'id': list_id,
                       'class': y_pred_exam})

# Map the target 'class' column to 0 and 1
output['class'] = output['class'].map({0:'e', 1:'p'})

output.to_csv('submission4.csv', index=False)

output.head()


# In[ ]:




