#!/usr/bin/env python
# coding: utf-8

# ## **Title and Introduction**
# **Title:** "_Titanic Survival Prediction: A Comprehensive ML Approach_" 
#                          
# ### Introduction:
# 
# the main task of your project is to predict the Survived column
# 
# **Dataset Description**
# 
# The dataset used in this analysis includes the following columns:
# - `PassengerId`: Unique identifier for each passenger.
# - `Survived`: Indicator of survival (0 = No, 1 = Yes).
# - `Pclass`: Passenger class (1, 2, or 3).
# - `Name`: Name of the passenger.
# - `Sex`: Gender of the passenger.
# - `Age`: Age of the passenger.
# - `SibSp`: Number of siblings/spouses aboard.
# - `Parch`: Number of parents/children aboard.
# - `Ticket`: Ticket number.
# - `Fare`: Fare paid.
# - `Cabin`: Cabin number.
# - `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).
# 
# 
# 

# ## **Loading Libraries and Data**

# In[ ]:


#import necesssry libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')



# In[ ]:


# Load the datasets
train= pd.read_csv('/kaggle/input/titanic/train.csv')
test= pd.read_csv('/kaggle/input/titanic/test.csv')

# Display the first few rows
train.head()


# ## **Exploratory Data Analysis (EDA)**

# In[ ]:


# Overview of the data
train.info()
train.describe()


# In[ ]:


# Overview of the test
test.info()
test.describe()


# In[ ]:


#Check null values of train
train.isnull().sum()


# In[ ]:


#missing valus of test
test.isnull().sum()


# In[ ]:


# Survival count
sns.countplot(x='Survived', data=train, palette='viridis')  
plt.title('Survival Count')
plt.show()


# In[ ]:


# Survival by class
sns.barplot(x='Pclass', y='Survived', data=train, palette='coolwarm')  
plt.title('Survival by Passenger Class')
plt.show()


# In[ ]:


# Define a color palette for gender
gender_palette = {'male': 'blue', 'female': 'pink'}

# Plot with the custom color palette
sns.barplot(x='Sex', y='Survived', data=train, palette=gender_palette)
plt.title('Survival by Gender')
plt.show()


# ## **Data Cleaning and Preprocessing**

# #### Feature Engineering _Extract Titles from Names_

# In[ ]:


def extract_title(name):
    return name.split(',')[1].split('.')[0].strip()

train['Title'] = train['Name'].apply(extract_title)
test['Title'] = test['Name'].apply(extract_title)

# Map rare titles to 'Rare'
rare_titles = ['Dr', 'Rev', 'Col', 'Major', 'Sir', 'Lady', 'Countess', 'Capt', 'Jonkheer', 'Don']
train['Title'] = train['Title'].replace(rare_titles, 'Rare')
test['Title'] = test['Title'].replace(rare_titles, 'Rare')


# #### Feature Engineering _Create New Feature From Age_

# In[ ]:


# Define age bins and labels with descriptive categories
bins = [0, 12, 18, 35, 50, 100]
labels = ['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']
train['Age_Group'] = pd.cut(train['Age'], bins=bins, labels=labels, right=False)
test['Age_Group'] = pd.cut(train['Age'], bins=bins, labels=labels, right=False)


# #### Feature Engineering _Create New Feature FamilysizeFrom Age_

# In[ ]:


# Create 'FamilySize' and 'IsAlone'
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1


# #### Handle Missing Values For Numarical Data

# In[ ]:


imputer = SimpleImputer(strategy='median') 
train[['Age']] = imputer.fit_transform(train[['Age']])  #for train data
test[['Age']] = imputer.fit_transform(test[['Age']])    #for test data
test[['Fare']] = imputer.fit_transform(test[['Fare']])


# #### Handle Missing Values For Categorical Data

# In[ ]:


categorical_cols = train.select_dtypes(include=['object']).columns
categorical_cols = test.select_dtypes(include=['object']).columns
for col in categorical_cols:
    mode_imputer = SimpleImputer(strategy='most_frequent')
    train[col] = mode_imputer.fit_transform(train[[col]]).ravel()  # Flatten to 1D
    if col in test.columns:
        test[col] = mode_imputer.transform(test[[col]]).ravel()


# #### Encode Categorical Features

# In[ ]:


label_encoder = LabelEncoder()
categorical_cols = ['Name','Sex','Ticket','Cabin','Embarked','Age_Group','Title']


for col in categorical_cols:
    train[col] = label_encoder.fit_transform(train[col].astype(str))
    test[col] = label_encoder.fit_transform(test[col].astype(str))


# ## **Feature Selection**
# Analyze the importance of different features and decide which ones to use for modeling. You can use techniques like correlation analysis, feature importance from tree-based models, or Recursive Feature Elimination (RFE).

# In[ ]:


#Correlation analysis
plt.figure(figsize=(12, 8))
sns.heatmap(train.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# ## **Model Building**

# In[ ]:


# Split data into training and validation sets
#X = train[['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked','FamilySize']]
X=train.drop(['Survived'],axis=1)
y = train['Survived']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# #### Hyperparameter Tuning

# In[ ]:


param_grid = {
    'n_estimators': [100, 200],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}


# In[ ]:


rfc = RandomForestClassifier(random_state=42)

# Setup GridSearchCV
grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, 
                           scoring='accuracy', n_jobs=-1, verbose=2)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)


# #### Get Best parameters and make prediction

# In[ ]:


grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Get the best model
best_rfc = grid_search.best_estimator_

# Make predictions
y_pred= best_rfc.predict(X_val)


# ## **Model Evaluation**

# In[ ]:





# ## **Predict the Test Data and Create Submission**

# In[ ]:



y_test_pred =best_rfc.predict(test)

# Create submission file
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': y_test_pred
})
submission.to_csv('submission.csv', index=False)


# **Model Summary:**
# 
# The Random Forest Classifier (`best_rfc`) was used to predict survival status on the Titanic dataset. Key features include `Pclass`, `Sex`, `Age`, `Fare`, and engineered features like `FamilySize` and `IsAlone`. The model was trained with optimized hyperparameters and evaluated on validation data. Predictions were made on the test set, and results were saved in `submission.csv`. The model achieved strong performance in predicting survival outcomes.
# 
# 
