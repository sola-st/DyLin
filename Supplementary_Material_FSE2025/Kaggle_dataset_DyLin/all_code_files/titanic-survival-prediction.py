#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import make_scorer, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Set default plot size
sns.set_theme(context={'figure.figsize': (24, 12)})
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Load dataset with error handling
try:
    train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
    test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
    
    df = pd.concat([train_df, test_df], sort=True).reset_index(drop=True)
except FileNotFoundError:
    raise FileNotFoundError('The dataset file was not found. Please check the file path.')
except pd.errors.EmptyDataError:
    raise ValueError('The dataset file is empty. Please check the file.')

# Display the first few rows of the combined dataset to understand its structure
df.head()  


# In[ ]:


# Output general information about the datasets to understand data dimensions and structure



# In[ ]:


# Display the data types and the count of non-null values
df.info(show_counts=True)


# # Data Cleaning
# In this section, we will handle missing values and remove or modify any data that is not suitable for machine learning models.
# 

# In[ ]:


# Caclulate the correlation matrix for all features and convert it to absolute values
correlation_matrix = df.corr(numeric_only=True).abs()

# Unstack the correlation matrix into a Series and sort it in descending order
correlation_pairs = correlation_matrix.unstack().sort_values(kind='quicksort', ascending=False).reset_index()

# Rename the columns for better clarity
correlation_pairs.columns = ['First feature', 'Second feature', 'Correlation coefficient']

# Filter the rows where the first feature is 'Age'
age_correlation_pairs = correlation_pairs[correlation_pairs['First feature'] == 'Age']

age_correlation_pairs


# In[ ]:


# Fill missing 'Age' values using the median of groups ('Pclass', 'SibSp')
df['Age'] = df.groupby(['Pclass', 'SibSp'])['Age'].transform(lambda x: x.fillna(x.median()))

# Check for any remaining missing values in 'Age'


# Filling missing 'Age' values using the median from groups based on 'Pclass' and 'SibSp' is logical because passengers of the same class and similar family status may have similar ages.

# In[ ]:


# Fill missing 'Embarked' values with the most common embarkation point
df['Embarked'] = df['Embarked'].fillna('S')

# Check for any remaining missing values in 'Embarked'


# The most common embarkation point is 'S' (Southampton), so it's reasonable to fill missing values with it.

# In[ ]:


# Drop the 'Cabin' column because it contains too many missing values (almost 77%)
df.drop('Cabin', axis=1, inplace=True)

# Verify all columns after changes


# In[ ]:


# Retrieve the rows where the 'Fare' column has null values to analyze them
df[df['Fare'].isnull()]

# Fill missing 'Fare' values with median fare for passengers of similar characteristics (3rd class, no parents or siblings)
median_fare = df.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
df['Fare'] = df['Fare'].fillna(median_fare)

# Check for any remaining missing values in 'Fare'


# # Feature Engineering
# In this section, we will create new features that could potentially improve the predictive power of our models.

# In[ ]:


# Create a new feature 'Family_Size' to indicate the size of the family (siblings/spouses + parents/children + the passenger themselves)
df['Family_Size'] = df['SibSp'] + df['Parch'] + 1

# Display the count of unique values in 'Family_Size' to understand its distribution


#  Family size could be an important feature as larger families may have different survival patterns compared to individuals.

# In[ ]:


# Extract 'Title' from the 'Name' column to get the title part
df['Title'] = df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

# Grouping titles into broader categories
def group_titles(title):
    if title in ['Mr', 'Don', 'Sir']:
        return 'Mr'
    elif title in ['Mrs', 'Mme', 'Lady', 'the Countess', 'Dona']:
        return 'Mrs'
    elif title in ['Miss', 'Mlle']:
        return 'Miss'
    elif title == 'Master':
        return 'Master'
    elif title in ['Rev', 'Dr', 'Col', 'Major', 'Capt']:
        return 'Officer'
    else:
        return 'Other'

# Apply the grouping function to the 'Title' column
df['Title'] = df['Title'].apply(group_titles)

# Display the count of unique values in the 'Title' column to understand its distribution


#  Titles can give insight into a person's age, sex, and social status, which may influence survival rates.

# In[ ]:


# Initialize 'Is_Married' column with 0 and update it to 1 where the 'Title' is 'Mrs'
df['Is_Married'] = 0
df.loc[df['Title'] == 'Mrs', 'Is_Married'] = 1

# Display the count of unique values in the 'Is_Married' column
df.Is_Married.value_counts()


# Marital status could be an important factor in survival, as married women (Mrs) may have had different priorities or protections.

# # Exploratory Data Analysis (EDA)
# In this section, we will visualize various aspects of the dataset to understand the data better.

# In[ ]:


# Plot to show survival based on marital status
sns.countplot(
    data=df, 
    x='Survived', 
    hue='Is_Married',
    palette='coolwarm'
)

# Add labels and title
plt.title('Survival Based on Marital Status', fontsize=16, fontweight='bold')
plt.xlabel('Marital Status', fontsize=12, fontweight='bold')
plt.ylabel('Number of Passengers', fontsize=12, fontweight='bold')

# Customize the legend
plt.legend(title='Survived', title_fontsize='16', fontsize='16', loc='upper right', frameon=True)

# Improve layout spacing
plt.tight_layout()

# Show the plot
plt.show()


# In[ ]:


# Plot the count of each title after grouping
sns.barplot(
    x=df['Title'].value_counts().index,
    y=df['Title'].value_counts().values,
    palette='coolwarm'
           )

# Add labels and title
plt.title('Title feature value counts after grouping', fontsize=16, fontweight='bold')
plt.xlabel('Title', fontsize=12, fontweight='bold')
plt.ylabel('Number of Passengers', fontsize=12, fontweight='bold')

# Add data labels on top of each bar
title_counts = df['Title'].value_counts()
for i, count in enumerate(title_counts.values):
    plt.text(i, count + 1, str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')

# Display the plot
plt.tight_layout()
plt.show()


# In[ ]:


# Create the countplot
sns.countplot(
    data=df, 
    x='Sex', 
    hue='Survived',
    palette='coolwarm'
)

# Add labels and title
plt.title('Survival Based on Sex', fontsize=16, fontweight='bold')
plt.xlabel('Sex', fontsize=12, fontweight='bold')
plt.ylabel('Number of Passengers', fontsize=12, fontweight='bold')

# Customize the legend
plt.legend(title='Survived', title_fontsize='16', fontsize='16', loc='upper right', frameon=True)

# Improve layout spacing
plt.tight_layout()

# Show the plot
plt.show()


# In[ ]:


# Create the barplot that counts unique values in 'Family_Size' column
sns.barplot(
    x=df['Family_Size'].value_counts().index,
    y=df['Family_Size'].value_counts().values,
    palette='coolwarm'
           )

# Add labels and title
plt.title('Count of people by family size', fontsize=16, fontweight='bold')
plt.xlabel('Title', fontsize=12, fontweight='bold')
plt.ylabel('Number of Passengers', fontsize=12, fontweight='bold')

# Add data labels on top of each bar
title_counts = df['Family_Size'].value_counts()
for i, count in enumerate(title_counts.values):
    plt.text(i, count + 1, str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
    
# Display the plot
plt.tight_layout()
plt.show()


# In[ ]:


# Create the barplot that counts unique values in 'Pclass' column
sns.barplot(
    x=df['Pclass'].value_counts().index.astype(str),
    y=df['Pclass'].value_counts().values,
    palette='coolwarm'
           )

# Add labels and title
plt.title('Count of people by Pclass', fontsize=16, fontweight='bold')
plt.xlabel('Pclass', fontsize=12, fontweight='bold')
plt.ylabel('Number of Passengers', fontsize=12, fontweight='bold')

# Add data labels on top of each bar
pclass_counts = df['Pclass'].value_counts()
for i, count in enumerate(pclass_counts.values):
    plt.text(i, count + 1, str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
    
# Display the plot
plt.tight_layout()
plt.show()


# In[ ]:


# Create the age distribution plot
sns.histplot(
    data=df['Age'],
    kde=True,
    bins=30,
    color='#e74c3c',
    alpha=0.7
)

# Add title and labels
plt.title('Age Distribution of Passengers', fontsize=16, fontweight='bold')
plt.xlabel('Age', fontsize=12, fontweight='bold')
plt.ylabel('Number of Passengers', fontsize=12, fontweight='bold')

# Customize the ticks
plt.xticks(fontsize=10, fontweight='bold')
plt.yticks(fontsize=10, fontweight='bold')

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()


# # Feature Transformation
# In this section, we will transform categorical features into numerical ones to be used in machine learning models.

# In[ ]:


# Convert categorical variables to dummy variables
df = pd.get_dummies(df, columns=['Sex', 'Title', 'Embarked', 'SibSp', 'Parch'])


# In[ ]:


# Drop irrelevant columns that do not provide useful information for predicting survival
df = df.drop(['Name', 'Ticket'], axis=1)

# Display the first few rows of the modified dataset to check changes
df.head()


# In[ ]:


# Plot the correlation matrix of DataFrame features to see if any features are strongly correlated
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, linecolor='black')

plt.title('Correlation Matrix of Dataset Features', fontsize=20)
plt.xticks(rotation=90, fontsize=12)
plt.yticks(fontsize=12)
plt.show()


#  The correlation matrix helps identify if any features are highly correlated, which might affect model performance.

# # Model training and evalution
# In this section, we will train multiple machine learning models and evaluate their performance.

# In[ ]:


# Divide dataframe to training and test datasets
train_df, test_df = df.loc[:890], df.loc[891:].drop(['Survived'], axis=1)

# Output general information about datasets



# In[ ]:


# Split training dataset into features (X) and target (y)
X_train = train_df[[column for column in train_df.columns if column not in ['PassengerId', 'Survived']]]
y_train = train_df['Survived']

X_test = test_df[[column for column in train_df.columns if column not in ['PassengerId', 'Survived']]]

# Initialize a standard scaler to normalize feature values
scaler = StandardScaler()

# Standardize the numerical features
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


# Define various models and their hyperparameters to test
models = {
    'LogisticRegression': {
        'model': LogisticRegression(max_iter=500),
        'params': {
            'solver': ['liblinear', 'lbfgs',],
            'C': [0.01, 0.1, 1, 10, 100, 200],
            'penalty': ['l2'],
        }
    },
    'DecisionTreeClassifier': {
        'model': DecisionTreeClassifier(),
        'params': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 20, 30, 50, 100],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8, 10, 12],
            'max_features': [None, 'sqrt', 'log2']
        }
    },
    'RandomForestClassifier': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10, 20, 30],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8, 10],
            'bootstrap': [True, False],
            'max_features': ['sqrt', 'log2']
        }
    },
    'GradientBoostingClassifier': {
        'model': GradientBoostingClassifier(),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'max_depth': [3, 5, 7, 9],
            'min_samples_split': [2, 5, 10, 20],
            'subsample': [0.8, 0.9, 1.0]
        }
    }
}


#  Different models and hyperparameters are chosen to identify the best model that predicts survival most accurately.

# In[ ]:


# Define the StartifiedKFold cross-validator
cv = StratifiedKFold(n_splits=10, shuffle=True)

# Function to perform grid search and evaluate models
def evaluate_models(models, X_train, y_train):
    best_models = {}
    best_scores = {}

    for name, model_dict in models.items():
        model = model_dict['model']
        params = model_dict['params']
        
         # Define multiple scoring metrics
        scoring = {
            'f1': 'f1',
            'accuracy': make_scorer(accuracy_score)
        }
        
        # Perform Grid Search for each model
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=params,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
            refit='f1'
        )

        grid_search.fit(X_train, y_train)

        # Best parameters and score
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        best_accuracy = grid_search.cv_results_['mean_test_accuracy'][grid_search.best_index_]
        
        best_models[name] = grid_search.best_estimator_
        best_scores[name] = {'f1': best_score, 'accuracy': best_accuracy}
        
        print(f'{name} Best Parameters: {best_params}')
        print(f'{name} Best F1 score: {best_score:.4f} ')
        print(f'{name} Best Accuracy: {best_accuracy:.4f} \n')

    return best_models, best_scores


# This function helps to find the best model and its parameters using cross-validation and GridSearchCV.

# In[ ]:


# Evaluate models and find the best one
best_models, best_scores = evaluate_models(models, X_train, y_train)

# Determine and save the best model
best_model_name = max(best_scores, key=lambda k: best_scores[k]['accuracy'])
best_model_scores = best_scores[best_model_name]



# In[ ]:


# Evaluate the best model on the test set
best_model = best_models[best_model_name]
predictions = best_model.predict(X_test).astype(int)

output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)


#  The best model is used to make predictions on the test data, and the results are saved in a CSV file for submission on Kaggle.
