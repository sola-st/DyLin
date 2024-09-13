#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Load the train and test dataset
data = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')


# In[ ]:


data.shape


# In[ ]:


data.head(2)


# In[ ]:


# Data preprocessing

# Drop Rows with Missing Values
data.fillna(0, inplace=True)


# Drop the Passenger ID Column as its the Unique Identifier
data.drop('PassengerId', inplace=True, axis=1)


# Convert mixed type columns to string
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].astype(str)
        
        
# Identify categorical columns
categorical_cols = [col for col in data.columns if data[col].dtype == 'object']


# Apply label encoding to categorical columns
oe = {}
for col in categorical_cols:
    oe[col] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    data[col] = oe[col].fit_transform(data[col].values.reshape(-1, 1))


# In[ ]:


# Separating features and target
X = data.drop('Transported', axis=1)
y = data['Transported']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


(X_train.shape, y_train.shape), (X_test.shape, y_test.shape)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

# Print a detailed classification report


# In[ ]:


# Load Submission Data
submission_data = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')

# Drop Rows with Missing Values
submission_data.fillna(0, inplace=True)

# Drop the Passenger ID Column as it's the Unique Identifier
submission_data.drop('PassengerId', inplace=True, axis=1)

# Convert mixed type columns to string
for col in submission_data.columns:
    if submission_data[col].dtype == 'object':
        submission_data[col] = submission_data[col].astype(str)

# Apply label encoding to the submission data
for column, encoder in oe.items():
    submission_data[column] = encoder.transform(submission_data[[column]])

# Use the trained model to predict the 'Transported' status for the submission data
y_pred_submission = model.predict(submission_data)

# Add the predictions to the submission data
submission_data['Transported'] = y_pred_submission

# Keep only the 'PassengerId' and 'Transported' columns for submission
submission_data = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')[['PassengerId']]
submission_data['Transported'] = y_pred_submission

# Save the results to a CSV file
submission_data.to_csv('/kaggle/working/submission.csv', index=False, header=True)

