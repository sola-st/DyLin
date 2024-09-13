#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the read-only "/kaggle/input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Load Data

# In[ ]:


# Load the training data from the specified CSV file
train_data = pd.read_csv("/kaggle/input/titanic-cleaned-data/train_clean.csv")

# Display the first few rows of the training data
train_data.head()


# In[ ]:


# Load the test data from the specified CSV file
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

# Display the first few rows of the test data
test_data.head()


# In[ ]:


# Filter the training data to include only rows where 'Sex' is 'female'
women = train_data.loc[train_data.Sex == 'female']["Survived"]

# Calculate the survival rate for women
rate_women = sum(women) / len(women)

# Print the percentage of women who survived


# In[ ]:


# Filter the training data to include only rows where 'Sex' is 'male'
men = train_data.loc[train_data.Sex == 'male']["Survived"]

# Calculate the survival rate for men
rate_men = sum(men) / len(men)

# Print the percentage of men who survived


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Define the target variable
y = train_data["Survived"]

# Specify the features to be used for prediction
features = ["Pclass", "Sex", "SibSp", "Parch"]

# Convert categorical features into dummy/indicator variables for both train and test data
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

# Initialize and configure the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

# Train the model using the training data
model.fit(X, y)

# Make predictions on the test data
predictions = model.predict(X_test)

# Prepare the submission file with PassengerId and corresponding predictions
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

# Save the predictions to a CSV file
output.to_csv('submission.csv', index=False)

# Print a confirmation message

