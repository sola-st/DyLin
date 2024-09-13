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


# <h1 style="background-color:#800080; border:2px solid #4A90E2; padding:15px; border-radius:10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); font-family:Georgia, serif; color:#333; text-align:center;">
#   <span style="color:#4A90E2; text-shadow: 1px 1px #FFF;">🍄 Exploring Mushroom Classifier Dataset 🍄</span>
# </h1>
# 

# ![mushroom-8761211_1280.jpg](attachment:a448e4a0-efdf-4abe-85f6-c50cff44f434.jpg)

# <h1 style="color:#800080; text-align:center; text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);">
#   📥 Importing Necessary libraries 📥
# </h1>

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,matthews_corrcoef
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, matthews_corrcoef
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder 
 


# <div style="background-color:#E6F9E6; border:2px solid #66C2A5; padding:20px; border-radius:10px; box-shadow: 0 2px 4px 0 rgba(0,0,0,0.1); font-family:Georgia, serif;">
#   <h1 style="color:#4A90E2; text-shadow: 1px 1px #FFF; text-align:center;">Mushroom Classification Using Voting Classifier (XGBoost and Random Forest)</h1>
#   <p style="color:#333; font-size:16px;">
#     This notebook applies a <span style="color:#FF6347;"><strong>Voting Classifier</strong></span> that combines two powerful models: <span style="color:#1E90FF;"><strong>XGBoost</strong></span> and <span style="color:#1E90FF;"><strong>Random Forest</strong></span>. The objective is to classify mushroom species as <span style="color:#32CD32;"><strong>edible</strong></span> or <span style="color:#FF4500;"><strong>poisonous</strong></span> based on various features. The following steps were carried out:
#   </p>
# 
#   <h2 style="color:#66C2A5; text-shadow: 1px 1px 2px rgba(0,0,0,0.1);">1. Data Preparation</h2>
#   <p style="color:#333; font-size:16px;">
#     We begin by loading the training data and dividing it into features (<span style="color:#1E90FF;"><code>X</code></span>) and the target variable (<span style="color:#1E90FF;"><code>y</code></span>). The categorical target variable was transformed into numerical values using <span style="color:#32CD32;"><strong>Label Encoding</strong></span>.
#   </p>
# 
#   <h2 style="color:#66C2A5; text-shadow: 1px 1px 2px rgba(0,0,0,0.1);">2. Model Definition and Pipelines</h2>
#   <p style="color:#333; font-size:16px;">
#     We defined parameters for the <span style="color:#1E90FF;"><strong>XGBoost</strong></span> classifier, utilizing <span style="color:#32CD32;"><strong>GPU acceleration</strong></span> for faster training. A <span style="color:#1E90FF;"><strong>Random Forest</strong></span> classifier was also defined as a baseline model. Both models are integrated into pipelines that handle missing data using <span style="color:#32CD32;"><strong>SimpleImputer</strong></span>.
#   </p>
# 
#   <h2 style="color:#66C2A5; text-shadow: 1px 1px 2px rgba(0,0,0,0.1);">3. Frequency Encoding</h2>
#   <p style="color:#333; font-size:16px;">
#     <span style="color:#32CD32;"><strong>Frequency encoding</strong></span> was applied to categorical features, converting them into numerical representations based on the frequency of their occurrences in the data. This encoding method helps prepare the data for machine learning models.
#   </p>
# 
#   <h2 style="color:#66C2A5; text-shadow: 1px 1px 2px rgba(0,0,0,0.1);">4. Voting Classifier Setup</h2>
#   <p style="color:#333; font-size:16px;">
#     The <span style="color:#32CD32;"><strong>Voting Classifier</strong></span> combines predictions from both the <span style="color:#1E90FF;"><strong>XGBoost</strong></span> and <span style="color:#1E90FF;"><strong>Random Forest</strong></span> models. Using <em>soft voting</em>, the classifier averages predicted probabilities to make more robust final predictions.
#   </p>
# 
#   <h2 style="color:#66C2A5; text-shadow: 1px 1px 2px rgba(0,0,0,0.1);">5. Model Training and Evaluation</h2>
#   <p style="color:#333; font-size:16px;">
#     The data was split into training and test sets, and the ensemble model was trained on the training data. 
#     Model evaluation metrics include accuracy, confusion matrix, classification report, and the <span style="color:#FF6347;"><strong>Matthews Correlation Coefficient (MCC)</strong></span>.
#   </p>
# </div>
# 

# <h1 style="color:#800080; text-align:center; text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);">
#   📥 Importing Training Data 📥
# </h1>

# In[ ]:


train_data=pd.read_csv('/kaggle/input/playground-series-s4e8/train.csv')
train_data


# In[ ]:


train_data.info()


# In[ ]:


train_data.columns


# In[ ]:


train_data=train_data.drop('id',axis=1)


# In[ ]:


train_data.describe()


# <h1 style="color:#FFA07A; text-align:center; text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);">
#   🔍 Exploratory Data Analysis (EDA) 🔍
# </h1>
# 

# <h1 style="color:#800080; text-align:left; text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);">
#   ● Univariate Analysis with <code>Box-plot</code>
# </h1>
# 

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set the style
plt.style.use('seaborn-v0_8-whitegrid')

# List of numerical features
numerical_feature = ['cap-diameter', 'stem-height', 'stem-width']

# Loop through each numerical feature
for column in numerical_feature:
    plt.figure(figsize=(8, 6))  # Set the figure size
    
    # Create a boxplot for the feature
    sns.boxplot(data=train_data, x=column)
    
    # Set the title for each plot
    plt.title(f'Boxplot of {column}', size=20)
    
    # Show the plot
    plt.show()


# <h1 style="color:#800080; text-align:left; text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);">
#   ● Univariate Analysis with <code>Count-plot</code>
# </h1>
# 

# In[ ]:


df=train_data.copy()
df


# In[ ]:


categorical_feature = ['cap-shape', 'cap-surface', 'cap-color', 'does-bruise-or-bleed', 'gill-attachment','veil-color','stem-root','veil-type','spore-print-color',
                       'gill-spacing', 'gill-color', 'stem-color', 'has-ring', 'ring-type', 'habitat', 'season','stem-surface']
for i, col in enumerate(categorical_feature):
    plt.figure(figsize=(8, 6))
    
    fil_data = df[col].value_counts()
    fil_cat = fil_data[fil_data>=100].index
    fil_df = df[df[col].isin(fil_cat)]
    
    sns.countplot(x=col, hue='class', data=fil_df)
    
    plt.title(f"Count Plot of {col}", size=20)
    plt.show()


# <h1 style="color:#800080; text-align:left; text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);">
#   ● Matthews Correlation Coefficient (MCC)
# </h1>

# The Matthews Correlation Coefficient (MCC) is a metric used to evaluate the quality of binary classifications. It takes into account true and false positives and negatives and is generally regarded as a balanced measure, even if the classes are of very different sizes. It is particularly useful for assessing the performance of a binary classifier when the data is imbalanced.
# 
# ### Formula:
# 
# The MCC is calculated using the formula:
# 
# \[
# \text{MCC} = \frac{(TP \times TN) - (FP \times FN)}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}
# \]
# 
# Where:
# - **TP**: True Positives
# - **TN**: True Negatives
# - **FP**: False Positives
# - **FN**: False Negatives
# 
# ### Interpretation:
# - **MCC = +1**: Perfect prediction (all true positives and true negatives).
# - **MCC = 0**: No better than random guessing.
# - **MCC = -1**: Completely incorrect predictions (all false positives and false negatives).
# 
# The MCC value ranges from -1 to +1:
# - **+1**: Perfect classification.
# - **0**: Random or chance predictions.
# - **-1**: Inverse prediction, meaning the classifier's predictions are entirely wrong.
# 
# ### Why Use MCC?
# - **Imbalanced Datasets**: MCC is especially useful in cases of imbalanced datasets, where accuracy may be misleading (for example, when one class dominates the data).
# - **Balanced Measure**: MCC provides a single summary statistic that incorporates all four components of the confusion matrix (TP, TN, FP, FN), making it a balanced measure that can be trusted even in imbalanced scenarios.
# 
# ### Example Usage in Python:
# 
# ```python
# from sklearn.metrics import matthews_corrcoef
# 
# # Example predictions and ground truth
# y_true = [1, 0, 1, 1, 0, 1, 0, 0]
# y_pred = [1, 0, 1, 0, 0, 1, 0, 1]
# 
# # Calculate MCC
# mcc = matthews_corrcoef(y_true, y_pred)
# print('Matthews Correlation Coefficient:', mcc)
# ```
# 
# In this example, the MCC will indicate how well the binary classifier has performed by providing a correlation coefficient between the actual and predicted values.

# <h1 style="color:#800080; text-align:left; text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);">
#   ● Dealing With Null Value
# </h1>

# In[ ]:


train_data.drop('class',axis=1).isnull().sum()


# <h1 style="color:#800080; text-align:center; text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);">
#   📊 Percentage of Missing Values for Each Feature 📊
# </h1>
# 

# In[ ]:


# Calculate the percentage of missing values for each feature
missing_percentage = (train_data.drop(['class', 'stem-height', 'stem-width', 'season'], axis=1).isnull().sum() / train_data.shape[0]) * 100

# Convert the result to a DataFrame
missing_df = pd.DataFrame({'Feature': missing_percentage.index, 'Missing Percentage': missing_percentage.values})

# Plot using Seaborn
plt.figure(figsize=(12, 6))
sns.barplot(x='Feature', y='Missing Percentage', data=missing_df, palette='viridis')

# Set plot title and labels
plt.title('Percentage of Missing Values for Each Feature', size=20)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Feature', size=14)
plt.ylabel('Missing Percentage (%)', size=14)

# Show the plot
plt.tight_layout()
plt.show()


# <h1 style="background-color:#F0F4C3; border:2px solid #4A90E2; padding:15px; border-radius:10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);text-align:center; font-family:Georgia, serif; color:#333;">
#   <span style="color:#4A90E2; text-shadow: 1px 1px #FFF;">🔮 Voting Classifier Magic</span>
# </h1>
# 

# In[ ]:


# Import necessary libraries
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder
import pandas as pd


# <h1 style="color:#4A90E2; text-align:center; text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);">
#   ✔️ Training with All Data Using Voting Classifier ✔️
# </h1>
# 
# 

# In[ ]:


# Assuming df is your DataFrame with the dataset
df = train_data.copy()  # Copy the train_data

# Define features and target
X = df.drop('class', axis=1)
y = df['class']

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Frequency encoding for categorical features
categorical_feature = ['cap-shape', 'cap-surface', 'cap-color', 'does-bruise-or-bleed', 'gill-attachment', 'veil-color', 'stem-root', 
                       'veil-type', 'spore-print-color', 'gill-spacing', 'gill-color', 'stem-color', 'has-ring', 'ring-type', 
                       'habitat', 'season', 'stem-surface']

# Apply frequency encoding to training data
for column in categorical_feature:
    freq_encoding = X[column].value_counts().to_dict()
    X[column] = X[column].map(freq_encoding)



# Define LightGBM Classifier parameters
lgb_params = {
    'n_estimators': 2407,
    'learning_rate': 0.009462133032592785,
    'max_depth': 31,
    'min_child_weight': 47,
    'subsample': 0.6956431754146083,
    'colsample_bytree': 0.3670732604094118,
    'num_leaves': 73,
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'device': 'gpu',  # Use GPU if available
    'n_jobs': -1
}

# Create LightGBM pipeline
lightgbm_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values
    ('lightgbm', LGBMClassifier(**lgb_params))
])

# Create Random Forest pipeline
random_forest_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values
    ('random_forest', RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42, oob_score=True))
])

# Combine models in a Voting Classifier (Soft Voting)
voting_classifier = VotingClassifier(
    estimators=[
        ('lightgbm', lightgbm_pipeline),
        ('random_forest', random_forest_pipeline)
    ],
    voting='soft'  # Soft voting for probability-based decision making
)

# Train the voting classifier
voting_classifier.fit(X, y_encoded)


# <h1 style="color:#800080; text-align:center; text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);">
#   📥 Importing New Test Data 📥
# </h1>
# 

# In[ ]:


new_test_data=pd.read_csv('/kaggle/input/playground-series-s4e8/test.csv')
new_test_data


# In[ ]:


new_test_data.isnull().sum()


# In[ ]:


Id=new_test_data['id']


# In[ ]:


new_test_data=new_test_data.drop('id',axis=1)


# In[ ]:


new_test_data


# In[ ]:


voting_clf


# <h1 style="color:#800080; text-align:center; text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);">
#   🔮 Prediction of New Test Data 🔮
# </h1>
# 

# In[ ]:


X = df.drop('class', axis=1)

# Apply frequency encoding to new_test_data based on the encoding from X (the entire dataset before splitting)
for column in categorical_feature:
    freq_encoding = X[column].value_counts().to_dict()  # Get encoding from the full X
        
    new_test_data[column] = new_test_data[column].map(freq_encoding)  # Apply frequency encoding to new test data
    

# Use the trained pipeline to predict on new_test_data
y_test_pred =voting_clf.predict(new_test_data)
y_test_pred_original = label_encoder.inverse_transform(y_test_pred)


# <h1 style="color:#32CD32; text-align:center; text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);">
#   📤 Submission  📤
# </h1>
# 

# In[ ]:


submission = pd.DataFrame({'id': Id, 'class': y_test_pred_original})
submission 


# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




