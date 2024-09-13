#!/usr/bin/env python
# coding: utf-8

# # <p style="background-color: #21d179; color: #960206; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #960206; border: 5px dotted #960206;">🍄🔮 Mushrooms Danger Unveiled: Binary Prediction of Poisonous Species | PS4E8 |V2🍄🔍</p>
# 

# ![_5713ed55-8a0b-4e0a-a408-fca497d2c24d (1).jpg](attachment:bda62137-efbd-40aa-80c5-971327918ec8.jpg)

# # <p style="background-color: #21d179; color: #960206; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #960206; border: 5px dotted #960206;">Aims and Objectives</p>
# 

# * This is the Binary Prediction of Poisonous Mushrooms Competition Data.
# * The aim to take this competition is to predict whether a mushroom is edible or poisonous based on its physical characteristics.
# * For this I firstly concatenate the Train Data of competition with secondary-mushroom-dataset-data-set in order to enhance the Data quality and diversity and to improve the mcc score.
# * Then I take a look at the Duplicates of the Data. There is no duplicates present in competition data * but when i combine the train and original data then I found the Duplicates so, I Drop the Duplicates.
# * Then I Impute the missing values of numeric and categorical columns.
# * Then I apply the KFold Cross-Validation and Train the XGBoost Model.
# * Then in order to improve mcc score further I train the model again on best parameters
# * Then I create the submission File

# # <p style="background-color: #21d179; color: #960206; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #960206; border: 5px dotted #960206;">About Author</p>
# 

# * Hi Kagglers! I'm Maria Nadeem, a passionate Data Scientist with keen interest in exploring and applying diverse data science techniques.
# * As dedicated to derive meaningful insights and making impactful decisions through data, I actively engage in projects and contribute to Kaggle by sharing detailed analysis and actionable insights.
# * I'm excited to share my latest project on Binary Prediction of Poisonous Mushrooms.
# * In this notebook, I begin by conatenating train data of competition with secondary-mushroom-dataset-data-set.
# * Following this, I use the KFold Cross Validation, perform XGBoost model training. Then in order to improve mcc score further I train the model again on best parameters

# | Name               | Email                                               | LinkedIn                                                  | GitHub                                           | Kaggle                                        |
# |--------------------|-----------------------------------------------------|-----------------------------------------------------------|--------------------------------------------------|-----------------------------------------------|
# | **Maria Nadeem**  | marianadeem755@gmail.com | <a href="https://www.linkedin.com/in/maria-nadeem-4994122aa/" style="text-decoration: none; font-size: 16px;"><img src="https://img.shields.io/badge/LinkedIn-%2300A4CC.svg?style=for-the-badge&logo=LinkedIn&logoColor=white" alt="LinkedIn Badge"></a> | <a href="https://github.com/marianadeem755" style="text-decoration: none; font-size: 16px;"><img src="https://img.shields.io/badge/GitHub-%23FF6F61.svg?style=for-the-badge&logo=GitHub&logoColor=white" alt="GitHub Badge"></a> | <a href="https://www.kaggle.com/marianadeem755" style="text-decoration: none; font-size: 16px;"><img src="https://img.shields.io/badge/Kaggle-%238a2be2.svg?style=for-the-badge&logo=Kaggle&logoColor=white" alt="Kaggle Badge"></a> |
# 

# # <p style="background-color: #21d179; color: #960206; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #960206; border: 5px dotted #960206;">Import Libraries</p>
# 

# In[ ]:


#get_ipython().system('pip install optuna category_encoders')


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import matthews_corrcoef
from sklearn.base import clone
from xgboost import XGBClassifier
import optuna
from optuna.samplers import TPESampler


# # <p style="background-color: #21d179; color: #960206; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #960206; border: 5px dotted #960206;">Load Dataset</p>
# 

# In[ ]:


# Load datasets
train = pd.read_csv("/kaggle/input/playground-series-s4e8/train.csv", index_col="id")
original = pd.read_csv("/kaggle/input/secondary-mushroom-dataset-data-set/MushroomDataset/secondary_data.csv", sep=";")
test = pd.read_csv("/kaggle/input/playground-series-s4e8/test.csv", index_col="id")


# In[ ]:


# Combine training datasets
train = pd.concat([train, original], ignore_index=True)


# # <p style="background-color: #21d179; color: #960206; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #960206; border: 5px dotted #960206;">Remove Duplicates</p>
# 

# In[ ]:


# Remove duplicates
cols = train.columns.tolist()
cols.remove("class")
train = train.drop_duplicates(subset=cols, keep='first')



# # <p style="background-color: #21d179; color: #960206; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #960206; border: 5px dotted #960206;">Impute Missing values of Numeric Columns</p>
# 

# In[ ]:


# Identify columns with more than 70% missing values
missing_threshold = 0.7
missing_train = train.isnull().mean()
columns_to_drop = missing_train[missing_train > missing_threshold].index.tolist()

# Drop columns (except 'class')
columns_to_drop = [col for col in columns_to_drop if col != 'class']
train.drop(columns=columns_to_drop, inplace=True)
test.drop(columns=columns_to_drop, inplace=True)


# Identify numerical columns
numerical_cols = train.select_dtypes(include=[np.number]).columns.tolist()

# Fill missing numerical values with median
for col in numerical_cols:
    median = train[col].median()
    train[col].fillna(median, inplace=True)
    test[col].fillna(median, inplace=True)


# # <p style="background-color: #21d179; color: #960206; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #960206; border: 5px dotted #960206;">Impute Missing values of Categorical Columns</p>
# 

# In[ ]:


# Identify categorical columns
categorical_cols = train.select_dtypes(include=[object]).columns.tolist()

# Ensure 'class' column is not removed or processed
for col in categorical_cols:
    if col != 'class':
        # Fill missing values with 'not_present'
        train[col].fillna('not_present', inplace=True)
        test[col].fillna('not_present', inplace=True)
        
        # Combine rare categories (frequency less than 1%)
        freq = train[col].value_counts(normalize=True)
        rare_categories = freq[freq < 0.01].index
        train[col] = train[col].replace(rare_categories, 'infrequent')
        test[col] = test[col].replace(rare_categories, 'infrequent')
    


# # <p style="background-color: #21d179; color: #960206; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #960206; border: 5px dotted #960206;">Lets Encode Categorical Data</p>
# 

# In[ ]:


# Exclude 'class' from categorical columns
categorical_cols = [col for col in categorical_cols if col != 'class']

# Combine train and test data only for the categorical columns
combined = pd.concat([train[categorical_cols], test[categorical_cols]], axis=0)

# Ordinal Encoding
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
combined_encoded = encoder.fit_transform(combined)

# Split back to train and test
train_encoded = combined_encoded[:len(train)]
test_encoded = combined_encoded[len(train):]

# Replace original categorical columns with encoded ones
train[categorical_cols] = train_encoded
test[categorical_cols] = test_encoded


# # <p style="background-color: #21d179; color: #960206; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #960206; border: 5px dotted #960206;">Lets Select Featues and Target Variables</p>
# 

# In[ ]:


# Features and target
X = train.drop('class', axis=1)
y = train['class']

# Label encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Check shapes


# # <p style="background-color: #21d179; color: #960206; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #960206; border: 5px dotted #960206;">Lets Apply KFold Cross-Validation</p>
# 

# In[ ]:


# Define Stratified K-Fold
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

def k_fold_model_trainer(model, X, y, skf, test_data):
    """
    Trains the model using Stratified K-Fold cross-validation and evaluates using MCC.
    Returns the average validation MCC and test predictions.
    """
    train_mcc_scores = []
    val_mcc_scores = []
    test_preds = np.zeros(len(test_data))
    
    fold_number = 1
    for train_index, val_index in skf.split(X, y):
        print(f'Fold {fold_number}')
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # Clone the model to ensure fresh parameters
        model_clone = clone(model)
        
        # Fit the model
        model_clone.fit(X_train, y_train)
        
        # Predict on training and validation sets
        y_train_pred = model_clone.predict(X_train)
        y_val_pred = model_clone.predict(X_val)
        
        # Calculate MCC scores
        train_mcc = matthews_corrcoef(y_train, y_train_pred)
        val_mcc = matthews_corrcoef(y_val, y_val_pred)
        
        print(f'Training MCC: {train_mcc:.4f}')
        print(f'Validation MCC: {val_mcc:.4f}\n')
        
        train_mcc_scores.append(train_mcc)
        val_mcc_scores.append(val_mcc)
        
        # Predict on test data
        test_preds += model_clone.predict_proba(test_data)[:, 1]
        
        fold_number += 1
    
    # Average test predictions
    test_preds /= n_splits
    
    # Average MCC scores
    avg_train_mcc = np.mean(train_mcc_scores)
    avg_val_mcc = np.mean(val_mcc_scores)
    
    print(f'Average Training MCC: {avg_train_mcc:.4f}')
    print(f'Average Validation MCC: {avg_val_mcc:.4f}')
    
    return avg_val_mcc, test_preds


# # <p style="background-color: #21d179; color: #960206; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #960206; border: 5px dotted #960206;">Train XGBoost Model</p>
# 

# In[ ]:


# Initialize XGBoost model
xgboost_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Dictionary to store model performances
model_performance = {}

# Train XGBoost Model
xgboost_val_mcc, xgboost_test_preds = k_fold_model_trainer(xgboost_model, X, y, skf, test)
model_performance['XGBoost'] = {'val_mcc': xgboost_val_mcc, 'test_preds': xgboost_test_preds}


# # <p style="background-color: #21d179; color: #960206; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #960206; border: 5px dotted #960206;">Lets Train and Evaluate XGBoost Model with Best parameters for improving the mcc</p>
# 

# In[ ]:


from xgboost import XGBClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split

# Define the best hyperparameters found by Optuna
best_params = {
    'learning_rate': 0.035566934845282595,
    'max_depth': 14,
    'n_estimators': 300,
    'subsample': 0.6934347461494363,
    'colsample_bytree': 0.5491621501191348,
    'gamma': 0.0022153348155456998,
    'reg_alpha': 0.0020231866652905533,
    'reg_lambda': 0.01164512632039466
}

# Initialize the XGBoost model with the best hyperparameters
model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss', random_state=42)

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on the validation set
y_val_pred = model.predict(X_val)

# Calculate the validation MCC
val_mcc = matthews_corrcoef(y_val, y_val_pred)


# In[ ]:


# Train XGBoost with best hyperparameters from Optuna
best_xgboost_model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss', random_state=42)

# Perform K-Fold cross-validation with the best model
best_xgboost_val_mcc, best_xgboost_test_preds = k_fold_model_trainer(best_xgboost_model, X, y, skf, test)

# Prepare final predictions
final_predictions = np.where(best_xgboost_test_preds > 0.5, 1, 0)
final_predictions_labels = label_encoder.inverse_transform(final_predictions)


# # <p style="background-color: #21d179; color: #960206; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #960206; border: 5px dotted #960206;">Submission File</p>
# 

# In[ ]:


# Create and save the submission file
submission = pd.DataFrame({'id': test.index, 'class': final_predictions_labels})
submission.to_csv('submission.csv', index=False)


# In[ ]:


submission.head()


# # <p style="background-color: #21d179; color: #960206; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #960206; border: 5px dotted #960206;">Conclusion</p>
# 

# * This is the Binary Prediction of Poisonous Mushrooms Competition Data.
# * The aim to take this competition is to predict whether a mushroom is edible or poisonous based on its physical characteristics.
# * For this I firstly concatenate the Train Data of competition with secondary-mushroom-dataset-data-set in order to enhance the Data quality and diversity and to improve the mcc score.
# * Then I take a look at the Duplicates of the Data. There is no duplicates present in competition data * but when i combine the train and original data then I found the Duplicates so, I Drop the Duplicates.
# * Then I Impute the missing values of numeric and categorical columns.
# * Then I apply the KFold Cross-Validation and Train the XGBoost Model.
# * Then in order to improve mcc score further I train the model again on best parameters and get the best result.
# * Then I create the submission File

# # <p style="background-color: #21d179; color: #960206; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #960206; border: 5px dotted #960206;">Thankyou for taking the time to explore my notebook</p>
# 
