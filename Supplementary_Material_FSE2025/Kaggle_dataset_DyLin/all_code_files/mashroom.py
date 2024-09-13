#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import warnings

from sklearn.metrics import accuracy_score, classification_report

from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore')


# In[ ]:


df_train = pd.read_csv('/kaggle/input/playground-series-s4e8/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s4e8/test.csv')


# In[ ]:


def styled_heading(text):
     return f"""
    <div style=" padding:20px; border-radius:10px; margin-bottom:20px;">
        <h1 style="color:#2E86C1; text-align:center; font-size:36px; font-weight:bold;">{text}</h1>
        <p style="font-size:16px;"></p>
    </div>
    """

def print_error(message):
    display(HTML(styled_heading("Error")))
    print(f"An error occurred: {message}")

# Helper function to generate colored horizontal line
def colored_line(color='#323c6a'):
    return ""

def print_dataset_analysis(train_dataset, test_dataset, n_top=5, heading_color='#323c6a', line_color='#323c6a'):
    try:
        # Printing top values
        train_heading = styled_heading(f"Top {n_top} rows of Training Dataset")
        test_heading = styled_heading(f"Top {n_top} rows of Test Dataset")

        display(HTML(colored_line(line_color)))
        display(HTML(train_heading))
        display(HTML(colored_line(line_color)))
        display(HTML(train_dataset.head(n_top).to_html()))

        display(HTML(colored_line(line_color)))
        display(HTML(test_heading))
        display(HTML(colored_line(line_color)))
        display(HTML(test_dataset.head(n_top).to_html()))
        
        # Printing dataset summary
        summary_heading = styled_heading("Summary of Dataset")
        display(HTML(colored_line(line_color)))
        display(HTML(summary_heading))
        display(HTML(colored_line(line_color)))
        display(HTML(train_dataset.describe().to_html()))

        # Printing null values
        null_heading = styled_heading("Null Values in Datasets")
        
        train_null_count = train_dataset.isnull().sum()
        test_null_count = test_dataset.isnull().sum()

        display(HTML(colored_line(line_color)))
        display(HTML(null_heading))
        display(HTML(colored_line(line_color)))
        display(HTML("<h3>Training Dataset:</h3>"))
        if train_null_count.sum() == 0:
            display(HTML("<p>No null values in the training dataset.</p>"))
        else:
            display(HTML(train_null_count[train_null_count > 0].to_frame().to_html()))
            display(HTML("<p>These are the null values.</p>"))

        display(HTML("<h3>Test Dataset:</h3>"))
        if test_null_count.sum() == 0:
            display(HTML("<p>No null values in the test dataset.</p>"))
        else:
            display(HTML(test_null_count[test_null_count > 0].to_frame().to_html()))
            display(HTML("<p>These are the null values.</p>"))

        # Printing duplicate values
        duplicate_heading = styled_heading("Duplicate Values in Datasets")
        
        train_duplicates = train_dataset.duplicated().sum()
        test_duplicates = test_dataset.duplicated().sum()

        display(HTML(colored_line(line_color)))
        display(HTML(duplicate_heading))
        display(HTML(colored_line(line_color)))
        display(HTML("<h3>Training Dataset:</h3>"))
        display(HTML(f"<p>{train_duplicates} duplicate rows</p>"))

        display(HTML("<h3>Test Dataset:</h3>"))
        display(HTML(f"<p>{test_duplicates} duplicate rows</p>"))
        
        # Printing number of rows and columns
        shape_heading = styled_heading("Number of Rows and Columns")
        display(HTML(colored_line(line_color)))
        display(HTML(shape_heading))
        display(HTML(colored_line(line_color)))
        display(HTML("<h3>Training Dataset:</h3>"))
        display(HTML(f"<p>Rows: {train_dataset.shape[0]}, Columns: {train_dataset.shape[1]}</p>"))
        display(HTML("<h3>Test Dataset:</h3>"))
        display(HTML(f"<p>Rows: {test_dataset.shape[0]}, Columns: {test_dataset.shape[1]}</p>"))

    except Exception as e:
        print_error(str(e))

def print_unique_values(test_dataset, heading_color='#323c6a', line_color='#323c6a'):
    try:
        unique_values_heading = styled_heading("Unique Values in Training Dataset")
        
        display(HTML(colored_line(line_color)))
        display(HTML(unique_values_heading))
        display(HTML(colored_line(line_color)))
        
        unique_values_table = "<table border='1'><tr><th>Column Name</th><th>Data Type</th><th>Unique Values</th></tr>"
        
        for column in test_dataset.columns:
            unique_values = test_dataset[column].unique()[:7]  # Taking at least 7 unique values
            unique_values_str = ', '.join(map(str, unique_values))
            data_type = test_dataset[column].dtype
            unique_values_table += f"<tr><td>{column}</td><td>{data_type}</td><td>{unique_values_str}</td></tr>"
        
        unique_values_table += "</table>"
        display(HTML(unique_values_table))
    
    except Exception as e:
        print_error(str(e))


# In[ ]:




# In[ ]:


df_train.isnull().sum()


# In[ ]:


df_train.info()


# In[ ]:


# Impute missing values in 'cap-diameter' using the median
df_train['cap-diameter'].fillna(df_train['cap-diameter'].median(), inplace=True)
df_test['cap-diameter'].fillna(df_test['cap-diameter'].median(), inplace=True)
df_test['stem-height'].fillna(df_test['stem-height'].median(), inplace=True)


# In[ ]:


# Select categorical columns
categorical_cols = df_train.select_dtypes(include=['object']).columns

# Calculate missing percentage for categorical columns in the training set
missing_perc_train = df_train[categorical_cols].isnull().mean() * 100

# Identify columns to impute and columns to replace with 'not present'
cols_to_impute = missing_perc_train[missing_perc_train < 50].index
cols_to_np = missing_perc_train[missing_perc_train >= 50].index

# Impute missing values in the training dataset
for col in cols_to_impute:
    df_train[col].fillna(df_train[col].mode()[0], inplace=True)
for col in cols_to_np:
    df_train[col].fillna('not present', inplace=True)

# Ensure the test set columns match the train set columns
# Filter columns in df_test to match categorical_cols
filtered_categorical_cols = [col for col in categorical_cols if col in df_test.columns]

# Calculate missing percentage for categorical columns in the test set
missing_perc_test = df_test[filtered_categorical_cols].isnull().mean() * 100

# Impute missing values in the test dataset
for col in cols_to_impute:
    if col in df_test.columns:
        df_test[col].fillna(df_train[col].mode()[0], inplace=True)
for col in cols_to_np:
    if col in df_test.columns:
        df_test[col].fillna('not present', inplace=True)


# In[ ]:


# # Identify columns with more than 70% missing values
# missing_threshold = 0.7
# missing_train = df_train.isnull().mean()
# columns_to_drop = missing_train[missing_train > missing_threshold].index.tolist()

# # Drop columns (except 'class')
# columns_to_drop = [col for col in columns_to_drop if col != 'class']
# df_train.drop(columns=columns_to_drop, inplace=True)
# df_test.drop(columns=columns_to_drop, inplace=True)

# print(f'Columns dropped: {columns_to_drop}')

# # Identify numerical columns
# numerical_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()

# # Fill missing numerical values with median
# for col in numerical_cols:
#     median = df_train[col].median()
#     df_train[col].fillna(median, inplace=True)
#     df_test[col].fillna(median, inplace=True)


# In[ ]:


# # Identify categorical columns
# categorical_cols = df_train.select_dtypes(include=[object]).columns.tolist()

# # Ensure 'class' column is not removed or processed
# for col in categorical_cols:
#     if col != 'class':
#         # Fill missing values with 'not_present'
#         df_train[col].fillna('not_present', inplace=True)
#         df_test[col].fillna('not_present', inplace=True)
        
#         # Combine rare categories (frequency less than 1%)
#         freq = df_train[col].value_counts(normalize=True)
#         rare_categories = freq[freq < 0.01].index
#         df_train[col] = df_train[col].replace(rare_categories, 'infrequent')
#         df_test[col] = df_test[col].replace(rare_categories, 'infrequent')


# In[ ]:


from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
# Exclude 'class' from categorical columns
categorical_cols = [col for col in categorical_cols if col != 'class']

# Combine train and test data only for the categorical columns
combined = pd.concat([df_train[categorical_cols], df_test[categorical_cols]], axis=0)

# Ordinal Encoding
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
combined_encoded = encoder.fit_transform(combined)

# Split back to train and test
train_encoded = combined_encoded[:len(df_train)]
test_encoded = combined_encoded[len(df_train):]

# Replace original categorical columns with encoded ones
df_train[categorical_cols] = train_encoded
df_test[categorical_cols] = test_encoded


# In[ ]:


# Features and target
X = df_train.drop('class', axis=1)
y = df_train['class']

# Label encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Check shapes


# In[ ]:


from sklearn.model_selection import StratifiedKFold, train_test_split
X_train, X_test, y_train, y_test = train_test_split(X , y ,test_size=0.2,random_state=5 )


# In[ ]:


from xgboost import XGBClassifier

# Create an XGBoost classifier
xgb_model = XGBClassifier()

# Fit the model on your data
xgb_model.fit(X_train , y_train)


# In[ ]:


y_pred = xgb_model.predict(X_test)


# In[ ]:


from sklearn.metrics import matthews_corrcoef, classification_report

# Assuming y_test and y_pred are your true and predicted labels
mcc = matthews_corrcoef(y_test, y_pred)


# In[ ]:


# import numpy as np
# import pandas as pd

# # Assuming `y_pred` are your model's predicted probabilities
# final_predictions = np.where(y_pred > 0.5, 1, 0)  # Convert probabilities to binary predictions

# # Assuming `label_encoder` was used to encode the 'e' and 'p' labels during training
# final_predictions_labels = label_encoder.inverse_transform(final_predictions)  # Convert to 'e' or 'p'

# # Create a DataFrame for submission
# submission = pd.DataFrame({'id': df_test.index, 'class': final_predictions_labels})

# # Save the submission file
# submission.to_csv('submission.csv', index=False)
# print("Final submission file created successfully!")

