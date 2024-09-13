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


# This notebook draws inspiration from: https://www.kaggle.com/code/annastasy/ps4e8-data-cleaning-and-eda-of-mushrooms
# 
# I've incorporated some elements of her code to expedite the completion of this project within my two-day deadline.
# 
# As this is only my second Kaggle notebook, I appreciate your kindness and support. :)))

# # **Read the data and conduct a preliminary data analysis.**

# In[ ]:


import pandas as pd
import numpy as np 
from scipy import stats
import seaborn as sns 
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder,QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef


# In[ ]:


df_train = pd.read_csv('/kaggle/input/playground-series-s4e8/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s4e8/test.csv')

df_train


# In[ ]:


df_train.dtypes


# In[ ]:


df_train.info()


# In[ ]:


df_train.describe()


# In[ ]:


df_test


# In[ ]:


df_test.info()


# In[ ]:


df_test.describe()


# In[ ]:


df_train.duplicated().sum()


# In[ ]:


plt.figure(figsize=(12,8))
plt.title("Missing Values")
sns.heatmap(df_train.isnull(), cbar=False, yticklabels=False);


# **We discovered that the data contains numerous categorical variables and a significant amount of missing values. To simplify the analysis, we converted categorical variables into numerical representations. For missing values in categorical columns, we employed the mode (most frequent value). In numerical columns, we filled missing values with the column's mean.**

# In[ ]:


df_train_cleaned = df_train.copy()
df_test_cleaned = df_test.copy()

df_train_cleaned


# In[ ]:


df_train_cleaned = df_train_cleaned.drop(['id'], axis=1)

df_train_cleaned


# In[ ]:


target_column = 'class'
categorical_columns = df_train_cleaned.select_dtypes(include=['object']).columns.drop(target_column)
numerical_columns = df_train_cleaned.select_dtypes(exclude=['object']).columns.drop(target_column, errors='ignore')




# In[ ]:


for column in categorical_columns:
    num_unique = df_train_cleaned[column].nunique()
    print(f"'{column}' has {num_unique} unique categories.")


# In[ ]:


# Print top 10 unique value counts for each categorical column
for column in categorical_columns:
    print(f"\nTop value counts in '{column}':\n{df_train_cleaned[column].value_counts().head(10)}")


# # **Preprocessing**

# In[ ]:


# Define a function to identify and replace infrequent categories
def replace_infrequent_categories(df, column, threshold=70):
    value_counts = df[column].value_counts()
    infrequent = value_counts[value_counts <= threshold].index
    df[column] = df[column].apply(lambda x: "Unknown" if x in infrequent else x)
    return df

# Handle invalid values and infrequent categories for all categorical columns
for col in categorical_columns:
    df_train_cleaned = replace_infrequent_categories(df_train_cleaned, col)
    df_test_cleaned = replace_infrequent_categories(df_test_cleaned, col)

# Print out number of unique columns after a replacement
for column in categorical_columns:
    num_unique = df_train_cleaned[column].nunique()
    print(f"'{column}' has {num_unique} unique categories.")


# In[ ]:


# Compute medians for numerical columns in the training set
medians = df_train_cleaned[numerical_columns].median()

# Fill missing values in the training and testing sets
df_train_cleaned[numerical_columns] = df_train_cleaned[numerical_columns].fillna(medians)
df_test_cleaned[numerical_columns] = df_test_cleaned[numerical_columns].fillna(medians)



# In[ ]:


# Impute any missing values with 'Unknown'
df_train_cleaned = df_train_cleaned.fillna("Unknown")
df_test_cleaned = df_test_cleaned.fillna("Unknown")


# In[ ]:




# In[ ]:


df_train_cleaned = df_train_cleaned.drop_duplicates()


# In[ ]:


df_train_cleaned


# In[ ]:


df_test_cleaned


# In[ ]:


# convert class colum into {0,1}

label_encoder = LabelEncoder()
df_train_cleaned['class'] = label_encoder.fit_transform(df_train_cleaned['class'])

df_train_cleaned


# In[ ]:


for col in categorical_columns:
    if 'Unknown' in df_train_cleaned[col].values:
        mode_value = df_train_cleaned[col][df_train_cleaned[col] != 'Unknown'].mode()[0]
        df_train_cleaned[col].replace('Unknown', mode_value, inplace=True)
        df_test_cleaned[col].replace('Unknown', mode_value, inplace=True)
        
df_train_cleaned


# In[ ]:


df_train_cleaned[['cap-shape', 'class']].groupby(['cap-shape'], as_index=False).mean()


# In[ ]:


# Define the mapping for 'cap-shape'
cap_shape_mapping = {
    "b": 1, "c": 2, "f": 3, "o": 4,
    "p": 5, "s": 6, "x": 7
}

# Convert 'cap-shape' in both training and test datasets
for dataset in [df_train_cleaned, df_test_cleaned]:
    # Apply the mapping to 'cap-shape' column
    dataset['cap-shape'] = dataset['cap-shape'].map(cap_shape_mapping)
    
    # Fill any NaN values (resulting from unmapped values) with 0
    dataset['cap-shape'] = dataset['cap-shape'].fillna(0)

# Print a summary to confirm changes


# In[ ]:


df_train_cleaned[["cap-surface", "class"]].groupby(['cap-surface'], as_index=False).mean()


# In[ ]:


# Define the mapping for 'cap-shape'
cap_surface_mapping = {
    "d": 1, "e": 2, "f": 3, "g": 4,"h" : 5, "i":6,
    "k": 7, "l": 8, "n": 9,"s":10,"t":11,"w":12,"y":13
}

# Convert 'cap-shape' in both training and test datasets
for dataset in [df_train_cleaned, df_test_cleaned]:
    # Apply the mapping to 'cap-shape' column
    dataset['cap-surface'] = dataset['cap-surface'].map(cap_surface_mapping)
    
    # Fill any NaN values (resulting from unmapped values) with 0
    dataset['cap-surface'] = dataset['cap-surface'].fillna(0)

df_train_cleaned


# In[ ]:


df_train_cleaned[["cap-color", "class"]].groupby(['cap-color'], as_index=False).mean()


# In[ ]:


# Define the mapping for 'cap-shape'
cap_colour_mapping = {
    "b": 1, "e": 2, "g": 3, "k": 4,"l" : 5, "n":6,"o":7,
    "p": 8, "r": 9, "u": 10,"w":11,"y":12
}

# Convert 'cap-shape' in both training and test datasets
for dataset in [df_train_cleaned, df_test_cleaned]:
    # Apply the mapping to 'cap-shape' column
    dataset['cap-color'] = dataset['cap-color'].map(cap_colour_mapping)
    
    # Fill any NaN values (resulting from unmapped values) with 0
    dataset['cap-color'] = dataset['cap-color'].fillna(0)

df_train_cleaned


# In[ ]:


df_train_cleaned[["does-bruise-or-bleed", "class"]].groupby(['does-bruise-or-bleed'], as_index=False).mean()


# In[ ]:


bOd_mapping = {"f": 1, "t": 0}

# Convert 'cap-shape' in both training and test datasets
for dataset in [df_train_cleaned, df_test_cleaned]:
    # Apply the mapping to 'cap-shape' column
    dataset['does-bruise-or-bleed'] = dataset['does-bruise-or-bleed'].map(bOd_mapping)
    
    # Fill any NaN values (resulting from unmapped values) with 0
    dataset['does-bruise-or-bleed'] = dataset['does-bruise-or-bleed'].fillna(0)

df_train_cleaned.head(10)


# In[ ]:


df_train_cleaned[["gill-attachment", "class"]].groupby(['gill-attachment'], as_index=False).mean()


# In[ ]:


ga_mapping = {
    "a": 1, "c": 2, "d": 3, "e": 4,"f" : 5, "p":6,"s":7,
    "x": 8
}

# Convert 'cap-shape' in both training and test datasets
for dataset in [df_train_cleaned, df_test_cleaned]:
    # Apply the mapping to 'cap-shape' column
    dataset['gill-attachment'] = dataset['gill-attachment'].map(ga_mapping)
    
    # Fill any NaN values (resulting from unmapped values) with 0
    dataset['gill-attachment'] = dataset['gill-attachment'].fillna(0)

df_train_cleaned.head(10)


# In[ ]:


df_train_cleaned[["gill-spacing", "class"]].groupby(['gill-spacing'], as_index=False).mean()


# In[ ]:


gs_mapping = {
    "c": 1, "d": 2, "f": 3
}

# Convert 'cap-shape' in both training and test datasets
for dataset in [df_train_cleaned, df_test_cleaned]:
    # Apply the mapping to 'cap-shape' column
    dataset['gill-spacing'] = dataset['gill-spacing'].map(gs_mapping)
    
    # Fill any NaN values (resulting from unmapped values) with 0
    dataset['gill-spacing'] = dataset['gill-spacing'].fillna(0)

df_train_cleaned


# In[ ]:


df_train_cleaned[["gill-color", "class"]].groupby(['gill-color'], as_index=False).mean()


# In[ ]:


# Define the mapping for 'cap-shape'
gc_mapping = {
    "b": 1, "e": 2, "f": 3, "g": 4,"k" : 5, "n":6,"o":7,
    "p": 8, "r": 9, "u": 10,"w":11,"y":12
}

# Convert 'cap-shape' in both training and test datasets
for dataset in [df_train_cleaned, df_test_cleaned]:
    # Apply the mapping to 'cap-shape' column
    dataset['gill-color'] = dataset['gill-color'].map(gc_mapping)
    
    # Fill any NaN values (resulting from unmapped values) with 0
    dataset['gill-color'] = dataset['gill-color'].fillna(0)

df_train_cleaned


# In[ ]:


df_train_cleaned[["stem-root", "class"]].groupby(['stem-root'], as_index=False).mean()


# In[ ]:


# Define the mapping for 'cap-shape'
sr_mapping = {
    "b": 1, "c": 2, "f": 3, "r": 4,"s" : 5, 
}

# Convert 'cap-shape' in both training and test datasets
for dataset in [df_train_cleaned, df_test_cleaned]:
    # Apply the mapping to 'cap-shape' column
    dataset['stem-root'] = dataset['stem-root'].map(sr_mapping)
    
    # Fill any NaN values (resulting from unmapped values) with 0
    dataset['stem-root'] = dataset['stem-root'].fillna(0)

df_train_cleaned


# In[ ]:


df_train_cleaned[["stem-surface", "class"]].groupby(['stem-surface'], as_index=False).mean()


# In[ ]:


# Define the mapping for 'cap-shape'
ss_mapping = {
    "f": 1, "g": 2, "h": 3, "i": 4,"k" : 5, "s":6,"t":7,"y":8
}

# Convert 'cap-shape' in both training and test datasets
for dataset in [df_train_cleaned, df_test_cleaned]:
    # Apply the mapping to 'cap-shape' column
    dataset['stem-surface'] = dataset['stem-surface'].map(ss_mapping)
    
    # Fill any NaN values (resulting from unmapped values) with 0
    dataset['stem-surface'] = dataset['stem-surface'].fillna(0)

df_train_cleaned


# In[ ]:


df_train_cleaned[["stem-color", "class"]].groupby(['stem-color'], as_index=False).mean()


# In[ ]:


# Define the mapping for 'cap-shape'
sc_mapping = {
    "b": 1, "e": 2, "f": 3, "g": 4,"k" : 5, "l":6,"n":7,
    "o": 8, "p": 9, "r": 10,"u":11,"w":12,"y":13
}

# Convert 'cap-shape' in both training and test datasets
for dataset in [df_train_cleaned, df_test_cleaned]:
    # Apply the mapping to 'cap-shape' column
    dataset['stem-color'] = dataset['stem-color'].map(sc_mapping)
    
    # Fill any NaN values (resulting from unmapped values) with 0
    dataset['stem-color'] = dataset['stem-color'].fillna(0)

df_train_cleaned


# In[ ]:


df_train_cleaned[["veil-type", "class"]].groupby(['veil-type'], as_index=False).mean()


# Since the 'veil-type' feature only contained the value 'u', it was removed as it was unlikely to contribute significantly to improving the model's accuracy.

# In[ ]:


df_train_cleaned = df_train_cleaned.drop(['veil-type'], axis=1)
df_test_cleaned = df_test_cleaned.drop(['veil-type'], axis=1)


# In[ ]:


df_train_cleaned[["veil-color", "class"]].groupby(['veil-color'], as_index=False).mean()


# In[ ]:


# Define the mapping for 'cap-shape'
vc_mapping = {
    "e": 1, "h": 2, "n": 3, "u": 4,"w" : 5, "y":6
}

# Convert 'cap-shape' in both training and test datasets
for dataset in [df_train_cleaned, df_test_cleaned]:
    # Apply the mapping to 'cap-shape' column
    dataset['veil-color'] = dataset['veil-color'].map(vc_mapping)
    
    # Fill any NaN values (resulting from unmapped values) with 0
    dataset['veil-color'] = dataset['veil-color'].fillna(0)

df_train_cleaned


# In[ ]:


df_train_cleaned[["has-ring", "class"]].groupby(['has-ring'], as_index=False).mean()


# In[ ]:


# Define the mapping for 'cap-shape'
hr_mapping = {
    "t": 1, "f": 0,
}

# Convert 'cap-shape' in both training and test datasets
for dataset in [df_train_cleaned, df_test_cleaned]:
    # Apply the mapping to 'cap-shape' column
    dataset['has-ring'] = dataset['has-ring'].map(hr_mapping)
    
    # Fill any NaN values (resulting from unmapped values) with 0
    dataset['has-ringr'] = dataset['has-ring'].fillna(0)

df_train_cleaned


# In[ ]:


df_train_cleaned[["ring-type", "class"]].groupby(['ring-type'], as_index=False).mean()


# In[ ]:


# Define the mapping for 'cap-shape'
rt_mapping = {
    "e": 1, "f": 2, "g": 3, "l": 4,"m" : 5, "p":6,"r":7,
    "t": 8, "z": 9, 
}

# Convert 'cap-shape' in both training and test datasets
for dataset in [df_train_cleaned, df_test_cleaned]:
    # Apply the mapping to 'cap-shape' column
    dataset['ring-type'] = dataset['ring-type'].map(rt_mapping)
    
    # Fill any NaN values (resulting from unmapped values) with 0
    dataset['ring-type'] = dataset['ring-type'].fillna(0)

df_train_cleaned


# In[ ]:


df_train_cleaned[["spore-print-color", "class"]].groupby(['spore-print-color'], as_index=False).mean()


# In[ ]:


# Define the mapping for 'cap-shape'
spc_mapping = {
    "g": 1, "k": 2, "n": 3, "p": 4,"r" : 5, "u":6,"w":7,
}

# Convert 'cap-shape' in both training and test datasets
for dataset in [df_train_cleaned, df_test_cleaned]:
    # Apply the mapping to 'cap-shape' column
    dataset['spore-print-color'] = dataset['spore-print-color'].map(spc_mapping)
    
    # Fill any NaN values (resulting from unmapped values) with 0
    dataset['spore-print-color'] = dataset['spore-print-color'].fillna(0)

df_train_cleaned


# In[ ]:


df_train_cleaned[["habitat", "class"]].groupby(['habitat'], as_index=False).mean()


# In[ ]:


# Define the mapping for 'cap-shape'
h_mapping = {
    "d": 1, "g": 2, "h": 3, "l": 4,"m" : 5, "p":6,"u":7,"w":8
}

# Convert 'cap-shape' in both training and test datasets
for dataset in [df_train_cleaned, df_test_cleaned]:
    # Apply the mapping to 'cap-shape' column
    dataset['habitat'] = dataset['habitat'].map(h_mapping)
    
    # Fill any NaN values (resulting from unmapped values) with 0
    dataset['habitat'] = dataset['habitat'].fillna(0)

df_train_cleaned


# In[ ]:


df_train_cleaned[["season", "class"]].groupby(['season'], as_index=False).mean()


# In[ ]:


# Define the mapping for 'cap-shape'
s_mapping = {
    "a": 1, "s": 2, "u": 3, "w": 4}

# Convert 'cap-shape' in both training and test datasets
for dataset in [df_train_cleaned, df_test_cleaned]:
    # Apply the mapping to 'cap-shape' column
    dataset['season'] = dataset['season'].map(s_mapping)
    
    # Fill any NaN values (resulting from unmapped values) with 0
    dataset['season'] = dataset['season'].fillna(0)

df_train_cleaned


# # **Visualization**

# In[ ]:


for column in numerical_columns:
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=df_train_cleaned, x='class', y=column) 
    plt.title(f'Distribution of {column} by class')

    plt.tight_layout()
    plt.show()


# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Compute the correlation matrix
corr_matrix = df_train_cleaned.corr()

# Plot the correlation matrix using seaborn
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix,  center=0, linewidths=0.5)
plt.title('Correlation Matrix of Training Data with Target')
plt.show()


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt



# Calculate correlations with the target variable
correlation = df_train_cleaned.corr()['class'].sort_values(ascending=False)

# Plotting the correlation
plt.figure(figsize=(10, 6))
correlation.plot(kind='bar')
plt.title('Feature Correlation with Survived')
plt.xlabel('Feature')
plt.ylabel('Correlation Coefficient')
plt.show()


# Many features were found to have a limited impact on the class variable. While feature engineering could potentially improve performance, due to time constraints, I didn't explore those techniques in this analysis. :)

# In[ ]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

# Step 1: Separate the target variable from the features
X = df_train_cleaned.drop('class', axis=1)
y = df_train_cleaned['class']

# Step 2: Apply PCA
pca = PCA(n_components=2)  # Here, we keep 2 principal components for visualization
X_pca = pca.fit_transform(X)

# Display the amount of variance explained by each principal component

# Step 3 Visualize the PCA result
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=100)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Mario Dataset with Original')
plt.colorbar(label='class')
plt.show()


# In[ ]:


# # Step 1: Separate the target variable from the features
# X = df_train_cleaned.drop('class', axis=1)
# y = df_train_cleaned['class']

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_scaled = normalize(X_scaled)

# # Step 2: Apply PCA
# pca = PCA(n_components=2)  # Here, we keep 2 principal components for visualization
# X_pca = pca.fit_transform(X_scaled)

# # Display the amount of variance explained by each principal component
# print("Explained variance ratio:", pca.explained_variance_ratio_)

# # Step 3 Visualize the PCA result
# plt.figure(figsize=(8, 6))
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=100)
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title('PCA of Mario Dataset with Scale and Norm')
# plt.colorbar(label='class')
# plt.show()


# In[ ]:


# # Step 1: Separate the target variable from the features
# X = df_train_cleaned.drop('class', axis=1)
# y = df_train_cleaned['class']

# scaler = QuantileTransformer(output_distribution='normal')
# X_scaled = scaler.fit_transform(X)
# X_scaled = normalize(X_scaled)


# # Step 2: Apply PCA
# pca = PCA(n_components=2)  # Here, we keep 2 principal components for visualization
# X_pca = pca.fit_transform(X_scaled)

# # Display the amount of variance explained by each principal component
# print("Explained variance ratio:", pca.explained_variance_ratio_)

# # Step 3 Visualize the PCA result
# plt.figure(figsize=(8, 6))
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=100)
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title('PCA of Mario Dataset with QT')
# plt.colorbar(label='class')
# plt.show()


# In[ ]:


# # Step 1: Separate the target variable from the features
# X = df_train_cleaned.drop('class', axis=1)
# y = df_train_cleaned['class']

# scaler = PowerTransformer()
# X_scaled = scaler.fit_transform(X)

# # Step 2: Apply PCA
# pca = PCA(n_components=2)  # Here, we keep 2 principal components for visualization
# X_pca = pca.fit_transform(X_scaled)

# # Display the amount of variance explained by each principal component
# print("Explained variance ratio:", pca.explained_variance_ratio_)

# # Step 3 Visualize the PCA result
# plt.figure(figsize=(8, 6))
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=100)
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title('PCA of Mario Dataset with PT')
# plt.colorbar(label='class')
# plt.show()


# **I selected "PowerTransformer()" for proseccing the data.**

# # **Training and prediction**

# In[ ]:


from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest

# Prepare X and Y
X_train = df_train_cleaned.drop("class", axis=1)
Y_train = df_train_cleaned["class"]

scaler = PowerTransformer()
X_train = scaler.fit_transform(X)

isolation_forest = IsolationForest(contamination=0.024, random_state=42)
outlier_labels = isolation_forest.fit_predict(X_train)

non_outliers_mask = outlier_labels != -1
X_train = X_train[non_outliers_mask]
Y_train = Y_train[non_outliers_mask]

# Prepare X_test
X_test = df_test_cleaned.drop("id", axis=1).copy()
X_test = scaler.transform(X_test)  # Use transform for test data

# # Fit Random Forest model and perform cross-validation
random_forest = RandomForestClassifier()
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# # Cross-validation predictions
# Y_pred = cross_val_predict(random_forest, X_train, Y_train, cv=skf)

# # Calculate the Matthews Correlation Coefficient
# mcc = matthews_corrcoef(Y_train, Y_pred)

# # Output results
# print("Shape of X_train:", X_train.shape)
# print("Shape of Y_train:", Y_train.shape)
# print("Shape of X_test:", X_test.shape)
# print("Matthews Correlation Coefficient:", mcc)


# In[ ]:


random_forest.fit(X_train, Y_train)
test_preds = random_forest.predict(X_test)
test_preds = label_encoder.inverse_transform(test_preds)


# In[ ]:


output = pd.DataFrame({'id': df_test['id'],
                       'class': test_preds})

output.to_csv('submission.csv', index=False)

output.head()


# In[ ]:




