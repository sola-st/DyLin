#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPool2D, BatchNormalization, Dropout
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization,LeakyReLU
from keras.utils import to_categorical
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer, LabelEncoder,OneHotEncoder,QuantileTransformer
from keras.optimizers import Adam
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train = pd.read_csv('/kaggle/input/playground-series-s4e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e8/test.csv')


# In[ ]:


train.head()


# In[ ]:




# In[ ]:


(train.isnull().sum()/3116945)*100


# In[ ]:


def remove_highly_missing_columns(df, threshold=30):
    # Calculate the percentage of missing values for each column
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    
    # Select columns with more than the threshold percentage of missing values
    columns_to_drop = missing_percentage[missing_percentage > threshold].index
    
    # Drop the selected columns from the DataFrame
    df_cleaned = df.drop(columns=columns_to_drop)
    
    return df_cleaned

train = remove_highly_missing_columns(train, threshold=30)
test = remove_highly_missing_columns(test, threshold=30)


# In[ ]:


train.isnull().sum().sum()


# In[ ]:


# warnings.filterwarnings("ignore", category=UserWarning)
def fill_missing_values(df):
    # Fill missing values in categorical columns with the mode
    for column in df.select_dtypes(include=['object', 'category']).columns:
        df[column].fillna(df[column].mode()[0], inplace=True)
    
    # Fill missing values in numerical columns with the mean
    for column in df.select_dtypes(include=['number']).columns:
        df[column].fillna(df[column].mean(), inplace=True)
    
    return df

train = fill_missing_values(train)
test = fill_missing_values(test)


# In[ ]:


train.isnull().sum().sum()


# In[ ]:


class_counts = train['class'].value_counts().sort_index()
labels = ["Edible", "Poisonous"]
colors = plt.get_cmap('Blues')(np.linspace(0.2, 0.7, len(labels)))
plt.figure(figsize=(6, 4))
plt.pie(class_counts,  labels=labels,colors=colors, autopct='%.2f%%')
plt.axis('equal')

plt.show()


# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define feature lists
Numerical_Features = ['cap-diameter', 'stem-height', 'stem-width']
Categorical_Features = [
    'cap-shape', 'cap-surface', 'cap-color', 'does-bruise-or-bleed',
    'gill-attachment', 'gill-color', 'stem-color', 'has-ring',
    'ring-type', 'habitat', 'season'
]

# Convert categorical features to string to avoid any mixed-type issues
for col in Categorical_Features:
    train[col] = train[col].astype(str)

    
    
# Preprocessor definition
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), Numerical_Features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), Categorical_Features)
    ]
)


# In[ ]:


# Drop 'id' column from the training data
train = train.drop(columns='id')

# Extract features (X) and target (y)
X = train.drop(columns=['class'])
y = train['class']

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Verify the class distribution


# In[ ]:


# Preprocess the features
X_preprocessed = preprocessor.fit_transform(X)

# Split the preprocessed data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y_encoded, test_size=0.2, random_state=42)

# Build the deep learning model
model = Sequential()
model.add(Dense(512, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[ ]:


# Train the model
model.fit(X_train, y_train, epochs=2, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
y_pred_test = (model.predict(X_test) > 0.5).astype("int32")
accuracy = accuracy_score(y_test, y_pred_test)


# 0.9878

# In[ ]:


# Process the test set for final prediction
for col in Categorical_Features:
    test[col] = test[col].astype(str)

test_ids = test['id']
test = test.drop(columns='id')
X_test_final = preprocessor.transform(test)

# Predict on the final test set
y_pred_final = (model.predict(X_test_final) > 0.5).astype("int32")

# Create a DataFrame with the test IDs and predicted classes
output = pd.DataFrame({'id': test_ids,
                       'class': label_encoder.inverse_transform(y_pred_final.ravel())})

# Save the predictions to a CSV file
output.to_csv('submission.csv', index=False)


