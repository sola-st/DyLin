#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[ ]:


import numpy as np
import pandas as pd


# # Read & Check the datasets

# ### Training set

# In[ ]:


training_set_path = '/kaggle/input/playground-series-s4e8/train.csv'
train_dataset = pd.read_csv(training_set_path, index_col = 'id')
train_dataset.head()


# ### Testing set

# In[ ]:


testing_set_path = '/kaggle/input/playground-series-s4e8/test.csv'
test_dataset = pd.read_csv(testing_set_path, index_col = 'id')
test_dataset.head()


# In[ ]:


test_dataset.shape


# ### Check the number of all records in the dataset 

# In[ ]:




# In[ ]:




# ### View the information of the datasets (missing values, duplicates)

# In[ ]:


# Check for the categorical features for training set
train_dataset.info()


# #### The dataset contains many categorical features

# In[ ]:


# Check whether the dataset has missing values or not
missing_values = train_dataset.isnull().sum()
missing_values


# #### There are many missing values in the training set

# In[ ]:


# Check the missing values of the testing values
missing_values = test_dataset.isnull().sum()
missing_values 


# #### There are many missing values in the testing set

# In[ ]:


#let's get the missing values' precentage at each feature
def get_missing_values_precentage(dataset):
    m = len(dataset)
    for i in dataset.columns:
        print(f'The missing values precentage of {i} feature is {(dataset[i].isnull().any() / m) * 100} %')


# In[ ]:


get_missing_values_precentage(train_dataset)


# In[ ]:


# Check whether the training dataset has duplicates or not
is_duplicated_train_dataset = train_dataset.duplicated().any()
is_duplicated_train_dataset


# #### There is no duplicated values in the training set

# In[ ]:


# Check whether the testing datasets has duplicates or not
is_duplicated_test_dataset = test_dataset.duplicated().any()
is_duplicated_test_dataset


# In[ ]:


test_dataset.shape


# #### There is no duplicated values in the testing set

# In[ ]:


y = train_dataset['class']


# In[ ]:


X = train_dataset.drop(['class'], axis='columns')


# In[ ]:


test_dataset.shape


# # Handle the dataset

# ### 1. Handle Missing values in the categorical features using mode method

# In[ ]:


def handle_missing_val_in_cat(dataset):
    # first we need to get all categorical features
    cols = dataset.columns
    num_cols = dataset._get_numeric_data().columns
    cat_features = list(set(cols) - set(num_cols))
    
    for i in cat_features:
        missing = dataset[i].isnull().any()
        if missing != 0: #contain missing values
            mode_value = dataset[i].mode()[0]
            # Fill the missing values with mode
            dataset[i].fillna(mode_value, inplace=True)
            
    return dataset


# ### 2. Handle missing values in numerical features

# In[ ]:


def handle_missing_val_in_num(dataset):
    return dataset.interpolate()


# ### 3. Handle Categorical Features

# In[ ]:


# List of categorical columns to process
categorical_columns = ['cap-shape', 'cap-surface', 'cap-color','does-bruise-or-bleed', 'gill-attachment', 'gill-spacing'
                      ,'gill-color', 'stem-root', 'stem-surface','stem-color','veil-type','veil-color','has-ring',
                       'ring-type','spore-print-color','habitat'
                      ] 


# In[ ]:


numerical_columns = ['cap-diameter', 'stem-height', 'stem-width']  


# In[ ]:


columns_to_clean = [
    'cap-shape', 'cap-surface', 'cap-color', 'does-bruise-or-bleed',
    'gill-attachment', 'gill-spacing', 'gill-color', 'stem-root',
    'stem-surface', 'stem-color', 'veil-type', 'veil-color', 'has-ring',
    'ring-type', 'spore-print-color', 'habitat'
]


# In[ ]:


valid_categories = {
    'cap-shape': ['f', 'x', 'p', 'b', 'o', 'c', 's'],
    'cap-surface': ['s', 'h', 'y', 'l'],
    'cap-color': ['u', 'o', 'b', 'g', 'w', 'n', 'e', 'y', 'r', 'p', 'k'],
    'does-bruise-or-bleed': ['f', 't'],
    'gill-attachment': ['a', 'x', 's', 'd'],
    'gill-spacing': ['c', 'd'],
    'gill-color': ['w', 'n', 'g', 'k'],
    'stem-root': ['b', 'c', 'r', 's', 'f'],
    'stem-surface': ['y', 's', 't', 'g'],
    'stem-color': ['w', 'o', 'n', 'y', 'e'],
    'veil-type': ['u', 'd'],
    'veil-color': ['n', 'w', 'k', 'y'],
    'has-ring': ['f', 't'],
    'ring-type': ['f', 'z', 'e', 'p'],
    'spore-print-color': ['k', 'w', 'p', 'n'],
    'habitat': ['d', 'l', 'g', 'h', 'p', 'm', 'u']
}


# In[ ]:


# Replace unexpected values with NaN
for column in columns_to_clean:
    X[column] = X[column].apply(lambda x: x if x in valid_categories[column] else np.nan)
    test_dataset[column] = test_dataset[column].apply(lambda x: x if x in valid_categories[column] else np.nan)


# In[ ]:


X = handle_missing_val_in_cat(X)


# In[ ]:


X = handle_missing_val_in_num(X)


# In[ ]:


X.isnull().any()


# In[ ]:


test_dataset = handle_missing_val_in_cat(test_dataset)


# In[ ]:


test_dataset = handle_missing_val_in_num(test_dataset)


# In[ ]:


test_dataset.isnull().any()


# In[ ]:


# X, test
X.shape, test_dataset.shape


# In[ ]:


from sklearn.preprocessing import LabelEncoder
categorical_columns_encoded = [value+"_encoded" for value in categorical_columns]


# In[ ]:


label_encoder = LabelEncoder()

# Process each categorical column
for column in categorical_columns:
    # Convert the column to string to handle mixed types
    X[column] = X[column].astype(str)
    
    # Apply Label Encoding
    X[column + '_encoded'] = label_encoder.fit_transform(X[column])
    
    # Convert the column to string to handle mixed types
    test_dataset[column] = test_dataset[column].astype(str)
    
    # Apply Label Encoding
    test_dataset[column + '_encoded'] = label_encoder.fit_transform(test_dataset[column])


# In[ ]:


X.shape, test_dataset.shape


# #### Encoding the label (class) 

# In[ ]:


le = LabelEncoder()
y = le.fit_transform(y)


# In[ ]:


y.shape


# In[ ]:


columns = numerical_columns + categorical_columns_encoded
X = X[columns]


# In[ ]:


test_dataset = test_dataset[columns]


# In[ ]:


X.head()


# In[ ]:


test_dataset.head()


# # Build the model

# In[ ]:


from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
# Apply Robust Scaling to each numerical column
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
test_dataset[numerical_columns] = scaler.fit_transform(test_dataset[numerical_columns])


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


X_train


# #### 1. Neural Network

# In[ ]:


import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.initializers import RandomNormal


# In[ ]:


model = tf.keras.models.Sequential()

# Neurons, dropouts
model.add(tf.keras.layers.Dense(units=64, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),  input_dim=X_train.shape[1] ))
model.add(tf.keras.layers.BatchNormalization()),
model.add(tf.keras.layers.Dense(units=128, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01), activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization()),
model.add(tf.keras.layers.Dense(units=32, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),  activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization()),
model.add(tf.keras.layers.Dense(units=16,kernel_initializer=RandomNormal(mean=0.0, stddev=0.01), activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization()),
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# In[ ]:


model.summary()


# In[ ]:


model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.1),
    metrics=['accuracy'],
)


# In[ ]:


history = model.fit(
    X_train,
    y_train,
    batch_size=5000,
    epochs=10,
    validation_data=(X_test, y_test),
    
)


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(10)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[ ]:


prediction = model.predict(test_dataset)


# In[ ]:


prediction


# In[ ]:


for i in range(len(prediction)):
    if prediction[i] >= 0.5:
        prediction[i] = int(1)
    else:
        prediction[i] = int(0)


# In[ ]:


prediction


# In[ ]:


y_pred = prediction.astype(np.int32)


# In[ ]:


y_pred


# In[ ]:


test_pred_class = le.inverse_transform(y_pred)


# In[ ]:


test_pred_class


# In[ ]:


len(test_pred_class)


# In[ ]:


submit = pd.read_csv('/kaggle/input/playground-series-s4e8/sample_submission.csv')


# In[ ]:


submit.shape


# In[ ]:


submit['class'] = test_pred_class


# In[ ]:


submit.to_csv('/kaggle/working/submit.csv', index = False)


# In[ ]:


pd.read_csv('/kaggle/working/submit.csv')


# In[ ]:




