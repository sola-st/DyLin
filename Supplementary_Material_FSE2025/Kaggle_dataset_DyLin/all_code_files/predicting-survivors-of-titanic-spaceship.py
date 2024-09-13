#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.losses import BinaryCrossentropy


# Load, preview and check what preprocessing needed

# In[ ]:


data_train = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')
data_test = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')


# In[ ]:


# Preview first 5 rows
data_train.head()


# In[ ]:


data_test.head()


# In[ ]:


# Check for shape and number of non-null values
data_train.info()


# In[ ]:


data_test.info()


# In[ ]:


# check for total null values


# In[ ]:


# Check for consistency
data_train.describe()


# In[ ]:


data_test.describe()


# In[ ]:


def check_unique_values(df):
    columns_to_check = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
    # Get unique values for specific columns
    unique_values = {col: df[col].unique() for col in columns_to_check}
    # Print unique values for each specified column
    for col, values in unique_values.items():
        print(f"Unique values in column '{col}': {values}")


# In[ ]:


check_unique_values(data_train)


# In[ ]:


check_unique_values(data_test)


# In[ ]:


# Check for balancing


# TODO list for data preprocessing:
# 1. Drop following columns as they are not relevant to if passenger survived or not: 'PassengerId', 'Name'.
# 2. Change True and False values in binary categorical columns to 1 and 0
# 3. Handle 'Cabin' column
# 4. One hot encoding for categorical columns
# 5. Split training data into training and validation sets
# 6. Scaling

# In[ ]:


#TODO 1. Drop following columns as they are not relevant to if passenger survived or not: 'PassengerId', 'Name'.
data_train_dropped = data_train.drop('PassengerId', axis=1)
data_train_dropped = data_train_dropped.drop('Name', axis=1)
data_train_dropped.head()


# In[ ]:


data_train = data_train_dropped
data_train.shape


# In[ ]:


#TODO 2. Change True and False values in binary categorical columns to 1 and 0
col_to_change = ['CryoSleep', 'VIP', 'Transported']
for col in col_to_change:
    data_train[col] = data_train[col].replace({True: 1, False: 0})


# In[ ]:


data_train.head()


# In[ ]:


check_unique_values(data_train)


# In[ ]:


#TODO 3. Handle 'Cabin' column
data_train['Deck'] = data_train['Cabin'].str[0]  # Extract the first letter as the deck
data_train['Deck']


# In[ ]:


data_train['Deck'].unique()


# In[ ]:


data_train['Side'] = data_train['Cabin'].str[-1]  # Extract the last letter as the side
data_train['Side'].unique()


# In[ ]:


data_train = data_train.drop(columns=['Cabin'])


# In[ ]:


data_train.head()


# In[ ]:


def check_unique_values_new(df):
    columns_to_check = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
    # Get unique values for specific columns
    unique_values = {col: df[col].unique() for col in columns_to_check}
    # Print unique values for each specified column
    for col, values in unique_values.items():
        print(f"Unique values in column '{col}': {values}")
check_unique_values_new(data_train)


# In[ ]:


# Convert 'Side' column to binary values
data_train['Side'] = data_train['Side'].replace({'P': 1, 'S': 0})


# In[ ]:


check_unique_values_new(data_train)


# In[ ]:


# Handle the nulls
data_train['CryoSleep'] = data_train['CryoSleep'].astype(str)
data_train['VIP'] = data_train['VIP'].astype(str)
data_train['Side'] = data_train['Side'].astype(str)
data_train['HomePlanet'].fillna('Unknown', inplace=True)
data_train['Destination'].fillna('Unknown', inplace=True)
data_train['Deck'].fillna('Unknown', inplace=True)
data_train['CryoSleep'].fillna('Unknown', inplace=True)
data_train['VIP'].fillna('Unknown', inplace=True)
data_train['Side'].fillna('Unknown', inplace=True)


# In[ ]:


#TODO 4. One hot encoding for categorical columns 'HomePlanet', 'Destination', and 'Deck'
data_train = pd.get_dummies(data_train, columns=['CryoSleep', 'VIP', 'Side', 'HomePlanet', 'Destination', 'Deck'])
data_train.head()


# In[ ]:


data_train.columns


# In[ ]:


# Convert dummy variables (True/False) to 1/0
columns_to_convert = [
    'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
    'CryoSleep_0.0', 'CryoSleep_1.0', 'CryoSleep_nan',
    'VIP_0.0', 'VIP_1.0', 'VIP_nan', 'Side_0.0', 'Side_1.0',
    'Side_nan', 'HomePlanet_Earth', 'HomePlanet_Europa',
    'HomePlanet_Mars', 'HomePlanet_Unknown', 'Destination_55 Cancri e',
    'Destination_PSO J318.5-22', 'Destination_TRAPPIST-1e',
    'Destination_Unknown', 'Deck_A', 'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E',
    'Deck_F', 'Deck_G', 'Deck_T', 'Deck_Unknown'
]

# Replace True/False with 1/0
data_train[columns_to_convert] = data_train[columns_to_convert].replace({True: 1, False: 0})


# In[ ]:


data_train.head()


# In[ ]:


#TODO 5. Split training data into training and validation sets
# first create numpy arrays X and Y
X = data_train.drop(columns=['Transported']).to_numpy()
Y = data_train[['Transported']].to_numpy()


# In[ ]:


X[:5,:]


# In[ ]:


Y[:5]


# In[ ]:


X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[ ]:


# Handle the NaN in col 0-5
columns_to_replace = range(6)  # Columns indexed 0 to 5

# Replace NaN values with 0 in X_train_scaled
X_train[:, columns_to_replace] = np.nan_to_num(X_train[:, columns_to_replace], nan=0)

# Replace NaN values with 0 in X_val_scaled
X_val[:, columns_to_replace] = np.nan_to_num(X_val[:, columns_to_replace], nan=0)


# In[ ]:


#TODO 6. Scale the data
# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the first 6 columns of X_train, which are numerical features
scaler.fit(X_train[:, :6])

# Transform the first 6 columns of X_train and X_val
X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()

X_train_scaled[:, :6] = scaler.transform(X_train[:, :6])
X_val_scaled[:, :6] = scaler.transform(X_val[:, :6])


# In[ ]:


X_train_scaled[:5,:]


# In[ ]:


X_val_scaled[:5,:]


# In[ ]:


# Count the number of 1s and 0s in Y


# Build the model:
# 
# 32 features and 1 binary classification label. 
# 
# Output layer has two nodes, output function 'sigmoid', minimize Binary Cross Entropy.
# 
# Create and train neural network model. First add one hidden layer, add more nodes gradually to and evaluate each model.
# 
# Then add new hidden layer and gradually add nodes.
# 
# Evaluation method: print classification report and check Binary Cross Entropy(BCE) for each freshly-trained model.

# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


# In[ ]:


# Create an empty list to track validation BCE when I add more nodes in hidden layer
bce = BinaryCrossentropy(from_logits=True)

J_train = []
J_val = []
num_nodes_1 = []
num_nodes_2 = []


# In[ ]:


def append_losses(model, units_in_hidden_1, units_in_hidden_2):
    """Append loss to the list, so I can track which model does the best"""
    num_nodes_1.append(units_in_hidden_1)
    print("number of nodes in 1st hidden layer:", num_nodes_1)
    
    num_nodes_2.append(units_in_hidden_2)
    print("number of nodes in 2nd hidden layer:", num_nodes_2)
    
    y_val_hat = model.predict(X_val_scaled)
    bce_val = bce(Y_val, y_val_hat).numpy()
    J_val.append(bce_val)
    print("J_val:",J_val)
    
    y_train_hat = model.predict(X_train_scaled)
    bce_train = bce(Y_train, y_train_hat).numpy()
    J_train.append(bce_train)
    print("J_train:",J_train)


# In[ ]:


# model_1: 8 nodes in 1st hidden layer, and 4 nodes in 2nd hidden layer. 
UNITS_IN_HIDDEN_1 = 8
UNITS_IN_HIDDEN_2 = 4
model_1 = 0
model_1 = Sequential()
model_1.add(Dense(units=UNITS_IN_HIDDEN_1, activation='relu'))
model_1.add(Dense(units=UNITS_IN_HIDDEN_2, activation='relu'))
model_1.add(Dense(units=1, activation='sigmoid'))
model_1.compile(loss='binary_crossentropy', metrics=['accuracy','Precision', 'Recall', 'AUC'], optimizer='adam')
model_1.fit(X_train_scaled, Y_train, epochs=800, verbose=1, validation_data=(X_val_scaled, Y_val), callbacks=[early_stopping])
J_list = model_1.history.history['loss']
plt.plot(J_list)


# In[ ]:


append_losses(model=model_1, units_in_hidden_1=UNITS_IN_HIDDEN_1, units_in_hidden_2=UNITS_IN_HIDDEN_2)


# In[ ]:


# model_2: 16 nodes in 1st hidden layer, and 8 nodes in 2nd hidden layer. 
UNITS_IN_HIDDEN_1 = 16
UNITS_IN_HIDDEN_2 = 8
model_2 = 0
model_2 = Sequential()
model_2.add(Dense(units=UNITS_IN_HIDDEN_1, activation='relu'))
model_2.add(Dense(units=UNITS_IN_HIDDEN_2, activation='relu'))
model_2.add(Dense(units=1, activation='sigmoid'))
model_2.compile(loss='binary_crossentropy', metrics=['accuracy','Precision', 'Recall', 'AUC'], optimizer='adam')
model_2.fit(X_train_scaled, Y_train, epochs=800, verbose=1, validation_data=(X_val_scaled, Y_val), callbacks=[early_stopping])
J_list = model_2.history.history['loss']
plt.plot(J_list)


# In[ ]:


append_losses(model=model_2, units_in_hidden_1=UNITS_IN_HIDDEN_1, units_in_hidden_2=UNITS_IN_HIDDEN_2)


# In[ ]:


# model_3: 32 nodes in 1st hidden layer, and 16 nodes in 2nd hidden layer. 
UNITS_IN_HIDDEN_1 = 32
UNITS_IN_HIDDEN_2 = 16
model_3 = Sequential()
model_3.add(Dense(units=UNITS_IN_HIDDEN_1, activation='relu'))
model_3.add(Dense(units=UNITS_IN_HIDDEN_2, activation='relu'))
model_3.add(Dense(units=1, activation='sigmoid'))
model_3.compile(loss='binary_crossentropy', metrics=['accuracy','Precision', 'Recall', 'AUC'], optimizer='adam')
model_3.fit(X_train_scaled, Y_train, epochs=800, verbose=1, validation_data=(X_val_scaled, Y_val), callbacks=[early_stopping])
J_list = model_3.history.history['loss']
plt.plot(J_list)


# In[ ]:


append_losses(model=model_3, units_in_hidden_1=UNITS_IN_HIDDEN_1, units_in_hidden_2=UNITS_IN_HIDDEN_2)


# In[ ]:


# model_4: 64 nodes in 1st hidden layer, and 32 nodes in 2nd hidden layer. 
UNITS_IN_HIDDEN_1 = 64
UNITS_IN_HIDDEN_2 = 32
model_4 = Sequential()
model_4.add(Dense(units=UNITS_IN_HIDDEN_1, activation='relu'))
model_4.add(Dense(units=UNITS_IN_HIDDEN_2, activation='relu'))
model_4.add(Dense(units=1, activation='sigmoid'))
model_4.compile(loss='binary_crossentropy', metrics=['accuracy','Precision', 'Recall', 'AUC'], optimizer='adam')
model_4.fit(X_train_scaled, Y_train, epochs=800, verbose=1, validation_data=(X_val_scaled, Y_val), callbacks=[early_stopping])
J_list = model_4.history.history['loss']
plt.plot(J_list)


# In[ ]:


append_losses(model=model_4, units_in_hidden_1=UNITS_IN_HIDDEN_1, units_in_hidden_2=UNITS_IN_HIDDEN_2)


# In[ ]:


# model_5: 128 nodes in 1st hidden layer, and 64 nodes in 2nd hidden layer. 
UNITS_IN_HIDDEN_1 = 128
UNITS_IN_HIDDEN_2 = 64
model_5 = Sequential()
model_5.add(Dense(units=UNITS_IN_HIDDEN_1, activation='relu'))
model_5.add(Dense(units=UNITS_IN_HIDDEN_2, activation='relu'))
model_5.add(Dense(units=1, activation='sigmoid'))
model_5.compile(loss='binary_crossentropy', metrics=['accuracy','Precision', 'Recall', 'AUC'], optimizer='adam')
model_5.fit(X_train_scaled, Y_train, epochs=800, verbose=1, validation_data=(X_val_scaled, Y_val), callbacks=[early_stopping])
J_list = model_5.history.history['loss']
plt.plot(J_list)


# In[ ]:


append_losses(model=model_5, units_in_hidden_1=UNITS_IN_HIDDEN_1, units_in_hidden_2=UNITS_IN_HIDDEN_2)


# In[ ]:


# Model_5 shows overfitting. Now go back to try less nodes.
# model_6: 128 nodes in 1st hidden layer, and 64 nodes in 2nd hidden layer. 
UNITS_IN_HIDDEN_1 = 64
UNITS_IN_HIDDEN_2 = 64
model_6 = Sequential()
model_6.add(Dense(units=UNITS_IN_HIDDEN_1, activation='relu'))
model_6.add(Dense(units=UNITS_IN_HIDDEN_2, activation='relu'))
model_6.add(Dense(units=1, activation='sigmoid'))
model_6.compile(loss='binary_crossentropy', metrics=['accuracy','Precision', 'Recall', 'AUC'], optimizer='adam')
model_6.fit(X_train_scaled, Y_train, epochs=800, verbose=1, validation_data=(X_val_scaled, Y_val), callbacks=[early_stopping])
J_list = model_6.history.history['loss']
plt.plot(J_list)


# In[ ]:


append_losses(model=model_6, units_in_hidden_1=UNITS_IN_HIDDEN_1, units_in_hidden_2=UNITS_IN_HIDDEN_2)


# This model_4 shows best performance so far. Now preprocess test data and prepare submission file.

# In[ ]:


#TODO 1. Drop following columns as they are not relevant to if passenger survived or not: 'PassengerId', 'Name'.
data_test_dropped = data_test.drop('PassengerId', axis=1)
data_test_dropped = data_test_dropped.drop('Name', axis=1)
data_test_dropped.head()


# In[ ]:


data_test = data_test_dropped
data_test.shape


# In[ ]:


#TODO 2. Change True and False values in binary categorical columns to 1 and 0
col_to_change = ['CryoSleep', 'VIP']
for col in col_to_change:
    data_test[col] = data_test[col].replace({True: 1, False: 0})


# In[ ]:


data_test.head()


# In[ ]:


check_unique_values(data_test)


# In[ ]:


#TODO 3. Handle 'Cabin' column
data_test['Deck'] = data_test['Cabin'].str[0]  # Extract the first letter as the deck
data_test['Deck']


# In[ ]:


data_test['Deck'].unique()


# In[ ]:


data_test['Side'] = data_test['Cabin'].str[-1]  # Extract the last letter as the side
data_test['Side'].unique()


# In[ ]:


data_test = data_test.drop(columns=['Cabin'])


# In[ ]:


data_test.head()


# In[ ]:


# Convert 'Side' column to binary values
data_test['Side'] = data_test['Side'].replace({'P': 1, 'S': 0})


# In[ ]:


check_unique_values_new(data_test)


# In[ ]:


# Handle the nulls
data_test['CryoSleep'] = data_test['CryoSleep'].astype(str)
data_test['VIP'] = data_test['VIP'].astype(str)
data_test['Side'] = data_test['Side'].astype(str)
data_test['HomePlanet'].fillna('Unknown', inplace=True)
data_test['Destination'].fillna('Unknown', inplace=True)
data_test['Deck'].fillna('Unknown', inplace=True)
data_test['CryoSleep'].fillna('Unknown', inplace=True)
data_test['VIP'].fillna('Unknown', inplace=True)
data_test['Side'].fillna('Unknown', inplace=True)


# In[ ]:


check_unique_values_new(data_test)


# In[ ]:


#TODO 4. One hot encoding for categorical columns 'HomePlanet', 'Destination', and 'Deck'
data_test = pd.get_dummies(data_test, columns=['CryoSleep', 'VIP', 'Side', 'HomePlanet', 'Destination', 'Deck'])
data_test.head()


# In[ ]:


data_test.columns


# In[ ]:


# Convert dummy variables (True/False) to 1/0
columns_to_convert = [
    'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
    'CryoSleep_0.0', 'CryoSleep_1.0', 'CryoSleep_nan',
    'VIP_0.0', 'VIP_1.0', 'VIP_nan', 'Side_0.0', 'Side_1.0',
    'Side_nan', 'HomePlanet_Earth', 'HomePlanet_Europa',
    'HomePlanet_Mars', 'HomePlanet_Unknown', 'Destination_55 Cancri e',
    'Destination_PSO J318.5-22', 'Destination_TRAPPIST-1e',
    'Destination_Unknown', 'Deck_A', 'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E',
    'Deck_F', 'Deck_G', 'Deck_T', 'Deck_Unknown'
]

# Replace True/False with 1/0
data_test[columns_to_convert] = data_test[columns_to_convert].replace({True: 1, False: 0})


# In[ ]:


data_test.shape


# In[ ]:


# create numpy arrays X and Y
X_test = data_test.to_numpy()


# In[ ]:


X_test.shape


# In[ ]:


X_test


# In[ ]:


# Handle the NaN in col 0-5
columns_to_replace = range(6)  # Columns indexed 0 to 5

# Replace NaN values with 0
X_test[:, columns_to_replace] = np.nan_to_num(X_test[:, columns_to_replace], nan=0)


# In[ ]:


# Transform the first 6 columns of X_test
X_test_scaled = X_test.copy()
X_test_scaled[:, :6] = scaler.transform(X_test[:, :6])


# In[ ]:


X_test_scaled[:5,:]


# In[ ]:


y_test_hat_cat = (model_4.predict(X_test_scaled) > 0.5).astype(int)


# In[ ]:


y_test_hat_cat = y_test_hat_cat.flatten()


# In[ ]:


data_test['Transported'] = y_test_hat_cat


# In[ ]:


data_test.head()


# In[ ]:


# load original test data again, extract the 'PassengerId' column and add to data_test above
df = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')
data_test['PassengerId'] = df['PassengerId']
data_test.head()


# In[ ]:


# Keep only 'PassengerId' and 'Transported' columns
df_selected = data_test[['PassengerId', 'Transported']]
# Show the first few rows of the new DataFrame to verify


# In[ ]:


# Convert 1 and 0 
df_selected['Transported'] = df_selected['Transported'].replace({1: True, 0: False})
df_selected.head()


# In[ ]:


df_selected.to_csv('/kaggle/working/titanic_spaceship_submission.csv', index=False)


# In[ ]:




