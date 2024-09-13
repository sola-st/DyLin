#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


df_test = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')
df_train = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')


# ## DATA ANALYSIS

# #### Quick glance at the training data.

# In[ ]:


df_train.info()


# In[ ]:


df_train.describe()


# #### Numerical features distributions

# In[ ]:


df_train.hist(figsize=(30, 20), bins=40)


# #### How each features are correlated with target?

# In[ ]:


df_train_numerc_only = df_train._get_numeric_data()
corr_matrix = df_train_numerc_only.corr()
corr_matrix['Transported'].sort_values(ascending=False)


# #### test data

# In[ ]:


df_test.info()


# ## DATA PREPROCESSING

# In[ ]:


df_test


# #### Concatenating train and test dataframes - to do all preprocessing operations at once

# In[ ]:


train_ids = df_train['PassengerId']
test_ids = df_test['PassengerId']

df = pd.concat([df_train.drop(columns=['Transported']), df_test], axis=0, ignore_index=True)


# #### Filling null values and dropping 'Name' column

# In[ ]:


df['HomePlanet'].fillna('Nan', inplace=True)
df['Cabin'].fillna('Nan', inplace=True)
df['Destination'].fillna('Nan', inplace=True)

columns_to_impute = ['CryoSleep', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

imputer = IterativeImputer(
    estimator=KNeighborsClassifier(),
    max_iter=10,
    random_state=12
)
df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])

df.drop(columns=['Name'], inplace=True)
df.info()


# #### Some feature engineering

# In[ ]:


luxury_amenities = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
shopping_features = ['FoodCourt', 'ShoppingMall']
service_features = ['RoomService', 'Spa', 'VRDeck']

df['Total_Spending'] = df[luxury_amenities].sum(axis=1)
df['No_spending'] = (df['Total_Spending']==0).astype(int)
df['UsedAmenities'] = df[luxury_amenities].gt(0).sum(axis=1)
df['Service_Spending'] = df[service_features].sum(axis=1)
df['Shopping_Spending'] = df[shopping_features].sum(axis=1)

df[["Deck", "Cabin_num", "Side"]] = df["Cabin"].str.split("/", expand=True)
df.drop(columns=['Cabin'], inplace=True)


# #### Converting boolean columns to integer

# In[ ]:


df['VIP'] = df['VIP'].astype(int)
df['CryoSleep'] = df['CryoSleep'].astype(int)


# #### One hot encoding on non-numerical columns

# In[ ]:


df_encoded = pd.get_dummies(df, columns=['HomePlanet', "Deck", "Cabin_num", "Side", 'Destination'])
df_encoded


# #### Scaling numerical columns

# In[ ]:


df_scaled = df_encoded
scaler = StandardScaler()
columns_to_scale = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df_scaled[columns_to_scale] = scaler.fit_transform(df_scaled[columns_to_scale])
df_scaled.set_index('PassengerId', inplace=True)


# #### Reducing dimensionality using Principal Component Analysis

# In[ ]:


pca = PCA(n_components=100)
df_pca = pd.DataFrame(pca.fit_transform(df_scaled), index=df_scaled.index)


# #### Splitting dataframe to train and test sets

# In[ ]:


test = df_pca.tail(df_test.shape[0])

X = df_pca.head(df_train.shape[0])
y = df_train['Transported'].astype(int)


# #### 'Cutting off' validation set from train set

# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# ## BUILDING AND TRAINING NN

# #### Building and compiling keras model

# In[ ]:


model = keras.models.Sequential([
    
    keras.layers.Dense(256, input_dim=100, activation='relu'),
    keras.layers.BatchNormalization(),        # Normalizing layer inputs to improve training speed and convergence.
    keras.layers.Dropout(0.5),             # Dropout for regularization
    
    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    
    keras.layers.Dense(64, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    
    keras.layers.Dense(32, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


# #### Callbacks: 
# ##### early stopping - stops training after model starts to overfitting, 
# ##### learning rate scheduler - adjusts the learning rate during training to improve convergence and accuracy

# In[ ]:


early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)


# In[ ]:


def learning_rate_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
    
lr_scheduler = LearningRateScheduler(learning_rate_scheduler)


# #### Training

# In[ ]:


history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=64,
    callbacks=[early_stopping, lr_scheduler]
)


# #### Learning curves

# In[ ]:


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# #### Predictions

# In[ ]:


predictions = pd.DataFrame({
    'Transported': [True if i > 0.5 else False for i in model.predict(test)],
    'PassengerId': test.index
}).set_index('PassengerId')
predictions.to_csv('predictions.csv')


# In[ ]:




