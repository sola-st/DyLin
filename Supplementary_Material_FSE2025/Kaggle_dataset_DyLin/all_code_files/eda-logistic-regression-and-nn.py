#!/usr/bin/env python
# coding: utf-8

# # Spaceship Titanic Kaggle Challenge!

# ### Label Descriptions (for self reference)

# - PassengerId - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.
# - HomePlanet - The planet the passenger departed from, typically their planet of permanent residence.
# - CryoSleep - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
# - Cabin - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
# - Destination - The planet the passenger will be debarking to.
# - Age - The age of the passenger.
# - VIP - Whether the passenger has paid for special VIP service during the voyage.
# - RoomService, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
# - Name - The first and last names of the passenger.
# - Transported - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.

# ## Preliminary Processing

# ### Make imports and configure

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support


# In[ ]:


sns.set_theme(style='whitegrid')
sns.set_palette('viridis')


# ### Read in and inspect data

# In[ ]:


df = pd.read_csv("/kaggle/input/spaceship-titanic/train.csv")
test = pd.read_csv("/kaggle/input/spaceship-titanic/test.csv")


# In[ ]:




# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe().T.sort_values(by='mean', ascending=False).style.background_gradient(cmap='BuGn')


# #### Splitting Cabin

# In[ ]:


def split_cabin(cabin):
    if not pd.isna(cabin):
        return pd.Series(cabin.split('/'), index=['Deck','Number', 'Side'])
    else:
        return pd.Series(dtype='object')


# In[ ]:


def apply_and_concat(df):
    cabin = df['Cabin'].apply(split_cabin)
    return pd.concat([df, cabin], axis=1)


# In[ ]:


df = apply_and_concat(df)
test = apply_and_concat(test)


# #### Convert Data to Correct Type

# In[ ]:


def convert_dtype(df):

    df['VIP'] = df['VIP'].astype(bool)
    df['CryoSleep'] = df['CryoSleep'].astype(bool)
    df['Number'] = df['Number'].astype(float)
    
    return df


# In[ ]:


df = convert_dtype(df)
test = convert_dtype(test)


# ### Null Value Analysis

# In[ ]:




# #### Column and Row Null Distribution

# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

null_cols = df.isnull().sum().sort_values(ascending=False)
null_cols.plot(kind='bar', ax = ax1)
ax1.set_xlabel('')

null_rows = x=df.isnull().sum(axis=1)
sns.countplot(x=null_rows[null_rows!=0], ax=ax2)

plt.show()


# ## Exploratory Data Analysis

# ### Distribution of Target Variable

# In[ ]:


sns.countplot(x='Transported', data=df)
plt.show()


# ### Categorical Variables

# #### Origin and Destinations

# In[ ]:


data = []
origins = list(df['HomePlanet'].dropna().unique())
destinations = list(df['Destination'].dropna().unique())

for origin in origins:
    for destination in destinations:
        temp = df[(df['HomePlanet'] == origin) & (df['Destination'] == destination)]
        row = [origin, destination, temp.shape[0]]
        data.append(row)
        # d[origin + ' -> ' + destination] = temp.shape[0]


# In[ ]:


trip_data = pd.DataFrame(data, columns=['Origin', 'Destination', 'Number of Trips'])


# In[ ]:


sns.barplot(x='Origin', y='Number of Trips', data=trip_data, hue='Destination')


# #### Distribution of Categorical Values

# In[ ]:


cat_var = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
for col in cat_var:
    print(df[col].value_counts())
    print('\n')


# In[ ]:


fig, axs = plt.subplots(2, 3, figsize=(20,10))
axs = axs.flatten()

i = 0
for col in cat_var:
    sns.countplot(x=col, data=df, ax=axs[i])
    i+=1

plt.tight_layout()
# plt.delaxes(axs[7])


# #### Which categories correlate to transportation?

# In[ ]:


fig, axs = plt.subplots(2, 3, figsize=(16,8))
axs = axs.flatten()

i = 0
for col in cat_var[:6]:
    sns.countplot(x=col, data=df, ax=axs[i], hue='Transported')
    i+=1
    
plt.tight_layout()


# - Europa have disproportionately more passengers being transported
# - Passengers in Cryosleep are disproportionately transported
# - Passengers departing for 55 Cancri e are disproportionately transported
# - Passengers in decks B, G and C are more likely to be transported
# - Passengers on Starboard more likely to be transported

# ### Continuous Variables

# #### Distribution Of Spending

# In[ ]:


spending = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']


# In[ ]:


df['TotalSpend'] = df[spending].sum(axis=1)


# In[ ]:


for spend in (spending + ['TotalSpend']):
    zeros = df[spend].value_counts()
    print(f'Number of 0 values in {spend} Spending is {zeros[0]} out of {zeros.sum()} or {(zeros[0]/zeros.sum())*100:.2f}%')


# In[ ]:


fig, axs = plt.subplots(3, 2, figsize=(20,20))
axs = axs.flatten()

i = 0
for spend in (spending + ['TotalSpend']):
    x = df[df[spend] > 0]
    sns.histplot(x=x[spend], bins=35, kde=True, ax=axs[i], hue=x['Transported'])
    i+=1
    
plt.tight_layout()


# #### Distribution Of Age

# In[ ]:


fig, (ax1,ax2) = plt.subplots(1,2,figsize=(16,6))
sns.histplot(x='Age', data=df, kde=True, bins=35, ax=ax1)
sns.histplot(x='Age', data=df, kde=True, bins=35, hue='Transported', ax=ax2)
plt.show()


# #### Distribution of Cabin Number

# In[ ]:


fig, (ax1,ax2) = plt.subplots(1,2,figsize=(16,6))
sns.histplot(x='Number', data=df, kde=True, bins=35, ax=ax1)
sns.histplot(x='Number', data=df, kde=True, bins=35, hue='Transported', ax=ax2)
plt.show()


# ### Correlation Matrix

# In[ ]:


plt.figure(figsize=(12,8))
mask = np.triu(df.corr())
sns.heatmap(df.corr(), annot=True, mask=mask)


# ## Pre-Processing

# In[ ]:


# Save PassengerId Column

ids = test['PassengerId']


# ### Dropping Unneeded Columns

# In[ ]:


def drop_rows(df):
    if 'TotalSpend' in df.columns:
        return df.drop(columns=['PassengerId' ,'Name', 'Cabin', 'TotalSpend'])
    else:
        return df.drop(columns=['PassengerId' ,'Name', 'Cabin'])


# In[ ]:


df = drop_rows(df)
test = drop_rows(test)


# ### NA Removal

# #### Continuous Variables

# Use the median values for the spending variables (think they're all 0 which is reasonable) and the mean value for the age

# In[ ]:


def del_na_cont(df):
    for spend in spending:
        med = df[spend].median()
        # print(f'Median for {spend} is {med}')
        df[spend] = df[spend].fillna(med)
    
    mean_age = df['Age'].mean()
    # print(f'Mean of Ages is {mean_age:.2f}')
    df['Age'] = df['Age'].fillna(mean_age)
    
    return df


# In[ ]:


df = del_na_cont(df)
test = del_na_cont(test)


# #### Categorical Variables and Cabin Number

# Replace values by a weighted sampling of variables using np.choice

# In[ ]:


cat_var = cat_var + ['Number']


# In[ ]:


def del_na_cat(df):
    for var in cat_var:
        
        # Get the length of na columns
        length_of_na = len(df[df[var].isnull()])
        
        # Get the names of the variables
        var_list = df[var].dropna().unique()
        
        # Find the prior distribution we need for random assignment
        probs = list(df[var].value_counts()) / (df[var].value_counts().sum())
        
        # Choice method to randomly assign
        inserts = np.random.choice(var_list, length_of_na, p=probs)
        
        # Insert it into the dataframe with loc
        df.loc[df[var].isnull(), var] = np.random.choice(var_list, length_of_na, p=probs)
        
    return df


# In[ ]:


df = del_na_cat(df)
test = del_na_cat(test)


# ### Encoding Categorical Data

# In[ ]:


def encode(df):

    # Nominal Data
    df = pd.concat([df, pd.get_dummies(df['HomePlanet'], drop_first=True), 
                    pd.get_dummies(df['Destination'], drop_first=True),
                    pd.get_dummies(df['Deck'], drop_first=True)],
                axis=1)
    
    df.drop(columns=['HomePlanet', 'Destination', 'Deck'], inplace=True)
    
    # Binary Data
    df['CryoSleep'] = df['CryoSleep'].map({True: 1, False:0})
    df['VIP'] = df['VIP'].map({True: 1, False:0})
    df['Side'] = df['Side'].map({'S': 1, 'P': 0})
    
    # Target Variable
    if 'Transported' in df.columns:
        df['Transported'] = df['Transported'].map({True: 1, False:0})
    
    return df


# In[ ]:


df = encode(df)
test = encode(test)


# ## Model Building

# In[ ]:


X = df.drop(columns=['Transported'])
y = df['Transported']


# ### Splitting the Dataset

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


# ### Logistic Regression

# In[ ]:


model_logit = LogisticRegression(max_iter=1000).fit(X_train, y_train)


# In[ ]:


preds = model_logit.predict(X_test)


# In[ ]:


sns.heatmap(
pd.DataFrame(confusion_matrix(y_test, preds), 
             columns = ['Predicted Not Transported', 'Predicted Transported'],
             index=['Not Transported', 'Transported'])
, annot=True, fmt='g', cbar=False)


# In[ ]:


eval_df = pd.DataFrame(
            [accuracy_score(y_test, preds)] + list(precision_recall_fscore_support(y_test, preds, average='binary')), 
            index=['Accuracy','Precision', 'Recall', 'F-Score', 'Support'],
            columns=['Logistic Regression']
            ).T


# In[ ]:


eval_df


# ### Neural Network

# #### Hyperparameters

# In[ ]:


n_input = len(X_train.columns)
n_hidden = 32
n_classes = 2
batch_size = 256
learning_rate = 0.001
n_epochs = 300


# #### Instantiate Dataloaders

# In[ ]:


X_train_tensor = torch.tensor(X_train.values)
y_train_tensor = torch.tensor(y_train.values)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

X_test_tensor = torch.tensor(X_test.values)
y_test_tensor = torch.tensor(y_test.values)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)


# In[ ]:


train_dataloader = DataLoader(train_dataset, 
                              batch_size=batch_size,
                              shuffle=True
                            ) 

test_dataloader = DataLoader(test_dataset, 
                              batch_size=batch_size,
                              shuffle=False
                            ) 


# #### Create Network

# In[ ]:


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.l3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, train_data):
        out = self.l1(train_data)
        out = self.relu(out)
        out = self.l2(out) 
        return out


# #### Instantiate model, loss function and optimizer

# In[ ]:


model_nn = NeuralNet(n_input, n_hidden, n_classes)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_nn.parameters(), lr=learning_rate)


# #### Training Loop

# In[ ]:


losses = []

for epoch in range(n_epochs):
    running_loss = 0
    
    for batch_no, (features, labels) in enumerate(train_dataloader):
        
        # forward 
        outputs = model_nn(features.float())
        loss = loss_fn(outputs, labels)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        
        # adam step
        optimizer.step()
        
        # add 
        running_loss += loss.item()

        
    loss = running_loss / len(train_dataloader)
    losses.append(loss)
    
    if ((epoch + 1) % 10) == 0:
        print(f'epoch --> {epoch+1} | loss --> {loss}')
    


# In[ ]:


plt.plot(range(n_epochs), losses)
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.show()


# #### Evaluate

# In[ ]:


test_output = model_nn(X_test_tensor.float())
_, preds = torch.max(test_output, 1)
acc = accuracy_score(y_test, preds)

eval_df.loc['Neural Network'] = [acc] + list(precision_recall_fscore_support(y_test, preds, average='binary'))


# In[ ]:


sns.heatmap(
pd.DataFrame(confusion_matrix(y_test, preds), 
             columns = ['Predicted Not Transported', 'Predicted Transported'],
             index=['Not Transported', 'Transported'])
, annot=True, fmt='g', cbar=False)


# In[ ]:


eval_df


# #### Prep for Submission

# In[ ]:


test_scores = model_nn(torch.tensor(test.values).float())
_, test_labels = torch.max(test_scores, 1)


# ## Submission

# In[ ]:


sub_logistic = pd.DataFrame([
                ids,
                model_logit.predict(test).astype(bool)],
                index = ['PassengerId', 'Transported']
            ).T


# In[ ]:


# sub_logistic.to_csv('submissions/logit.csv', index=False)


# In[ ]:


sub_nn = pd.DataFrame([
        ids,
        test_labels.numpy().astype(bool)],
        index = ['PassengerId', 'Transported']
    ).T


# In[ ]:


# sub_nn.to_csv('submissions/nn.csv', index=False)

