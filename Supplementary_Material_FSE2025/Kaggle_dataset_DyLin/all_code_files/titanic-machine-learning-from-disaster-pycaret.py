#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pip install -U pycaret -q


# In[ ]:


import pandas as pd
from pycaret.classification import *
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


# In[ ]:


# Load and preprocess the data
data = pd.read_csv("/kaggle/input/titanic/train.csv")
data = data.drop(columns=['Name','PassengerId', 'Ticket'])
data.head()


# In[ ]:


def encode_categorical_features(df):
    categorical_columns = df.select_dtypes(include=['object']).columns
    encoding_dict = {}
    
    for col in categorical_columns:
        unique_values = df[col].unique()
        value_to_int = {val: idx for idx, val in enumerate(unique_values)}
        df[col] = df[col].map(value_to_int)
        encoding_dict[col] = value_to_int
    
    return df, encoding_dict

data, encoding_dict = encode_categorical_features(data)
data = data.fillna(-1)
data.Age = data.Age.astype(int)
data = data.reset_index(drop=True)
data.head()


# In[ ]:


clf1 = setup(data, target='Survived', session_id=42)


# In[ ]:


# Compare models to find the best one
best_model = compare_models(n_select=3)


# In[ ]:


# Finalize the model
final_model = finalize_model(best_model[0])
final_model


# In[ ]:


param_grid = {
    'actual_estimator__depth': [6, 8],
    'actual_estimator__learning_rate': [0.2, 0.4],
    'actual_estimator__iterations': [50, 100],
    'actual_estimator__l2_leaf_reg': [5, 10]
}

grid_search = GridSearchCV(final_model, param_grid, cv=10, scoring='accuracy', n_jobs=-1)
grid_search.fit(data.drop(columns=['Survived']), data['Survived'])


# In[ ]:


model = grid_search.best_estimator_


# In[ ]:


data_test  = pd.read_csv("/kaggle/input/titanic/test.csv")
submission = data_test[['PassengerId']]
data_test  = data_test.drop(columns=['Name','PassengerId','Ticket'])
data_test.head()


# In[ ]:


def apply_encoding_dict(df, encoding_dict):
    for col, encoding in encoding_dict.items():
        if col in df.columns:
            df[col] = df[col].map(encoding)
            # Substituir valores desconhecidos por um valor específico (ex: -1)
            df[col] = df[col].fillna(-1)
    
    return df

data_test = apply_encoding_dict(data_test, encoding_dict)
data_test = data_test.fillna(-1)
data_test.Age = data_test.Age.astype(int)
data_test = data_test.reset_index(drop=True)
data_test.head()


# In[ ]:


predictions = model.predict(data_test)
submission['Survived'] = predictions.tolist()
submission.to_csv('submission.csv', index=False)
submission.head()

