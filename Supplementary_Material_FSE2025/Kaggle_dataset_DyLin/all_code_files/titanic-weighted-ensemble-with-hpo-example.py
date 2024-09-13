#!/usr/bin/env python
# coding: utf-8

# ## Simple weighted ensemble method using CV on <u>CatBoost</u>, <u>LightGBM</u> and <u>XGBoost</u> classifiers and <u>Optuna</u> to HPO

# In[ ]:


# import libraries:

import pandas as pd
import numpy as np

import optuna

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# In[ ]:


# Turn off logs 

optuna.logging.set_verbosity(optuna.logging.WARNING)


# In[ ]:


train_dataframe = pd.read_csv('/kaggle/input/titanic/train.csv')
test_dataframe = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:




# In[ ]:


train_dataframe.head(3)


# In[ ]:


# I'm not processing columns, as this is just an example

cat_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked']

train_dataframe[cat_features] = train_dataframe[cat_features].astype(str).astype('category')
test_dataframe[cat_features] = test_dataframe[cat_features].astype(str).astype('category')


# In[ ]:


# Ensemble model

class Model:
    def __init__(self, n_folds=5, n_trials=15):
        self.n_folds = n_folds
        self.n_trials = n_trials
        self.X = None
        self.y = None
        self.label = None
        self.test_data = None
        self.cat_features = None
        self.models = ['XGBClassifier','CatBoostClassifier', 'LGBMClassifier']
        self.outputs = None
        self.scores = None
    
    def load_data(self, train_data, test_data, label):
        self.label = label
        
        self.X = train_data.drop(columns=[self.label])
        self.Y = train_data[self.label]
        
        self.test_data = test_data
        
        self.cat_features = test_data.select_dtypes(include=['category']).columns.tolist()
        
    def fit(self):
        self.outputs = {
                'XGBClassifier' : np.zeros(self.test_data.shape[0]),
                'CatBoostClassifier' : np.zeros(self.test_data.shape[0]), 
                'LGBMClassifier' : np.zeros(self.test_data.shape[0])
            }
        self.scores = {}
        for model in self.models:
            print(model, "training..")
            
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            score_by_model = 0
            
            for fold, (train_idx, valid_idx) in enumerate(skf.split(self.X, self.Y)):
                X_train = self.X.iloc[train_idx]
                Y_train = self.Y.iloc[train_idx]
                
                X_val = self.X.iloc[valid_idx]
                Y_val = self.Y.iloc[valid_idx]
                
                X_train_spl, X_test_spl, y_train_spl, y_test_spl = train_test_split(X_train, Y_train, random_state=42, shuffle=True, test_size=0.2)
                
                cur_model = None
                
                if model == 'XGBClassifier':
                    def objective(trial):
                        params = {
                            'n_estimators' : trial.suggest_int("n_estimators", 100, 1000),
                            "verbosity" : 0,
                            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                            "max_depth": trial.suggest_int("max_depth", 1, 10),
                            "subsample": trial.suggest_float("subsample", 0.05, 1.0),
                            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
                            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                            "enable_categorical" : True
                        }
                        xgb = XGBClassifier(**params)
                        xgb.fit(X_train_spl, y_train_spl, verbose=False)
                        prediction = xgb.predict(X_test_spl)
                        return accuracy_score(y_test_spl, prediction)
                    
                    optimizer = optuna.create_study(direction='maximize')
                    optimizer.optimize(objective, n_trials=self.n_trials)
                    
                    cur_model = XGBClassifier(**optimizer.best_params, enable_categorical=True)
                    cur_model.fit(X_train, Y_train, verbose=False)
                elif model == 'CatBoostClassifier':
                    cat_features = self.cat_features
                    def objective(trial):
                        params = {
                            'iterations' : trial.suggest_int("iterations", 100, 1000),
                            'learning_rate' : trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                            "depth": trial.suggest_int("depth", 1, 10),
                            "subsample": trial.suggest_float("subsample", 0.05, 1.0),
                            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
                            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
                        }
                        cat = CatBoostClassifier(**params, silent=True, cat_features=cat_features)
                        cat.fit(X_train_spl, y_train_spl)
                        prediction = cat.predict(X_test_spl)
                        return accuracy_score(y_test_spl, prediction)
                    
                    optimizer = optuna.create_study(direction='maximize')
                    optimizer.optimize(objective, n_trials=self.n_trials)
                    
                    cur_model = CatBoostClassifier(**optimizer.best_params, silent=True, cat_features=cat_features)
                    cur_model.fit(X_train, Y_train)
                elif model == 'LGBMClassifier':
                    def objective(trial):
                        params = {
                            'n_estimators' : trial.suggest_int("n_estimators", 100, 1000),
                            'verbosity' : -1,
                            'learning_rate' : trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                            "num_leaves" : trial.suggest_int("num_leaves", 2, 2**10),
                            "subsample": trial.suggest_float("subsample", 0.05, 1.0),
                            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
                            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
                        }
                        lgbm = LGBMClassifier(**params)
                        lgbm.fit(X_train_spl, y_train_spl)
                        prediction = lgbm.predict(X_test_spl)
                        return accuracy_score(y_test_spl, prediction)
                    
                    optimizer = optuna.create_study(direction='maximize')
                    optimizer.optimize(objective, n_trials=self.n_trials)
                    
                    cur_model = LGBMClassifier(**optimizer.best_params, verbosity=-1)
                    cur_model.fit(X_train, Y_train)
                
                score_by_model += accuracy_score(Y_val, cur_model.predict(X_val)) / self.n_folds
                self.outputs[model] += cur_model.predict_proba(self.test_data)[:, 1] / self.n_folds
            
            print( f"{model}, average accuracy score on val: {score_by_model}")
            self.scores[model] = score_by_model
        
    def get_prediction(self):
        prediction = np.zeros(self.test_data.shape[0])
        for model in self.models:
            prediction += self.outputs[model] * (self.scores[model] ** 2 / sum([self.scores[model] ** 2 for model in self.models]))
            
        return prediction
                


# In[ ]:


model = Model(n_folds=10, n_trials=30)

model.load_data(train_dataframe.drop(columns=['PassengerId', 'Name']), test_dataframe.drop(columns=['PassengerId', 'Name']), 'Survived')

model.fit()


# In[ ]:


y_pred = model.get_prediction()

output = pd.DataFrame({
    'PassengerId' : test_dataframe['PassengerId'].values,
    'Survived' : np.round(y_pred)
})


# In[ ]:


output.head(5)

