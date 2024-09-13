#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings
warnings.filterwarnings('ignore')

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Hi everyone! It's a little close to the end of the competition, and while my original intention was to make it to the top 15%, it looks like I didn't quite make the cut. Still, there is much to be happy about, as this competition proved to be an incredible learning experience. My solution fosters a voting ensemble of three models: XGBoost, CatBoost, and LightGBM. I am by no means an expert, so if you can point out any errors/flaws in my process, please let me know — I am  more than willing to take advice. 

# In[ ]:


#get_ipython().system('pip install ucimlrepo')


# In[ ]:


from ucimlrepo import fetch_ucirepo 

original = fetch_ucirepo(id=848)['data']['original']

train = pd.read_csv("/kaggle/input/playground-series-s4e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s4e8/test.csv")
test_id = test['id']

train = train.drop('id', axis=1)
test = test.drop('id', axis=1)


# # ***PREPROCESSING***
# 
# My preprocessing takes a blends of strategies found in some of the top EDA notebooks. I combined the train and test set categories to promote label consistency, as it turns out the datasets weren't properly cleaned beforehand. Still, I believe there is merit in keeping unclean data in kaggle competitions; it closely mirrors inevitable human error that may occur in real datasets.
# 
# Additionally, while I originally attempted to impute the mode for all missing columns, I (and some others) discovered that grouping these missing values into their own categories proved to be more beneficial for this competition. It may not have made the whole difference, but it made a difference nonetheless. 
# 
# Lastly, I took the liberty to convert categorical features into the 'category' datatype and float64 features into the 'float32' datatype in hopes of speeding things up in the longrun. I didn't measure the change in memory (and I should've), but I think this helped a tiny bit when hyperparameter tuning. 

# In[ ]:


def preprocess(train, test):
    target = train['class']
    train = train.drop('class', axis=1)
    
    combined = pd.concat([train, test], keys=['train', 'test'])
    
    cat_features = combined.select_dtypes(include='object').columns
    float_features = combined.select_dtypes(include='float64').columns

    for col in cat_features: 
        # https://www.kaggle.com/code/ambrosm/pss4e8-eda-which-makes-sense#Ensembling
        valid_cat = original[col].unique().tolist()
        combined.loc[~combined[col].isin(valid_cat), col] = np.nan
        
        combined[col] = combined[col].fillna('no bueno').astype('category')
    
    for col in float_features: 
        combined[col] = combined[col].astype('float32')
        combined[col] = combined[col].fillna(combined[col].mean())
        
    new_train = combined.loc['train'].copy()
    new_test = combined.loc['test'].copy()
    
    new_train['class'] = target
    
    return new_train, new_test

train, test = preprocess(train, test)


# In[ ]:


X = train.drop(['class'], axis=1)
y = train['class']


# Here, we encode the response variable into what the machine can understand (0's and 1's) and one-hot encode the categorical features. 

# In[ ]:


from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

le = LabelEncoder()
y = le.fit_transform(y)

cat_features = X.select_dtypes(include='category').columns

encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
transformer = ColumnTransformer([('encoder', encoder, cat_features)],
                                  remainder='passthrough')

X = transformer.fit_transform(X)
test = transformer.transform(test)


# I was pretty surprised and upset that I didn't discover this sooner, but it turns out that the MCC can be sped up quite considerably (one kaggler promised an ~ 8 second speedup) over sklearn's implementation of the metric. Not entirely sure what goes on in the sklearn side of things that makes this metric so much slower, but hey, it's not up to me. 

# In[ ]:


from sklearn.metrics import make_scorer

# https://www.kaggle.com/competitions/playground-series-s4e8/discussion/528193
def matthews_corrcoef_fast(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0,1]

fast_mcc = make_scorer(matthews_corrcoef_fast)


# In[ ]:


from sklearn.model_selection import train_test_split 

def split_fit_predict(model, X, y):
    """
    Perform a simple train-test split, fit the model, and provide
    an MCC. To be used in hyperparameter tuning.
    """
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mcc = matthews_corrcoef_fast(y_test, y_pred)
    print(f'MCC: {mcc:.5f}' + ' '*8 + '-'*50) # Linebreak
        
    return mcc 


# # ***HYPERPARAMETER TUNING***
# 
# As I mentioned, I am not an expert. For this competition, I pretty much learned all there is to know about the most common approaches for hyperparameter tuning. I cycled between sklearn's GridSearchCV and skopt's BayesSearchCV, but ultimately ended up with Optuna after a few recommendations. I've sampled the hyperparameters used for XGBoost below, but forks of this notebook ran in parallel for my XGB, CATB, and LGBM models. Hyperparameters for this model were selected to the best of my ability, but I admit I do not quite yet understand these models well enough to make informed decisions on the best parameters to tune. I'd like to thank @tilii7 for some help on this aspect of the competition.

# In[ ]:


# import optuna
# from xgboost import XGBClassifier

# # https://www.kaggle.com/code/bextuychiev/no-bs-guide-to-hyperparameter-tuning-with-optuna
# def objective(trial):
#     params = {
#             'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
#             'max_depth': trial.suggest_int('max_depth', 3, 20),
#             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#             'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#             'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#             'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
#             'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),
#             'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),
#             'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0, log=True)
#         }

#     model = XGBClassifier(seed=42, **params)
#     mcc = split_fit_predict(model, X, y) 
    
#     return mcc 


# In[ ]:


# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=1000, timeout=39600) # 11 hours
# trial = study.best_trial

# print(f"Number of finished trials: {len(study.trials)}")


# # ***ENSEMBLING***
# First, let's define a simple cross-validation function to test model performance. 

# In[ ]:


from sklearn.model_selection import StratifiedKFold

skfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

def mcc_cv(model, X, y):
    """
    Stratified 3-fold cross validation with MCC as the scoring
    metric. Prints the CV number and MCC for each fold. 
    """
    mcc_sum = 0

    for fold, (train_index, test_index) in enumerate(skfolds.split(X, y)): 
        X_train_folds, y_train_folds = X[train_index], y[train_index]
        X_test_fold, y_test_fold = X[test_index], y[test_index]
        
        model.fit(X_train_folds, y_train_folds)
        y_pred_fold = model.predict(X_test_fold)
        
        mcc = matthews_corrcoef_fast(y_test_fold, y_pred_fold)
        
        print(f"CV: {fold+1}/3. MCC: {mcc}.")
        mcc_sum += mcc
                    
    return mcc_sum / 3


# In[ ]:


catb_params = {'iterations': 998,
               'depth': 12,
               'learning_rate': 0.027217341550328074,
               'subsample': 0.9925314893125786,
               'colsample_bylevel': 0.2013552770915636,
               'min_child_samples': 58,
               'reg_lambda': 0.03351906637892442,
               'random_strength': 3.26894334350654e-06,
               'scale_pos_weight': 1.0047296624414477,
               'max_bin': 2029,
               'grow_policy': 'SymmetricTree'}

lgbm_params = {'n_estimators': 679, 
               'max_depth': 16,
               'learning_rate': 0.07568390710874888,
               'subsample': 0.3154861476124139,
               'colsample_bytree': 0.590958978613878,
               'min_child_samples': 40, 
               'reg_lambda': 2.8406954934626265, 
               'reg_alpha': 3.9253650491020436e-08, 
               'scale_pos_weight': 1.006404258824218, 
               'num_leaves': 731, 
               'feature_fraction': 0.2931010583859128, 
               'bagging_fraction': 0.9627359563866602, 
               'bagging_freq': 3,
               'boosting_type': 'gbdt'}

xgb_params = {'n_estimators': 866,
              'max_depth': 20, 
              'learning_rate': 0.020293657731919768, 
              'subsample': 0.6918499150810423, 
              'colsample_bytree': 0.36422864453573944,
              'gamma': 6.739821170460403e-07, 
              'lambda': 2.0780571409806596e-06, 
              'alpha': 0.8181170074617319, 
              'scale_pos_weight': 1.0498070510623436, 
              'max_bin': 1879,
              'grow_policy': 'lossguide',
              'tree_method': 'hist'}


# In[ ]:


from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# Construct the models for the vote ensemble
catb = CatBoostClassifier(random_state = 42, verbose = False, **catb_params)
xgb = XGBClassifier(seed=42, **xgb_params)
lgbm = LGBMClassifier(random_state=42, verbose=-1, **lgbm_params)


# In[ ]:


from sklearn.ensemble import VotingClassifier

gbm_ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb),
        ('lgbm', lgbm),
        ('catb', catb)
    ]
)

mcc = mcc_cv(gbm_ensemble, X, y)


# # ***SUBMISSION***

# In[ ]:


gbm_ensemble.fit(X, y)
y_pred = gbm_ensemble.predict(test)

submission = pd.DataFrame()
submission['id'] = test_id
submission['class'] = le.inverse_transform(y_pred)


# In[ ]:


submission.to_csv('/kaggle/working/VOTING_ENSEMBLE.csv', index=False)

