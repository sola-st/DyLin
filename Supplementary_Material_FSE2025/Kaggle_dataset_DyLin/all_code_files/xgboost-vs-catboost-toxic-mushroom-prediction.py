#!/usr/bin/env python
# coding: utf-8

# # A. Import Libraries 

# In[ ]:


import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score,ConfusionMatrixDisplay,classification_report,recall_score,f1_score ,precision_score,confusion_matrix
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance

import time

from catboost import CatBoostClassifier

from sklearn.preprocessing import OneHotEncoder

try:
    import shap
except:
    get_ipython().system('pip install shap')

from imblearn.under_sampling import RandomUnderSampler

import optuna
from xgboost import XGBClassifier

plt.style.use("ggplot")
sns.set_palette("Set2")

import warnings
warnings.filterwarnings("ignore")


# **Make a reusable function to plot the different metrics across thresholds**

# In[ ]:


# 'Decision Plot Function
def plot_metrics_vs_threshold(y_val, y_pred_prob, threshold_step=0.05):

    # Initialize lists to store the results
    accuracy = []
    auc = []
    precision = []
    recall = []
    F1 = []
    thresholds = []

    # Define the range of thresholds
    threshold_values = np.arange(0.1, 0.9 + threshold_step, threshold_step)

    # Calculate metrics for each threshold
    for i in threshold_values:
        y_pred = (y_pred_prob >= i).astype(int)
        accuracy.append(accuracy_score(y_val, y_pred))
        auc.append(roc_auc_score(y_val, y_pred))
        precision.append(precision_score(y_val, y_pred, zero_division=0))
        recall.append(recall_score(y_val, y_pred, zero_division=0))
        F1.append(f1_score(y_val, y_pred, zero_division=0))
        thresholds.append(i)

    # Plot the metrics
    plt.figure(figsize=(16, 8))

    plt.plot(thresholds, accuracy, label='Accuracy', color='blue',)
    plt.plot(thresholds, auc, label='AUC', color='green')
    plt.plot(thresholds, precision, label='Precision', color='red', )
    plt.plot(thresholds, recall, label='Recall', color='purple',)

    # Add vertical line at threshold 0.5
    plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold 0.5')

    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.title('Decision Chart')
    plt.legend()
    plt.tight_layout()
    plt.show()


# # B. Data Preprocessing and Analysis

# ## Loading And Analysis

# In[ ]:


df = pd.read_csv("/kaggle/input/playground-series-s4e8/train.csv")
df.head()


# In[ ]:


df.tail()


# In[ ]:


df['class'] = df['class'].replace({"e":0,'p':1})


# In[ ]:


df.info()


# In[ ]:




# In[ ]:


df = df.drop(columns=['spore-print-color','veil-color','veil-type','stem-root','stem-root','stem-surface','stem-surface'])


# In[ ]:


df = df.dropna()


# In[ ]:


df.select_dtypes('object').nunique()


# In[ ]:


df.describe()


# In[ ]:


plt.figure(figsize=(6, 4))
sns.countplot(x='class', data=df)
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()


# ## Preprocessing

# In[ ]:


df= df.drop(columns=['id'])


# In[ ]:


target  = 'class'
X = df.drop(columns='class')
y = df['class']


# In[ ]:


encoder = LabelEncoder()
for  i in X.select_dtypes('object').columns.to_list() :
    X[i] = encoder.fit_transform(X[i])


# In[ ]:


X


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X,y,test_size = 0.2 ,random_state = 2)
X_train.shape, X_val.shape, y_val.shape, y_val.shape


# In[ ]:


under_sampler = RandomUnderSampler()
X_train,y_train= under_sampler.fit_resample(X_train,y_train)


# In[ ]:


X_train.shape,y_train.shape


# In[ ]:


X_val.shape,y_val.shape


# In[ ]:




# In[ ]:


kfolds_number = 3


# # C. Build Models

# ## XGBOOST Model 

# In[ ]:


def make_xgboost_model(X_train, X_val, y_train, y_val, model_name, kfolds_number=kfolds_number, threshold=0.50):
    print("[INFO] Starting XGBoost Model Training")
    
    # Define objective function for Optuna (if using Optuna for hyperparameter tuning)
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 700),
            'max_depth': trial.suggest_int('max_depth', 1, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'verbosity': 0,
            'tree_method':  'gpu_hist'
        }

        xgb = XGBClassifier(**param)
        xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_val)
        score = roc_auc_score(y_val, y_pred)
        return score
    
    
    # Perform hyperparameter optimization using Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    best_trial = study.best_trial
    # Use the best parameters to train the final model
    best_params = best_trial.params
    # Cross-validation setup
    kfold = KFold(n_splits=kfolds_number, shuffle=True, random_state=1)
    fold_metrics = {'accuracy': [], 'auc': [], 'precision': [], 'recall': [], 'f1_score': []}
    

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train), start=1):
        X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = XGBClassifier(tree_method='gpu_hist',**best_params)
        model.fit(X_cv_train, y_cv_train)

        y_cv_val_pred =  model.predict(X_cv_val)
        # Calculate metrics
        aucscore = roc_auc_score(y_cv_val,y_cv_val_pred)
        accurayscore = accuracy_score(y_cv_val, y_cv_val_pred)
        precisionscore = precision_score(y_cv_val, y_cv_val_pred)
        recallscore = recall_score(y_cv_val,y_cv_val_pred)
        f1score = f1_score(y_cv_val, y_cv_val_pred)

        # Store metrics for each fold
        fold_metrics['accuracy'].append(accurayscore)
        fold_metrics['auc'].append(aucscore)
        fold_metrics['precision'].append(precisionscore)
        fold_metrics['recall'].append(recallscore)
        fold_metrics['f1_score'].append(f1score)

        # Print metrics for the current fold
        print(f"[INFO] Fold {fold} - Accuracy: {accurayscore:.4f}")
        print(f"[INFO] Fold {fold} - AUC: {aucscore:.4f}")
        print(f"[INFO] Fold {fold} - Precision: {precisionscore:.4f}")
        print(f"[INFO] Fold {fold} - Recall: {recallscore:.4f}")
        print(f"[INFO] Fold {fold} - F1 Score: {f1score:.4f}")

    # Average metrics across all folds
    avg_metrics = {key: np.mean(value) for key, value in fold_metrics.items()}

    print(f"[INFO] Average Accuracy across all folds: {avg_metrics['accuracy']:.4f}")
    print(f"[INFO] Average AUC across all folds: {avg_metrics['auc']:.4f}")
    print(f"[INFO] Average Precision across all folds: {avg_metrics['precision']:.4f}")
    print(f"[INFO] Average Recall across all folds: {avg_metrics['recall']:.4f}")
    print(f"[INFO] Average F1 Score across all folds: {avg_metrics['f1_score']:.4f}")

    # Train the final model on the full dataset
    final_model = XGBClassifier(tree_method='gpu_hist',**best_params)
    print("[INFO] Training Final Model on Full Dataset")
    final_model.fit(X_train, y_train)

    # Predict on validation data with the final model
    y_pred_val_final_prob = final_model.predict_proba(X_val)[:, 1]
    y_pred_val_final = (y_pred_val_final_prob >= threshold).astype(int)

    final_metrics = (
        accuracy_score(y_val, y_pred_val_final),
        roc_auc_score(y_val, y_pred_val_final_prob),
        precision_score(y_val, y_pred_val_final),
        recall_score(y_val, y_pred_val_final),
        f1_score(y_val, y_pred_val_final)
    )

    # Print final metrics on evaluation dataset
    print(f"[INFO] Final Model - Accuracy on Validation Data: {final_metrics[0]:.4f}")
    print(f"[INFO] Final Model - AUC on Validation Data: {final_metrics[1]:.4f}")
    print(f"[INFO] Final Model - Precision on Validation Data: {final_metrics[2]:.4f}")
    print(f"[INFO] Final Model - Recall on Validation Data: {final_metrics[3]:.4f}")
    print(f"[INFO] Final Model - F1 Score on Validation Data: {final_metrics[4]:.4f}")

    # Prepare output
    model_metrics = {
        'Model_Name': model_name,
        "Accuracy Score": final_metrics[0],
        "AUC Score": final_metrics[1],
        "Precision Score": final_metrics[2],
        "Recall Score": final_metrics[3],
        'F1 Score': final_metrics[4]
    }
    avg_kfolds_metrics = {'Model_Name': model_name, **avg_metrics}

    return model_metrics, y_pred_val_final, final_model, avg_kfolds_metrics, y_pred_val_final_prob


# In[ ]:


start_time = time.time()
model_metrics_XGBOOST,y_pred_val_XGBOOST,model_XGBOOST,avg_kfolds_metrics_XGBOOST,y_pred_val_final_prob_XGBOOST = make_xgboost_model(X_train = X_train,X_val = X_val,y_train=y_train,y_val= y_val , model_name = 'XGBOOST')
end_time = time.time()
execution_time_XGBOOST = end_time - start_time
model_metrics_XGBOOST['execution_time'] = execution_time_XGBOOST


# In[ ]:




# In[ ]:


plot_metrics_vs_threshold(y_val, y_pred_val_final_prob_XGBOOST)


# ## CATBOOST Model

# In[ ]:


def make_catboost_model(X_train, X_val, y_train, y_val, model_name, kfolds_number=kfolds_number, threshold=0.50):
    print("[INFO] Starting CatBoost Model Training")
    
    # Define objective function for Optuna (if using Optuna for hyperparameter tuning)
    def objective(trial):
        param = {
            'iterations': trial.suggest_int('iterations', 300, 700),
            'depth': trial.suggest_int('depth', 1, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 1, 255),
            'auto_class_weights': trial.suggest_categorical('auto_class_weights', [None, 'Balanced']),
            'loss_function': 'Logloss',
            'task_type': 'GPU',  # Enable GPU training
            'devices': '0:1'  # Specify GPU devices (change as needed)
        }

        catboost = CatBoostClassifier(**param, verbose=0)
        catboost.fit(X_train, y_train)
        y_pred = catboost.predict(X_val)
        score = roc_auc_score(y_val, y_pred)
        return score
    
    # Perform hyperparameter optimization using Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    best_trial = study.best_trial
    # Use the best parameters to train the final model
    best_params = best_trial.params
    # Cross-validation setup
    kfold = KFold(n_splits=kfolds_number, shuffle=True, random_state=1)
    fold_metrics = {'accuracy': [], 'auc': [], 'precision': [], 'recall': [], 'f1_score': []}


    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train), start=1):
        X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = CatBoostClassifier(**best_params,task_type='GPU', verbose=0)
        model.fit(X_cv_train, y_cv_train)
        y_cv_val_pred =  model.predict(X_cv_val)
        # Calculate metrics
        aucscore = roc_auc_score(y_cv_val,y_cv_val_pred)
        accurayscore = accuracy_score(y_cv_val, y_cv_val_pred)
        precisionscore = precision_score(y_cv_val, y_cv_val_pred)
        recallscore = recall_score(y_cv_val,y_cv_val_pred)
        f1score = f1_score(y_cv_val, y_cv_val_pred)
        # Store metrics for each fold
        fold_metrics['accuracy'].append(accurayscore)
        fold_metrics['auc'].append(aucscore)
        fold_metrics['precision'].append(precisionscore)
        fold_metrics['recall'].append(recallscore)
        fold_metrics['f1_score'].append(f1score)

        # Print metrics for the current fold
        print(f"[INFO] Fold {fold} - Accuracy: {accurayscore:.4f}")
        print(f"[INFO] Fold {fold} - AUC: {aucscore:.4f}")
        print(f"[INFO] Fold {fold} - Precision: {precisionscore:.4f}")
        print(f"[INFO] Fold {fold} - Recall: {recallscore:.4f}")
        print(f"[INFO] Fold {fold} - F1 Score: {f1score:.4f}")

    # Average metrics across all folds
    avg_metrics = {key: np.mean(value) for key, value in fold_metrics.items()}

    print(f"[INFO] Average Accuracy across all folds: {avg_metrics['accuracy']:.4f}")
    print(f"[INFO] Average AUC across all folds: {avg_metrics['auc']:.4f}")
    print(f"[INFO] Average Precision across all folds: {avg_metrics['precision']:.4f}")
    print(f"[INFO] Average Recall across all folds: {avg_metrics['recall']:.4f}")
    print(f"[INFO] Average F1 Score across all folds: {avg_metrics['f1_score']:.4f}")

    # Train the final model on the full dataset
    final_model = CatBoostClassifier(**best_params,task_type='GPU', verbose=0)
    print("[INFO] Training Final Model on Full Dataset")
    final_model.fit(X_train, y_train)

    # Predict on validation data with the final model
    y_pred_val_final_prob = final_model.predict_proba(X_val)[:, 1]
    y_pred_val_final = (y_pred_val_final_prob >= threshold).astype(int)

    final_metrics = (
        accuracy_score(y_val, y_pred_val_final),
        roc_auc_score(y_val, y_pred_val_final_prob),
        precision_score(y_val, y_pred_val_final),
        recall_score(y_val, y_pred_val_final),
        f1_score(y_val, y_pred_val_final)
    )

    # Print final metrics on evaluation dataset
    print(f"[INFO] Final Model - Accuracy on Validation Data: {final_metrics[0]:.4f}")
    print(f"[INFO] Final Model - AUC on Validation Data: {final_metrics[1]:.4f}")
    print(f"[INFO] Final Model - Precision on Validation Data: {final_metrics[2]:.4f}")
    print(f"[INFO] Final Model - Recall on Validation Data: {final_metrics[3]:.4f}")
    print(f"[INFO] Final Model - F1 Score on Validation Data: {final_metrics[4]:.4f}")

    # Prepare output
    model_metrics = {
        'Model_Name': model_name,
        "Accuracy Score": final_metrics[0],
        "AUC Score": final_metrics[1],
        "Precision Score": final_metrics[2],
        "Recall Score": final_metrics[3],
        'F1 Score': final_metrics[4]
    }
    avg_kfolds_metrics = {'Model_Name': model_name, **avg_metrics}

    return model_metrics, y_pred_val_final, final_model, avg_kfolds_metrics, y_pred_val_final_prob


# In[ ]:


start_time = time.time()
model_metrics_CATBOOST,y_pred_val_CATBOOST,model_CATBOOST,avg_kfolds_metrics_CATBOOST,y_pred_val_final_prob_CATBOOST = make_catboost_model(X_train = X_train,X_val = X_val,y_train=y_train,y_val= y_val , model_name = 'CATBOOST')
end_time = time.time()
execution_time_CATBOOST = end_time - start_time
model_metrics_CATBOOST['execution_time'] = execution_time_CATBOOST


# In[ ]:




# In[ ]:


plot_metrics_vs_threshold(y_val, y_pred_val_final_prob_XGBOOST)


# # D. Comparison 

# ## Metrics Frame

# In[ ]:


# Combine the dictionaries into a list
model_metrics_list = [model_metrics_CATBOOST,model_metrics_XGBOOST]

# Create a DataFrame from the list of dictionaries
metrics_df = pd.DataFrame(model_metrics_list).sort_values(by='AUC Score',ascending=False)

metrics_df


# In[ ]:


# Metrics Of CrossValidation 

cross_metrics_df_list = [avg_kfolds_metrics_CATBOOST.values(),model_metrics_XGBOOST.values()]

# Create a DataFrame from the list of dictionaries
cross_metrics_df = pd.DataFrame(cross_metrics_df_list)
cross_metrics_df.columns = ['Model Name', 'Accuracy Score','AUC Score','Precision Score','Recall Score','F1 Score','d']
cross_metrics_df.drop(columns='d',inplace=True)


cross_metrics_df


# ## Features Importances

# In[ ]:


# Optianing the most features that had an impact of our price
def plot_feature_importance(model, feature_names=None, top_n=50, plot=True):

    feature_importance = model.feature_importances_
    
    if feature_names is None:
        feature_names = model.feature_name()
        
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})

    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(top_n)

    if plot:
        plt.figure(figsize=(10, 10))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
        plt.title('Feature Importance')
        plt.show()


# ### Xgboost 

# In[ ]:


plot_feature_importance(model_XGBOOST,feature_names=X_train.columns)


# ### Catboost

# In[ ]:


plot_feature_importance(model_CATBOOST,feature_names=X_train.columns)


# # E. Get Predictions

# In[ ]:


test = pd.read_csv("/kaggle/input/playground-series-s4e8/test.csv")


# In[ ]:


test_id = test.pop('id')
test = test.drop(columns=['spore-print-color','veil-color','veil-type','stem-root','stem-root','stem-surface','stem-surface'])
test = test.fillna(method='ffill')
for  i in test.select_dtypes('object').columns.to_list() :
    test[i] = encoder.fit_transform(test[i])
    
test_columns = test.columns


# In[ ]:


xgboost_pred = model_XGBOOST.predict(test)
catboost_pred = model_CATBOOST.predict(test)


# In[ ]:


xgboost_pred = pd.DataFrame({'class':xgboost_pred})
xgboost_pred['id'] =test_id
xgboost_pred['class'] = xgboost_pred['class'].replace({0:'e',1:'p'})
xgboost_pred.to_csv("xgboost_pred.csv",index = False)


# In[ ]:


catboost_pred = pd.DataFrame({'class':catboost_pred})
catboost_pred['id'] =test_id
catboost_pred['class'] = catboost_pred['class'].replace({0:'e',1:'p'})
catboost_pred.to_csv("catboost_pred.csv",index = False)

