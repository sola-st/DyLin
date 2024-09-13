#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# <h1> <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:3.0px; background: #cceeff; border-bottom: 8px solid #004466" > IMPORTS<br> <div> </h1> 

# In[ ]:


#get_ipython().run_cell_magic('capture', '', '\n!pip install -q autogluon.tabular\n!pip install -q ray==2.10.0\n!pip install -U ipywidgets\n\nimport pandas as pd, numpy as np\nimport os\nfrom warnings import filterwarnings \nfilterwarnings(action = "ignore")\nfrom gc import collect\nfrom colorama import Fore, Style, Back\nfrom tqdm.notebook import tqdm\n\nfrom autogluon.tabular import TabularDataset, TabularPredictor\nfrom sklearn.model_selection import StratifiedKFold as SKF\nfrom sklearn.metrics import roc_auc_score, accuracy_score\n\ndef PrintColor(text:str, color = Fore.BLUE, style = Style.BRIGHT):\n    "Prints color outputs using colorama using a text F-string";\n    print(style + color + text + Style.RESET_ALL)')


# <h1> <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:3.0px; background: #cceeff; border-bottom: 8px solid #004466" > CONFIGURATION<br> <div> </h1> 

# In[ ]:


#get_ipython().run_cell_magic('time', '', '\n# Set this to False for actual run and True for syntax check\ntest_req = False\n\nmodel_id = "AGV1_7"\ntarget   = "Transported"\n\nip_path  = f"/kaggle/input/spacetitanicfe"\nop_path  = f"/kaggle/working"\n\nif test_req:\n    num_gpu         = 0\n    time_budget     = 300\n    excluded_models = ["KNN"]\n    n_splits        = 3\n    verbosity       = 2\n    presets         = \'optimize_for_deployment\'\n    \nelse:\n    num_gpu         = 0\n    time_budget     = 1200\n    excluded_models = ["KNN"] \n    n_splits        = 10\n    verbosity       = 1\n    presets         = \'best_quality\'\n    \ncv = SKF(n_splits = n_splits, random_state = 42, shuffle = True)\n\nprint()\ncollect();')


# <h1> <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:3.0px; background: #cceeff; border-bottom: 8px solid #004466" > FOREWORD<br> <div> </h1> 

# This kernel is a demonstration of the usage of AutoGluon for the Spaceship Titanic challenge. **Accuracy score** is the eval metric for the competition <br>
# We start off with the features I created for my previous kernel for this competition as [here](https://www.kaggle.com/datasets/ravi20076/spacetitanicfe).<br>
# We then use AutoGluon on the dataset and extract probability predictions. We then blend this with other models and finally submit to the competition. <br>
# 
# Wishing you the best for the competition! <br>
# 
# Please find my artefacts for this competition as below- <br>
# 1. https://www.kaggle.com/code/ravi20076/spaceshiptitanic-models-v2 - manual ML models with Optuna ensemble <br>
# 2. https://www.kaggle.com/code/ravi20076/sptitanic-bootstrapensemble-pipeline - my first kernel for this assignment with manual ML models <br>
# 3. https://www.kaggle.com/datasets/ravi20076/spacetitanicfe - feature engineering dataset for quick model training <br>

# <h1> <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:3.0px; background: #cceeff; border-bottom: 8px solid #004466" > PREPROCESSING<br> <div> </h1> 

# In[ ]:


#get_ipython().run_cell_magic('time', '', '\ntrain  = pd.read_parquet(os.path.join(ip_path, "Train.parquet"))\ntest   = pd.read_parquet(os.path.join(ip_path, "Test.parquet"))\nsub_fl = pd.read_csv(f"/kaggle/input/spaceship-titanic/sample_submission.csv")\n\ntrain[target] = train[target].astype(np.uint8)\n\nwith np.printoptions(linewidth = 150):\n    PrintColor(f"\\nTrain data columns\\n")\n    print(np.array(train.columns))\n    PrintColor(f"\\nTest data columns\\n")\n    print(np.array(test.columns))\n    \nprint();\ncollect();')


# <h1> <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:3.0px; background: #cceeff; border-bottom: 8px solid #004466" > MODEL TRAINING<br> <div> </h1> 

# In[ ]:


#get_ipython().run_cell_magic('time', '', '\n# Initializing OOF and test set predictions:-\noof_preds = np.zeros(train.shape[0])\nmdl_preds = 0\nXtest     = TabularDataset(test)\n\nfor fold_nb, (train_idx, dev_idx) in tqdm(enumerate(cv.split(train, train[target]))):\n    PrintColor(f"\\n------- FOLD {fold_nb} -------\\n")\n    Xtr   = TabularDataset(train.iloc[train_idx])\n    Xdev  = TabularDataset(train.iloc[dev_idx])\n    \n    model = TabularPredictor(label        = target,\n                             eval_metric  = \'accuracy\',\n                             problem_type = \'binary\',\n                             path         = os.path.join(op_path, f"{model_id}Fold{fold_nb}"),                           \n                            )\n\n    model.fit(Xtr,\n              presets    = presets,\n              time_limit = time_budget,\n              verbosity  = verbosity,\n              excluded_model_types = excluded_models,\n             )\n\n    results   = model.fit_summary()\n    mdl_preds = mdl_preds + (model.predict_proba(Xtest, as_pandas = False)[:,-1] / n_splits)\n    oof_preds[dev_idx] = model.predict_proba(Xdev, as_pandas = False)[:,-1]\n    \n    print()\n    display(model.leaderboard().style.set_caption(f"Model leaderboard"))\n    print()\n    collect();\n    ')


# <h1> <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:3.0px; background: #cceeff; border-bottom: 8px solid #004466" > SUBMISSION<br> <div> </h1> 

# In[ ]:


#get_ipython().run_cell_magic('time', '', '\nscore = accuracy_score(train[target].values, np.where(oof_preds >= 0.598, 1, 0))\nPrintColor(f"\\n---> OOF score = {score :.6f}\\n")\n\nsub_fl[target] = np.where(mdl_preds >= 0.598, True, False)\nsub_fl.to_csv(os.path.join(op_path, f"SubBase.csv"), index = False)\n\nprint()\ndisplay(sub_fl.head(10).style.set_caption(f"Submission no blend"))\nprint()\n\npd.DataFrame(oof_preds, index = range(len(oof_preds)), columns = [model_id]).\\\nto_csv(os.path.join(op_path, "OOF_Preds.csv"))\n\n# Combining my model results with good public work:-\nsub1 = pd.read_csv(f"/kaggle/input/space-titanic-eda-advanced-feature-engineering/submission.csv")\nsub2 = pd.read_csv(f"/kaggle/input/space-titanic/XGB_best.csv")\nsub_fl[target] = sub_fl[target] | sub1[target] | sub2[target]\nsub_fl.to_csv(os.path.join(op_path, f"SubBlend.csv"), index = False)\n\nprint()\ndisplay(sub_fl.head(10).style.set_caption(f"Submission blend"))\nprint()\n\ncollect();')

