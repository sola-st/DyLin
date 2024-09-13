#!/usr/bin/env python
# coding: utf-8

# 
# 
# <h1> <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:3.0px; background: #cceeff; border-bottom: 8px solid #004466" > IMPORTS<br> <div> </h1> 

# In[ ]:


#get_ipython().run_cell_magic('capture', '', '\n!pip install -q lightautoml\n\nimport pandas as pd, numpy as np\nimport os\nfrom warnings import filterwarnings \nfilterwarnings(action = "ignore")\nfrom gc import collect\nfrom colorama import Fore, Style, Back\nfrom tqdm.notebook import tqdm\n\nfrom sklearn.model_selection import StratifiedKFold as SKF\nfrom sklearn.metrics import roc_auc_score, accuracy_score\n\nimport torch\nimport torch.nn as nn\nfrom lightautoml.automl.presets.tabular_presets import TabularAutoML\nfrom lightautoml.tasks import Task\n\ndef PrintColor(text:str, color = Fore.BLUE, style = Style.BRIGHT):\n    "Prints color outputs using colorama using a text F-string";\n    print(style + color + text + Style.RESET_ALL)')


# <h1> <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:3.0px; background: #cceeff; border-bottom: 8px solid #004466" > CONFIGURATION<br> <div> </h1> 

# In[ ]:


#get_ipython().run_cell_magic('time', '', '\n# Set this to False for actual run and True for syntax check\ntest_req = False\n\nmodel_id = "LAMAV1_2"\ntarget   = "Transported"\nip_path  = f"/kaggle/input/spacetitanicfe"\nop_path  = f"/kaggle/working"\n \nif test_req:\n    time_budget = 300\n    verbosity   = 3\n    algos       = ["denselight", "cb", "lgb"]\n    n_splits    = 3\n    batch_size  = 32\n    \nelse:\n    time_budget = 3600 * 6\n    verbosity   = 2    \n    algos       = ["denselight", "cb_tuned", "lgb_tuned", "xgb_tuned"]\n    n_splits    = 10\n    batch_size  = 32\n    \nprint()\ncollect();')


# <h1> <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:3.0px; background: #cceeff; border-bottom: 8px solid #004466" > FOREWORD<br> <div> </h1> 

# This kernel is a demonstration of the usage of AutoGluon for the Spaceship Titanic challenge. **Accuracy score** is the eval metric for the competition <br>
# We start off with the features I created for my previous kernel for this competition as [here](https://www.kaggle.com/datasets/ravi20076/spacetitanicfe).<br>
# We then use LightAutoML on the dataset and extract probability predictions. We then blend this with other models and finally submit to the competition. <br>
# 
# Wishing you the best for the competition! <br>
# 
# Please find my artefacts for this competition as below- <br>
# 1. https://www.kaggle.com/code/ravi20076/spaceshiptitanic-models-v2 - manual ML models with Optuna ensemble <br>
# 2. https://www.kaggle.com/code/ravi20076/sptitanic-bootstrapensemble-pipeline - my first kernel for this assignment with manual ML models <br>
# 3. https://www.kaggle.com/datasets/ravi20076/spacetitanicfe - feature engineering dataset for quick model training <br>
# 4. https://www.kaggle.com/code/ravi20076/spaceshiptitanic-autogluon-v1 - Autogluon with cross-validation explicitly performed <br>

# <h1> <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:3.0px; background: #cceeff; border-bottom: 8px solid #004466" > PREPROCESSING<br> <div> </h1> 

# In[ ]:


#get_ipython().run_cell_magic('time', '', '\ntrain  = pd.read_parquet(os.path.join(ip_path, "Train.parquet"))\ntest   = pd.read_parquet(os.path.join(ip_path, "Test.parquet"))\nsub_fl = pd.read_csv(f"/kaggle/input/spaceship-titanic/sample_submission.csv")\n\ntrain[target] = train[target].astype(np.uint8)\n\nwith np.printoptions(linewidth = 150):\n    PrintColor(f"\\nTrain data columns\\n")\n    print(np.array(train.columns))\n    PrintColor(f"\\nTest data columns\\n")\n    print(np.array(test.columns))\n    \nprint();\ncollect();')


# <h1> <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:3.0px; background: #cceeff; border-bottom: 8px solid #004466" > MODEL TRAINING<br> <div> </h1> 

# In[ ]:


#get_ipython().run_cell_magic('time', '', '\ntask = Task(\'binary\', loss = \'logloss\', metric = \'accuracy\')\n\nmodel = \\\nTabularAutoML(\n    task = task,\n    timeout = time_budget,\n    cpu_limit = 16,\n    general_params = {"use_algos": [algos]},\n\n    nn_params = {\n        "n_epochs"       : 30,\n        "bs"             : batch_size,\n        "num_workers"    : 0,\n        "path_to_save"   : None,\n        "freeze_defaults": True,\n        "cont_embedder"  : \'plr\',\n        \'cat_embedder\'   : \'weighted\',\n        \'act_fun\'        : \'ReLU\',\n        "hidden_size"    : [32, 8],\n        \'embedding_size\' : 32,\n        \'stop_by_metric\' : True,\n        \'verbose_bar\'    : False,\n        "snap_params"    : { \'k\': 2, \'early_stopping\': True, \'patience\': 2, \'swa\': True }\n    },\n\n    nn_pipeline_params = {"use_qnt": True, "use_te": False},\n\n    reader_params = {\'n_jobs\': 16, \'cv\': n_splits, \'random_state\': 42, \'advanced_roles\': True}\n)\n\noof_preds = model.fit_predict(train,\n                              roles = {\'target\': target},\n                              verbose = verbosity\n                              ).data\npreds     = model.predict(test).data\n\n\nprint();\ncollect();')


# <h1> <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:3.0px; background: #cceeff; border-bottom: 8px solid #004466" > SUBMISSION<br> <div> </h1> 

# In[ ]:


#get_ipython().run_cell_magic('time', '', '\nscore = accuracy_score(train[target].values, np.where(oof_preds >= 0.5, 1, 0))\nPrintColor(f"\\n---> OOF score = {score :.6f}\\n")\n\nsub_fl[target] = np.where(preds >= 0.5, True, False)\nsub_fl.to_csv(os.path.join(op_path, f"SubBase_{model_id}.csv"), index = False)\n\nprint()\ndisplay(sub_fl.head(10).style.set_caption(f"Submission no blend"))\nprint()\n\npd.DataFrame(oof_preds, index = range(len(oof_preds)), columns = [model_id]).\\\nto_csv(os.path.join(op_path, f"OOF_Preds_{model_id}.csv"))\n\n# Combining my model results with good public work:-\nsub1 = pd.read_csv(f"/kaggle/input/space-titanic-eda-advanced-feature-engineering/submission.csv")\nsub2 = pd.read_csv(f"/kaggle/input/space-titanic/XGB_best.csv")\nsub_fl[target] = sub_fl[target] | sub1[target] | sub2[target]\nsub_fl.to_csv(os.path.join(op_path, f"SubBlend_{model_id}.csv"), index = False)\n\nprint()\ndisplay(sub_fl.head(10).style.set_caption(f"Submission blend"))\nprint()\n\nsub_fl[target] = preds\nsub_fl.to_csv(os.path.join(op_path, f"Mdl_Preds_{model_id}.csv"), index = False)\n\ncollect();')

