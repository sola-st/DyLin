#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#get_ipython().system(' pip install --upgrade polars scikit-learn # necessary for polars integration')

import os
import copy
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import pandas as pd
import warnings
import sklearn
pl.Config (tbl_rows=30)
SEED = 42
KAGGLE_INPUT_PATH= '/kaggle/input/playground-series-s4e8'
KAGGLE_WORKING_PATH= '/kaggle/working'


# # Data Loading

# In[ ]:


schema_train = {
  'id': pl.Int64,
  'class': pl.String, # clean: 'p', 'e'
  'cap-diameter': pl.Float32, # 1 null
  'cap-shape': pl.String, # 1 null - bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s - contains floats
  'cap-surface': pl.String, # 1 null - fibrous=f,grooves=g,scaly=y,smooth=s - contains floats
  'cap-color': pl.String, # 1 null - brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y - contains floats
  'does-bruise-or-bleed': pl.String, # 1 null - original bruises: bruises=t,no=f - contains floats, many values and value has-ring
  # 'odor': pl.String, # original values: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s
  'gill-attachment': pl.String, # 1 null - attached=a,descending=d,free=f,notched=n - contains floats
  'gill-spacing': pl.String, # 1 null - close=c,crowded=w,distant=d - here 49 floating numbers - ordinal ? contains characters
  'gill-color': pl.String, # 1 null - black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y	- contains floats and weird text
  'stem-height': pl.Float32, # 0 null - not in the original dataset
  'stem-width': pl.Float32, # 0 null - not in the original dataset
  'stem-root': pl.String, # 1 null - bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=?	- contains floats and different encodings
  'stem-surface': pl.String, # 1 null - fibrous=f,scaly=y,silky=k,smooth=s	- contains floats and different encodings
  'stem-color': pl.String, # 1 null - brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y	- contains floats and different encodings
  'veil-type': pl.String, # 1 null - partial=p,universal=u	- contains floats and different encodings
  'veil-color': pl.String, # 1 null - brown=n,orange=o,white=w,yellow=y	- contains floats and different encodings
  'has-ring': pl.String, # 1 null - none=n,one=o,two=t	- contains floats and different encodings
  'ring-type': pl.String, # 1 null - cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z	- contains floats and different encodings
  'spore-print-color': pl.String, # 1 null - black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y - contains floats and different encodings
  'habitat': pl.String, # 1 null - grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d	- contains floats and different encodings
  'season': pl.String, # 0 null - abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y - clean different encodings: 's', 'u', 'a', 'w'
}
df_train = pl.read_csv(os.path.join(KAGGLE_INPUT_PATH, 'train.csv'), schema=schema_train)
df_train.describe()


# In[ ]:


schema_test = {
  'id': pl.Int64,
  # 'class': pl.String, # clean: 'p', 'e'
  'cap-diameter': pl.String, # 1 null - contains some characters
  'cap-shape': pl.String, # 1 null - bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s - contains floats
  'cap-surface': pl.String, # 1 null - fibrous=f,grooves=g,scaly=y,smooth=s - contains floats
  'cap-color': pl.String, # 1 null - brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y - contains floats
  'does-bruise-or-bleed': pl.String, # 1 null - original bruises: bruises=t,no=f - contains floats, many values and value has-ring
  'gill-attachment': pl.String, # 1 null - attached=a,descending=d,free=f,notched=n - contains floats
  'gill-spacing': pl.String, # 1 null - close=c,crowded=w,distant=d - here 49 floating numbers - ordinal ? contains characters
  'gill-color': pl.String, # 1 null - black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y	- contains floats and weird text
  'stem-height': pl.Float32, # 0 null - not in the original dataset
  'stem-width': pl.Float32, # 0 null - not in the original dataset
  'stem-root': pl.String, # 1 null - bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=?	- contains floats and different encodings
  'stem-surface': pl.String, # 1 null - fibrous=f,scaly=y,silky=k,smooth=s	- contains floats and different encodings
  'stem-color': pl.String, # 1 null - brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y	- contains floats and different encodings
  'veil-type': pl.String, # 1 null - partial=p,universal=u	- contains floats and different encodings
  'veil-color': pl.String, # 1 null - brown=n,orange=o,white=w,yellow=y	- contains floats and different encodings
  'has-ring': pl.String, # 1 null - none=n,one=o,two=t	- contains floats and different encodings
  'ring-type': pl.String, # 1 null - cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z	- contains floats and different encodings
  'spore-print-color': pl.String, # 1 null - black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y - contains floats and different encodings
  'habitat': pl.String, # 1 null - grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d	- contains floats and different encodings
  'season': pl.String, # 0 null - abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y - clean different encodings: 's', 'u', 'a', 'w'
}
df_test = pl.read_csv(os.path.join(KAGGLE_INPUT_PATH, 'test.csv'), schema=schema_test)
df_test.describe()


# In[ ]:


feature_columns = [col for col in df_train.columns if col not in ['id', 'class']]
numerical_columns = ['cap-diameter', 'stem-height', 'stem-width']
categorical_columns = [col for col in feature_columns if col not in numerical_columns]
valid_values_per_categorical = {}
for col in categorical_columns:
  col_value_counts = df_train.get_column(col).value_counts()
  value_counts = dict(col_value_counts.iter_rows())
  valid_values = [k for k,v in value_counts.items() if v > 100]
  print(f'{col}: {len(value_counts)} unique values - valid: {valid_values}')
  valid_values_per_categorical[col] = valid_values
  print(f'Invalid: {[k for k in value_counts.keys() if k not in valid_values_per_categorical[col]]}')
  print()


# # Data Cleaning

# In[ ]:


def clean_missing_noise(df):
    df = df.with_columns(
      pl.col(col).fill_null('NN') for col in valid_values_per_categorical.keys()
    )
    df = df.with_columns(
      pl.when(pl.col(col).is_in(valid_values)).then(pl.col(col)).fill_null('N') for col, valid_values in valid_values_per_categorical.items()
    )
    df = df.with_columns((
      pl.col('cap-diameter').cast(pl.Float32).fill_null(strategy='mean'),
      pl.col('stem-height').fill_null(strategy='mean')
    ))
    return df
df_train = clean_missing_noise(df_train)    
df_train = df_train.with_columns((
  pl.col('class').replace({'e': 1, 'p': 0}).cast(pl.Int8),
))
df_train.describe()


# In[ ]:


df_test = clean_missing_noise(df_test)
df_test.describe()


# In[ ]:


for col, valid_values in valid_values_per_categorical.items():
      if None in valid_values:
            valid_values.remove(None)
      valid_values.append('NN')
      valid_values.append('N')

df_train = df_train.with_columns(
  pl.col(col).cast(pl.Enum(valid_values)) for col, valid_values in valid_values_per_categorical.items()
)
df_test = df_test.with_columns(
  pl.col(col).cast(pl.Enum(valid_values)) for col, valid_values in valid_values_per_categorical.items()
)
df_test.head()


# # Data Preprocessing

# In[ ]:


from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, PowerTransformer
from sklearn.compose import ColumnTransformer

feat_transformer=ColumnTransformer(transformers=[
  # ('onehot', OneHotEncoder(drop='if_binary', sparse_output=False, dtype=int), categorical_columns),
  ('ordinal', OrdinalEncoder(handle_unknown='error', dtype=int), categorical_columns), 
  ('robust', RobustScaler(), numerical_columns),
  ('power', PowerTransformer(), numerical_columns),
], sparse_threshold=0.0, verbose_feature_names_out=True)

feat_transformer.set_output(transform="polars")
df_train.hstack(feat_transformer.fit_transform(df_train), in_place=True)
df_test.hstack(feat_transformer.transform(df_test), in_place=True)

# onehot_columns = [col for col in df_train.columns if col.startswith('onehot_')]
ordinal_columns = [col for col in df_train.columns if col.startswith('ordinal_')]
robust_columns = [col for col in df_train.columns if col.startswith('robust_')]
power_columns = [col for col in df_train.columns if col.startswith('power_')]
df_train.head()


# # PyTorch FeedForwardNeuralNetwork Hyperparameter Tuning

# ## Lightning Module with Categorical Embedding

# In[ ]:


def calc_emb_out(n: int, root): # dimensions of the categorical encodings
    if n <= 2:
        return n
    return int(np.ceil(n ** (1/root) )) # actually 4th-root
    
embedding_dim_in = [(col, df_train.select(col).n_unique()) for col in ordinal_columns]
[(col, emb_in, calc_emb_out(emb_in, 3)) for col, emb_in in embedding_dim_in]


# In[ ]:


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from sklearn.model_selection import train_test_split
import torchmetrics
from torch.nn.functional import binary_cross_entropy
from torchmetrics.functional import matthews_corrcoef

class BinaryClassificationModel(LightningModule):
    def __init__(self, config: dict, num_numerical=len(numerical_columns), embedding_dim_in: list[str, int]=embedding_dim_in):
        super(BinaryClassificationModel, self).__init__()
        self.save_hyperparameters()
        self.layer_1_size = config["layer_1_size"]
        self.layer_1_dropout = config["layer_1_dropout"]
        self.layer_2_size = config["layer_2_size"]
        self.lr = config["lr"]
        self.embedding_dimension_root = config["emb_root"]
        # Embedding layers for categorical features
        embedding_sizes = [(emb_in, calc_emb_out(emb_in, self.embedding_dimension_root)) for _, emb_in in embedding_dim_in]
        self.embeddings = nn.ModuleList([nn.Embedding(num_emb_in, num_emb_out) 
                                         for num_emb_in, num_emb_out in embedding_sizes])
        # Fully connected layers with specified hidden dimensions
        total_embedding_dim = sum(embed_dim for _, embed_dim in embedding_sizes)
        input_dim = num_numerical + total_embedding_dim
        
        self.ffnn = nn.Sequential(
            nn.Linear(input_dim, self.layer_1_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.layer_1_size),
            nn.Dropout1d(p=self.layer_1_dropout),
            nn.Linear(self.layer_1_size, self.layer_2_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.layer_2_size),
            nn.Linear(self.layer_2_size, 1),
            nn.Sigmoid()
        )
        self.eval_loss = []
        self.eval_mcc = []
        
    def forward(self, x_numerical, x_categorical):
        # Embedding lookup for categorical features
        embedded_cats = [self.embeddings[i](x_categorical[:, i]) for i in range(x_categorical.size(1))]
        embedded_cats = torch.cat(embedded_cats, dim=1)
        
        # Concatenate numerical and embedded categorical features
        x = torch.cat([x_numerical, embedded_cats], dim=1)
        
        # Feedforward through the sequential layers
        return self.ffnn(x)
    
    def training_step(self, batch, batch_idx):
        x_numerical, x_categorical, y = batch
        y_hat = self.forward(x_numerical, x_categorical)
        loss = binary_cross_entropy(y_hat, y)
        self.log("ptl/train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x_numerical, x_categorical, y = batch
        y_hat = self.forward(x_numerical, x_categorical)
        loss = binary_cross_entropy(y_hat, y)
        val_mcc = matthews_corrcoef(y_hat, y, task='binary')
        self.eval_mcc.append(val_mcc)
        self.eval_loss.append(loss)
        return {"val_loss": loss, "val_mcc": val_mcc}
        
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.eval_loss).mean()
        avg_mcc = torch.stack(self.eval_mcc).mean()
        self.log("ptl/val_loss", avg_loss, sync_dist=True)
        self.log("ptl/val_mcc", avg_mcc, sync_dist=True)
        self.eval_loss.clear()
        self.eval_mcc.clear()

    def predict_step(self, batch):
        x_numerical, x_categorical = batch
        return self.forward(x_numerical, x_categorical)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

model = BinaryClassificationModel(config={
                                      'lr': 1e-3,
                                      'layer_1_size': 128,
                                      'layer_1_dropout': 0.25,
                                      'layer_2_size': 32,
                                      'emb_root': 1
                                  }
)
ModelSummary(model, max_depth=-1)


# ## Lightning DataModule for train and prediction stages

# In[ ]:


class TensorDataset(Dataset):
    def __init__(self, X_numerical, X_categorical, y):
        self.X_numerical = X_numerical
        self.X_categorical = X_categorical
        self.y = y
    
    def __len__(self):
        return self.X_numerical.shape[0]
    
    def __getitem__(self, idx):
        batch_numerical = self.X_numerical[idx]
        batch_categorical = self.X_categorical[idx]
        if self.y is None:
            return batch_numerical, batch_categorical
        return batch_numerical, batch_categorical, self.y[idx]


class DataModule(LightningDataModule):
    def __init__(self, config: dict, df_train=df_train, df_test=df_test, categorical_features=ordinal_columns, 
                 batch_size=32, val_size=0.2, random_state=SEED, num_workers=8):
        super().__init__()
        self.df_train = df_train
        self.df_test = df_test
        if config['numerical_features'] == 'robust':
            self.numerical_features = robust_columns
        else:
            self.numerical_features = power_columns
        self.categorical_features = categorical_features
        self.batch_size = 2**config['batch_size_exp']
        self.val_size = val_size
        self.random_state = random_state
        self.num_workers = num_workers

    def _polars_to_dataset(self, df):
        # Extract numerical, categorical, and target columns for train set
        X_numerical = torch.tensor(df.select(self.numerical_features).to_numpy(), dtype=torch.float32)
        X_categorical = torch.tensor(df.select(self.categorical_features).to_numpy(), dtype=torch.long)
        if 'class' in df.columns:
            y = torch.tensor(df.select('class').to_numpy(), dtype=torch.float32)
        else:
            y = None
        return TensorDataset(X_numerical, X_categorical, y)
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            # Train-validation split
            train, val = train_test_split(self.df_train, test_size=self.val_size, random_state=self.random_state)
            # Extract numerical, categorical, and target columns for training and validation set
            self.train_dataset = self._polars_to_dataset(train)
            self.val_dataset = self._polars_to_dataset(val)
        elif stage == 'predict':
            self.predict_dataset = self._polars_to_dataset(self.df_test)
        else:
            raise NotImplementedError(f'Unsupported stage in datamodule setup: {stage}')
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=False)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=2**21, num_workers=0, pin_memory=False) # fits val data into one batch for inference

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=False)


# ## Parallel Ray Hyperparameter tuning on T4 x 2

# In[ ]:


import gc
from pytorch_lightning.callbacks import StochasticWeightAveraging
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.train.torch import TorchTrainer

def train_func(config): # https://docs.ray.io/en/latest/train/getting-started-pytorch-lightning.html
    datamodule = DataModule(
                         config,
                         num_workers=2, # 12 on single GPU with i5 13600
                         val_size=0.2)
    model = BinaryClassificationModel(config=config)
    swa_cb = StochasticWeightAveraging(1e-2)
    trainer = Trainer(
        devices="auto",
        accelerator="auto",
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback(), swa_cb],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule=datamodule)

search_space = {
    "layer_1_size": tune.choice([128, 256]),
    "layer_1_dropout": tune.uniform(0.2, 0.4),
    "layer_2_size": tune.choice([32, 64, 128]),
    "lr": tune.loguniform(1e-4, 1e-2),
    'emb_root': tune.randint(1, 4),
    "batch_size_exp": tune.randint(12, 16),
    "numerical_features": tune.choice(["robust", "power"])
}

scaling_config = ScalingConfig(
    num_workers=2, use_gpu=True, resources_per_worker={"CPU": 2, "GPU": 1}, 
    trainer_resources={
        "CPU": 0, "GPU": 0 # https://docs.ray.io/en/latest/train/user-guides/using-gpus.html#trainer-resources
    }
)

run_config = RunConfig(
    checkpoint_config=CheckpointConfig(
        num_to_keep=2,
        checkpoint_score_attribute="ptl/val_mcc",
        checkpoint_score_order="max",
    ),
)

# Define a TorchTrainer without hyper-parameters for Tuner
ray_trainer = TorchTrainer(
    train_func,
    scaling_config=scaling_config,
    run_config=run_config,
)

def tune_asha(num_samples=10, num_epochs=10):
    torch.cuda.empty_cache()
    gc.collect()
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)
    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="ptl/val_mcc",
            mode="max",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )
    return tuner.fit()

results = tune_asha(num_samples=30, num_epochs=15) # TODO running trials in parallel ?


# In[ ]:


best_result = results.get_best_result(metric="ptl/val_mcc", mode="max")
best_result.config


# # Submission

# In[ ]:


trainer = Trainer()
best_config = best_result.config['train_loop_config']
datamodule = DataModule(best_config,
                        num_workers=4, 
                        val_size=0.0) # no retraining - we load the best model from checkpoint
with best_result.checkpoint.as_directory() as checkpoint_dir:
    # The model state dict was saved under `model.pt` by the training function
    model = BinaryClassificationModel.load_from_checkpoint(os.path.join(checkpoint_dir, "checkpoint.ckpt"))
    
pred = trainer.predict(model, datamodule)
df = df_test.with_columns(pl.Series(name='ffnn', values=torch.concat(pred).squeeze().numpy())) # assign numpy array for edible to polars column
df = df.with_columns(pl.when(pl.col('ffnn') > 0.5).then(pl.lit("e")).otherwise(pl.lit("p")).alias('class'))
df.head()


# In[ ]:


df.select(['id', 'class']).write_csv(file=os.path.join(KAGGLE_WORKING_PATH, 'submission.csv'))

