#!/usr/bin/env python
# coding: utf-8

# <a name='T'>
# 
# <p style="padding: 20px;
#           background-color: black;
#           font-family: computermodern;
#           color: white;
#           font-size: 200%;
#           text-align: center;
#           border-radius: 40px 20px;
#           ">All Best Tabular Classifiers - Comparative Study<br>
#           </p>
# <p style="font-family: computermodern;
#           color: #000000;
#           font-size: 175%;
#           text-align: center;
#           ">Created by Alexandre Le Mercier on the 8th of September 2024<br>
#              </p>
#     
#     
# 
# ![TITLE IMAGE](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F17037041%2Fa176d8ab625c73936eb9db7ea29687a6%2FClassifiers%20Showdown%20Enhanced.png?generation=1725790655686010&alt=media)
# 
# <a id="TOC"></a>
# 
# <div style="background-color: #e8f5e9; border-left: 10px solid #66bb6a; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
#     <h1 style="color: #388e3c;">Table of Contents</h1>
#     <ul style="list-style-type: none;">
#         <li><a href="#s0" style="color: #2e7d32;"><strong>1. Introduction - Motivation and Game Rules</strong></a></li>
#         <li><a href="#s1" style="color: #2e7d32;"><strong>2. Imports and Constants</strong></a></li>
#         <li><a href="#s2" style="color: #2e7d32;"><strong>3. Data Preprocessing</strong></a></li>
#         <li><a href="#s3" style="color: #2e7d32;"><strong>4. First Experiment - Initial Performance</strong></a></li>
#         <li><a href="#s4" style="color: #2e7d32;"><strong>5. Second Experiment - Optuna Optimization</strong></a></li>
#         <li><a href="#s5" style="color: #2e7d32;"><strong>6. Third Experiment - Data Dependance</strong></a></li>
#         <li><a href="#s6" style="color: #2e7d32;"><strong>7. Fourth Experiment - Evasion Attacks</strong></a></li>
#         <li><a href="#s7" style="color: #2e7d32;"><strong>8. Results and Conclusions</strong></a></li>
# </div>

# <a id="s0"></a>
# <div style="color: black; background-color: #E6E6FA; padding: 10px; border-left: 5px solid purple; border-radius: 5px;">
# <h1>1. Introduction - Motivation and Game Rules</h1>
# </div>
# 
# [Back to table of contents](#TOC)
# 
# <div style="color: black; background-color: #ffcccc; padding: 10px; border-left: 5px solid #ff3333; border-radius: 5px;">
#            <strong>Remark:</strong> if you are runing this notebook for the first time and getting errors, think about checking the library versions compatibility! Some updates sometimes make older codes crash. You can force a given version to be installed with <code>!pip uninstall the_library -y</code>, then <code>!pip install the_library==1.1.1</code> if you want e.g. to install version 1.1.1. The versions I use are listed in section 2.
# </div>
# 
# ## Motivation
# 
# We are all using ensemble methods for tabular classification. Usually, `RandomForestClassifier` from scikit-learn is the perfect tool you need: it is easy to use, needs very few data preprocessing and generally reaches outstanding performances without needing hyperparameter tuning. However, scikit-learn doesn't have the monopole of tabular classification. In the [machine learning intermediate tutorial on Kaggle](https://www.kaggle.com/learn/intermediate-machine-learning) for instance, [XGBoost](https://xgboost.readthedocs.io/en/stable/) is designed as "*the most accurate modeling technique for structured data.*".
# 
# Nevertheless, from my personal experience and the fellow AI students I met, XGBoost didn't reach expected performances. Is `XGBoostClassifier` harder to use than its scikit-learn's fellow? Maybe. Does it work in more specific contexts, that do not correspond to the use cases my mates worked on? Maybe. The purpose of this study is to **give a general overview of those classifier's performances and best use case**, so that you have some experimental knowledge about those models. The goal is to help you deciding which classifier to pick up for your specific use case, and a few tips to use them correctly (though I do not intend to make an in-depth analysis of the libraries insights, there already exists several interesting notebooks on Kaggle about that).
# 
# ## Classifiers
# 
# I will compare the following classifiers over very similar tests. You will find below a table of the used classifiers and their corresponding preprocessing needs:
# 
# | Model          | Can Skip Normalization | Can Skip Scaling | Can Skip Missing Value Handling | Can Skip Categorical Value Handling |
# |----------------|------------------------|------------------|-------------------------------|------------------------------------|
# | Random Forest from Sklearn  | Yes                    | Yes              | No                            | No                                 |
# | Gradient Boosting from Sklearn| Yes                  | Yes              | No                            | No                                 |
# | XGBoost        | Yes                    | Yes              | Yes                           | No                                 |
# | LightGBM       | Yes                    | Yes              | Yes                           | Yes (with native support)          |
# | CatBoost       | Yes                    | Yes              | Yes                           | Yes (handles categorical directly) |
# | AdaBoost from Sklearn       | Yes                    | Yes              | No                            | No                                 |
# | ExtraTrees from Sklearn     | Yes                    | Yes              | No                            | No                                 |
# | Sequential from Keras | No         | No               | No                            | No                                 |
# 
# Of note, GPU acceleration will **not** be used through this notebook, to guarantee a good balance into computational ressources used.
# 
# ## Datasets
# 
# I choose 3 datasets (so far) among the most popular ones in Kaggle, that are all very different from each other, plus 3 other that I added later (v1.1) for diffent reasons. Those are the [credit card fraud detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (imbalanced binary classification), the [Netflix TV shows](https://www.kaggle.com/datasets/shivamb/netflix-shows) (multi-label classification), the [chest Xray pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (image classification, because it is still interesting to see how tabular models work on images), the space Titanic, the MNIST and the ICR datasets.
# 
# Regarding the pneumonia dataset, I will extract some features from the image itself using the library I developped: `fast-skimage`. CatBoost classification has proved to be very efficient in my [previous study about brain cancer classification](https://www.kaggle.com/code/alexandrelemercier/99-8-accuracy-on-brain-tumor-classification), so I think it is worth the shot trying it again. Tabular classifiers are much faster than Keras ones, and we could hence spare lots of intense training resources.
# 
# <div style="color: black; background-color: #ffcccc; padding: 10px; border-left: 5px solid #ff3333; border-radius: 5px;">
#            <strong>Remark:</strong> you are looking at version 1.1 of this study. Other datasets might be added in later versions.
# </div>
# 
# ## Experimental Tests
# 
# The following experimental tests will be conducted to compare those classifiers:
# 
# 1. Accuracy and F1-score computing with minimum possible preprocessing and no parameter tuning. This will give an insight about **training time** and **ease of use** (because we don't only want to optimize computational resources, but also time we spend developping models);
# 2. The same after an Optuna hyperparameter optimization (I am not a big fan of grid search). This will give an insight about **performance**. Note that the ease of use score can be derived from the difference between initial performance and optimized performance.
# 3. Same models, but training on a very small portion of each dataset. A good model should extract patterns from as few data as possible. This will give an insight about **data dependance**.
# 4. Finally, conduct [evasion attacks](https://www.ibm.com/docs/en/watsonx/saas?topic=atlas-evasion-attack) on the best models using the [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox). This will give an insight on **adversarial robustness**.
# 
# ## Disclaimer
# 
# The conclusions of this study are drawn from pure empirical analyses over a limited number of publicly available datasets. At version 1.0, there were only 3 of them. The confidence into my results is limited by the small number of datasets, and my limited experience about each library. **Feel free to improve this study by suggesting new implementation techniques or/and new datasets.**
# 
# Also, I will use ChatGPT 4o for several tasks. I believe it is an incredible tool that can make us all win a lot of time. Some will disagree, but I don't see any problem in using it for machine learning projects, and even strongly encourage collegues to do so.
# 
# ## Previous Work
# 
# This work is inspired from my first experiments on linear models (cf. [The Regularization Rumble](https://www.kaggle.com/code/alexandrelemercier/mlmfo-episode-1-the-regularization-rumble)). This is my most successfull notebook on Kaggle (at the time I am writing those lines), but I was much less experimented as I am now. I hope this study will be more usefull that the previous one.
# 
# As mentionned before, the methods used on image classification derives from my [brain tumor study](https://www.kaggle.com/code/alexandrelemercier/99-8-accuracy-on-brain-tumor-classification). I am particularly proud of this work, which was very well received by my image processing teacher and allowed me to obtain an internship at Sony R&D.
# 
# Finally, I discovered that autoencoders (AE) could reach high performance in anomaly detection during my work on [Quantized Autoencoders (QAE)](https://www.kaggle.com/code/alexandrelemercier/quantized-autoencoder-qae-ids-for-iot-devices) where I built the extensive QAE class to detect cyberattacks. This is also a notebook I am proud of.
# 
# ## Updates Log
# 
# You are currently looking at version **1.1** of this study. This is the original version.
# 

# <a id="s1"></a>
# <div style="color: black; background-color: #E6E6FA; padding: 10px; border-left: 5px solid purple; border-radius: 5px;">  
#     <h1>2. Imports and Constants</h1>
# </div>
# 
# [Back to table of contents](#TOC)

# In[ ]:


# Main Imports
#get_ipython().system('pip install tensorflow==2.17.0')
#get_ipython().system('pip install numpy')
#get_ipython().system('pip install matplotlib')
#get_ipython().system('pip install seaborn')
#get_ipython().system('pip install tqdm')
#get_ipython().system('pip install pandas')
#get_ipython().system('pip install scikit-learn')
#get_ipython().system('pip install xgboost')
#get_ipython().system('pip install catboost')
#get_ipython().system('pip install lightgbm')
#get_ipython().system('pip install fast-skimage==0.3.1')

# Other Important Imports
#get_ipython().system('pip install optuna')
#get_ipython().system('pip install adversarial-robustness-toolbox')

# Create Output Directories (c.f. constants names)
#get_ipython().system('mkdir Figures')
#get_ipython().system('mkdir Models')
#get_ipython().system('mkdir Dataframes')

# Extensions
#get_ipython().run_line_magic('load_ext', 'autoreload')


# In[ ]:


#get_ipython().run_line_magic('autoreload', '')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn
import xgboost
import lightgbm
import catboost
import sys
import os
import optuna
import pickle
import warnings
import art
import fast_skimage

from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split

imports = [tf, np, sns, pd, sklearn, xgboost, lightgbm, catboost, optuna, art]

for imp in imports:
    print(imp.__name__.ljust(20), imp.__version__)
    
RANDOM_SEED = 5
FIGURES_DIR = "Figures/"
MODELS_DIR = "Models/"
OUTPUT_DATA_DIR = "Dataframes/"
INPUT_DATA_PATHS = ["/kaggle/input/netflix-shows",
                   "/kaggle/input/creditcardfraud",
                   "/kaggle/input/chest-xray-pneumonia/chest_xray"]   

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

warnings.filterwarnings("ignore",category=FutureWarning)

for path in INPUT_DATA_PATHS:
    sys.path.append(os.path.abspath(path))

# Utility Functions
def load(model_name):
    if model_name.endswith(".pkl"):
        with open(model_name, 'rb') as file:
            return pickle.load(file)
    else:
        return tf.keras.models.load_model(model_name)

def save(model, model_name):
    if model_name.endswith(".pkl"):
        with open(MODELS_DIR+model_name, 'wb') as file:
            pickle.dump(model, file)
    else:
        model.save(MODELS_DIR+model_name)

def to_numpy(arr):
    try:
        return np.ndarray(arr)
    except:
        return arr


# <a id="s2"></a>
# <div style="color: black; background-color: #E6E6FA; padding: 10px; border-left: 5px solid purple; border-radius: 5px;">
# <h1>3. Data Preprocessing</h1>
# </div>
# 
# [Back to table of contents](#TOC)
# 
# 
# ## 3.1 Minimum Preprocessing for Tabular Data
# 
# This includes categorical data and missing value handling. Catboost will have a specially dedicated pipeline using the Pool object. We will create `df_credit`, `df_netflix` along with `X_train_credit`, etc. knowing that the label for credit card fraud is "Class" and the one for Netflix is "duration". For Netflix, we need to drop every row which "duration" is not in the form "X Seasons" or "1 Season". The labels will be the number of seasons, so we will just encode "1", "2", etc. in a new column "Class". Of course, we need to drop the previous "duration" column.
# 
# The Netflix dataset needs in-depth preprocessing, because most of features are categorical:
# - "show_id": to drop
# - "type": OHE encoding because few labels (2)
# - "title": extract the number of words in the title, then drop the title
# - "director": lots of null values, sometimes several directors, lots of different directors... keep for Pool's Catboost but drop for the others
# - "cast": extract actors in a list with ", " separator, detect the 20 most cited actors, OHE encoding for them. There are null values here too. In this case, put "0" everywhere for OHE
# - "country": 5017 different values. OHE over the 20 most cited, "other" for the others.
# - "date_added": extract the month (the content before the space character) and OHE other the 12 months
# - "realease_year": the only numerical feature
# - "rating": 3k+ different values, OHE for the 20 most common
# - "listed_in": 8k+ different values, do the same as for the actors (OHE over most cited categories)
# - "description": drop
# 
# We also make some categorical preprocessing for the Spaceship Titanic.
# 
# Of note, the columns from which OHE was extracted should be dropped. No OHE should be applied to Pool's Catboost.

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Load tabular datasets
df_credit = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
df_netflix = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')
df_ICR = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/train.csv')
df_sptit = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')
df_mnist = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')


# In[ ]:


# 3.1 Minimum Preprocessing for Credit Card Dataset (df_credit)
# Handling missing values
df_credit.dropna(inplace=True)

# Split credit card dataset
X_credit = df_credit.drop('Class', axis=1)
y_credit = df_credit['Class']
X_train_credit, X_test_credit, y_train_credit, y_test_credit = train_test_split(X_credit, y_credit, test_size=0.2, random_state=RANDOM_SEED)

# 3.2 Minimum Preprocessing for Netflix Dataset (df_netflix)

# 1. Drop rows where "duration" is not in "X Seasons" or "1 Season" form
df_netflix = df_netflix[df_netflix['duration'].str.contains(r'\d+ Season', na=False)]

# 2. Encode "duration" into a "Class" column with numerical values
df_netflix['Class'] = df_netflix['duration'].str.extract('(\d+)').astype(int)
df_netflix.drop('duration', axis=1, inplace=True)

# 3. Drop unnecessary columns
df_netflix.drop(['show_id', 'title', 'description'], axis=1, inplace=True)

# 4. One-Hot Encode "type" (few labels)
ohe_type = OneHotEncoder(drop='first', sparse=False)
encoded_type = ohe_type.fit_transform(df_netflix[['type']])
df_netflix[ohe_type.get_feature_names_out(['type'])] = encoded_type
df_netflix.drop('type', axis=1, inplace=True)

# 5. Handle "director" column (keep for CatBoost, drop for others)
df_netflix_director_catboost = df_netflix[['director']]  # Store for CatBoost
df_netflix.drop('director', axis=1, inplace=True)

# 6. Handle "cast" column: Extract top 20 actors, One-Hot Encode
top_actors = df_netflix['cast'].str.split(', ').explode().value_counts().head(20).index
for actor in top_actors:
    df_netflix[f'actor_{actor}'] = df_netflix['cast'].apply(lambda x: 1 if pd.notnull(x) and actor in x else 0)
df_netflix.drop('cast', axis=1, inplace=True)

# 7. Handle "country" column: One-Hot Encode top 20 countries, set others to "other"
top_countries = df_netflix['country'].value_counts().head(20).index
df_netflix['country'] = df_netflix['country'].apply(lambda x: x if x in top_countries else 'other')
ohe_country = OneHotEncoder(drop='first', sparse=False)
encoded_country = ohe_country.fit_transform(df_netflix[['country']])
df_netflix[ohe_country.get_feature_names_out(['country'])] = encoded_country
df_netflix.drop('country', axis=1, inplace=True)

# 8. Handle "date_added" column: Extract month and One-Hot Encode
df_netflix['month_added'] = df_netflix['date_added'].str.split(' ').str[0]
df_netflix.drop('date_added', axis=1, inplace=True)
ohe_month = OneHotEncoder(drop='first', sparse=False)
encoded_month = ohe_month.fit_transform(df_netflix[['month_added']])
df_netflix[ohe_month.get_feature_names_out(['month_added'])] = encoded_month
df_netflix.drop('month_added', axis=1, inplace=True)

# 9. "release_year" is numeric, no changes needed

# 10. Handle "rating" column: One-Hot Encode top 20 ratings
top_ratings = df_netflix['rating'].value_counts().head(20).index
df_netflix['rating'] = df_netflix['rating'].apply(lambda x: x if x in top_ratings else 'other')
ohe_rating = OneHotEncoder(drop='first', sparse=False)
encoded_rating = ohe_rating.fit_transform(df_netflix[['rating']])
df_netflix[ohe_rating.get_feature_names_out(['rating'])] = encoded_rating
df_netflix.drop('rating', axis=1, inplace=True)

# 11. Handle "listed_in" column: Extract top 20 categories, One-Hot Encode
top_categories = df_netflix['listed_in'].str.split(', ').explode().value_counts().head(20).index
for category in top_categories:
    df_netflix[f'category_{category}'] = df_netflix['listed_in'].apply(lambda x: 1 if pd.notnull(x) and category in x else 0)
df_netflix.drop('listed_in', axis=1, inplace=True)

# Split Netflix dataset into training and test sets
X_netflix = df_netflix.drop('Class', axis=1)
y_netflix = df_netflix['Class']
X_train_netflix, X_test_netflix, y_train_netflix, y_test_netflix = train_test_split(X_netflix, y_netflix, test_size=0.2, random_state=RANDOM_SEED)


# At this point, the preprocessed data is ready for model training


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

# Load ICR dataset
df_ICR = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/train.csv')

# 1. Drop the 'Id' column
df_ICR.drop(['Id', 'EJ'], axis=1, inplace=True)

# 2. Handle missing values by dropping rows containing NaNs
df_ICR.dropna(inplace=True)  # Drop rows with missing values across the entire dataset

# 3. Separate features and target
X_ICR = df_ICR.drop('Class', axis=1)
y_ICR = df_ICR['Class']

X_train_ICR, X_test_ICR, y_train_ICR, y_test_ICR = train_test_split(X_ICR, y_ICR, test_size=0.2, random_state=RANDOM_SEED)



# In[ ]:


X_train_ICR.columns


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings

# Load the dataset
df_sptit = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')

# 3.4 Minimum Preprocessing for Spaceship Titanic Dataset (df_sptit)
warnings.filterwarnings('ignore')

# Drop unnecessary columns: PassengerId, HomePlanet, Name, Destination
df_sptit.drop(['PassengerId', 'HomePlanet', 'Name', 'Destination'], axis=1, inplace=True)

# Handle missing values (optional: decide how to handle missing values)
df_sptit.fillna(df_sptit.mean(numeric_only=True), inplace=True)  # Fill numeric columns with mean
df_sptit.fillna(df_sptit.mode().iloc[0], inplace=True)  # Fill categorical columns with mode

# Split the 'Cabin' column into three components (X/Y/Z)
df_sptit[['Cabin_X', 'Cabin_Y', 'Cabin_Z']] = df_sptit['Cabin'].str.split('/', expand=True)
df_sptit.drop('Cabin', axis=1, inplace=True)  # Drop original Cabin column

# One-Hot Encode the Cabin_X component
ohe_cabin_x = OneHotEncoder(drop='first', sparse=False)
encoded_cabin_x = ohe_cabin_x.fit_transform(df_sptit[['Cabin_X']])
encoded_cabin_x_df = pd.DataFrame(encoded_cabin_x, columns=ohe_cabin_x.get_feature_names_out(['Cabin_X'])).astype(int)  # Convert to int
df_sptit = pd.concat([df_sptit.drop('Cabin_X', axis=1), encoded_cabin_x_df], axis=1)

# One-Hot Encode the Cabin_Y component
ohe_cabin_y = OneHotEncoder(drop='first', sparse=False)
encoded_cabin_y = ohe_cabin_y.fit_transform(df_sptit[['Cabin_Y']])
encoded_cabin_y_df = pd.DataFrame(encoded_cabin_y, columns=ohe_cabin_y.get_feature_names_out(['Cabin_Y'])).astype(int)  # Convert to int
df_sptit = pd.concat([df_sptit.drop('Cabin_Y', axis=1), encoded_cabin_y_df], axis=1)

# One-Hot Encode the Cabin_Z component
ohe_cabin_z = OneHotEncoder(drop='first', sparse=False)
encoded_cabin_z = ohe_cabin_z.fit_transform(df_sptit[['Cabin_Z']])
encoded_cabin_z_df = pd.DataFrame(encoded_cabin_z, columns=ohe_cabin_z.get_feature_names_out(['Cabin_Z'])).astype(int)  # Convert to int
df_sptit = pd.concat([df_sptit.drop('Cabin_Z', axis=1), encoded_cabin_z_df], axis=1)

# One-Hot Encode CryoSleep and VIP
ohe_cryosleep_vip = OneHotEncoder(drop='first', sparse=False)
encoded_cryosleep_vip = ohe_cryosleep_vip.fit_transform(df_sptit[['CryoSleep', 'VIP']])
encoded_cryosleep_vip_df = pd.DataFrame(encoded_cryosleep_vip, columns=ohe_cryosleep_vip.get_feature_names_out(['CryoSleep', 'VIP'])).astype(int)  # Convert to int
df_sptit = pd.concat([df_sptit.drop(['CryoSleep', 'VIP'], axis=1), encoded_cryosleep_vip_df], axis=1)

# Split Spaceship Titanic dataset into X and y
X_sptit = df_sptit.drop('Transported', axis=1)
y_sptit = df_sptit['Transported'].astype(int)  # Convert to 0/1 for binary classification

X_sptit = X_sptit.drop(X_sptit.select_dtypes(include=['object']).columns, axis=1)

# Train-test split for the Titanic dataset
X_train_sptit, X_test_sptit, y_train_sptit, y_test_sptit = train_test_split(X_sptit, y_sptit, test_size=0.2, random_state=RANDOM_SEED)



# In[ ]:


X_sptit.info()


# In[ ]:


# 3.5 Minimum Preprocessing for MNIST Dataset (df_mnist)

df_mnist = df_mnist[:3000]

# Normalize the pixel values (0-255) to the range (0-1)
X_mnist = df_mnist.drop('label', axis=1) / 255.0  # Scale pixel values
y_mnist = df_mnist['label']  # Labels (digits from 0 to 9)

# Train-test split for the MNIST dataset
X_train_mnist, X_test_mnist, y_train_mnist, y_test_mnist = train_test_split(X_mnist, y_mnist, test_size=0.2, random_state=RANDOM_SEED)



# In[ ]:


import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.stats import kurtosis, skew  # Importing kurtosis and skew functions
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy

# Define directories
data_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# Image size (ensuring all images are resized to the same dimensions)
img_size = (299, 299)  # You can adjust the size depending on your needs

# Helper function to calculate first-order features
def calculate_first_order_features(image_array):
    mean = np.mean(image_array)
    variance = np.var(image_array)
    std_dev = np.std(image_array)
    skewness = skew(image_array.flatten())
    kurt = kurtosis(image_array.flatten())
    
    return [mean, variance, std_dev, skewness, kurt]

# Helper function to calculate second-order features using Grey Level Co-occurrence Matrix (GLCM)
def calculate_second_order_features(image_array):
    # Convert to grayscale if necessary
    if len(image_array.shape) == 3:  # If RGB, convert to grayscale
        image_array = np.mean(image_array, axis=2).astype(np.uint8)

    # Calculate GLCM (graycomatrix requires uint8 image)
    glcm = graycomatrix(image_array, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    asm = graycoprops(glcm, 'ASM')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    entropy = shannon_entropy(image_array)
    
    return [contrast, energy, homogeneity, correlation, asm, dissimilarity, entropy]

# Helper function to extract features from a directory
def extract_features_from_directory(directory):
    features = []
    labels = []
    for label in ['NORMAL', 'PNEUMONIA']:
        path = os.path.join(directory, label)
        for file_name in tqdm(os.listdir(path), desc=f"Processing {label} images"):
            file_path = os.path.join(path, file_name)
            try:
                # Load image and resize
                img = Image.open(file_path).resize(img_size)
                img_array = np.array(img)
                
                # First-order features
                first_order_features = calculate_first_order_features(img_array)
                
                # Second-order features
                second_order_features = calculate_second_order_features(img_array)
                
                # Combine all features
                all_features = first_order_features + second_order_features
                features.append(all_features)
                
                # Append label: 0 for 'NORMAL', 1 for 'PNEUMONIA'
                labels.append(0 if label == 'NORMAL' else 1)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                
    return np.array(features), np.array(labels)

# Extract features from train and test directories
X_train_xray, y_train_xray = extract_features_from_directory(train_dir)
X_test_xray, y_test_xray = extract_features_from_directory(test_dir)

# To check the shapes of the resulting feature arrays:


# In[ ]:


from sklearn.model_selection import train_test_split
from catboost import Pool

# Spaceship Titanic Dataset (df_sptit)
df_sptit = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')  # Reload the Titanic dataset
df_sptit.drop(['PassengerId', 'HomePlanet', 'Name', 'Destination'], axis=1, inplace=True)  # Drop unnecessary columns
df_sptit.fillna(df_sptit.mean(numeric_only=True), inplace=True)  # Fill numeric columns with mean
df_sptit.fillna(df_sptit.mode().iloc[0], inplace=True)  # Fill categorical columns with mode

# Split the 'Cabin' column into three components (X/Y/Z)
df_sptit[['Cabin_X', 'Cabin_Y', 'Cabin_Z']] = df_sptit['Cabin'].str.split('/', expand=True)
df_sptit.drop('Cabin', axis=1, inplace=True)  # Drop original Cabin column

# Prepare X and y for Titanic
X_sptit = df_sptit.drop('Transported', axis=1)
y_sptit = df_sptit['Transported'].astype(int)  # Convert to 0/1 for binary classification

# Handle missing values in categorical features for Titanic
categorical_features_sptit = ['Cabin_X', 'Cabin_Y', 'Cabin_Z', 'CryoSleep', 'VIP']
X_sptit[categorical_features_sptit] = X_sptit[categorical_features_sptit].fillna('Unknown')

# Split the Titanic dataset into train and test sets
X_train_sptit, X_test_sptit, y_train_sptit, y_test_sptit = train_test_split(X_sptit, y_sptit, test_size=0.2, random_state=RANDOM_SEED)

# Create Pool objects for Spaceship Titanic
pool_sptit_train = Pool(data=X_train_sptit, label=y_train_sptit, cat_features=categorical_features_sptit)
pool_sptit_test = Pool(data=X_test_sptit, label=y_test_sptit, cat_features=categorical_features_sptit)


# Netflix Dataset (df_netflix2)
df_netflix2 = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')
df_netflix2 = df_netflix2[df_netflix2['duration'].str.contains(r'\d+ Season', na=False)]
df_netflix2['Class'] = df_netflix2['duration'].str.extract('(\d+)').astype(int)
df_netflix2.drop('duration', axis=1, inplace=True)
df_netflix2.drop(['show_id', 'title', 'description'], axis=1, inplace=True)

# Handle missing values in categorical features for Netflix
categorical_features_netflix = ['director', 'cast', 'listed_in', 'type', 'country', 'date_added', 'release_year', 'rating']
X_netflix2 = df_netflix2.drop('Class', axis=1)
y_netflix2 = df_netflix2['Class']
X_netflix2[categorical_features_netflix] = X_netflix2[categorical_features_netflix].fillna('Unknown')

# Split the Netflix dataset into train and test sets
X_train_netflix2, X_test_netflix2, y_train_netflix2, y_test_netflix2 = train_test_split(X_netflix2, y_netflix2, test_size=0.2, random_state=RANDOM_SEED)

# Create Pool objects for Netflix
pool_netflix_train = Pool(data=X_train_netflix2, label=y_train_netflix2, cat_features=categorical_features_netflix)
pool_netflix_test = Pool(data=X_test_netflix2, label=y_test_netflix2, cat_features=categorical_features_netflix)

# Credit Card Dataset (df_credit)
# No categorical features in this dataset, so no cat_features are needed.
pool_credit_train = Pool(data=X_train_credit, label=y_train_credit)
pool_credit_test = Pool(data=X_test_credit, label=y_test_credit)

# Separate features and target
X_ICR = df_ICR.drop('Class', axis=1)
y_ICR = df_ICR['Class']

# Train-test split
X_train_ICR, X_test_ICR, y_train_ICR, y_test_ICR = train_test_split(X_ICR, y_ICR, test_size=0.2, random_state=RANDOM_SEED)

"""# Specifying the categorical column index for 'EJ'
cat_features = ['EJ']  # Specify 'EJ' as categorical feature"""
cat_features= list()

# Create the CatBoost Pool with the 'EJ' column marked as categorical
pool_ICR_train = Pool(data=X_train_ICR, label=y_train_ICR, cat_features=cat_features)
pool_ICR_test = Pool(data=X_test_ICR, label=y_test_ICR, cat_features=cat_features)

# MNIST Dataset (df_mnist)
# No categorical features, directly create Pool objects
pool_mnist_train = Pool(data=X_train_mnist, label=y_train_mnist)
pool_mnist_test = Pool(data=X_test_mnist, label=y_test_mnist)

# X-Ray Dataset (train_dir and test_dir are preprocessed using feature extraction)
# No categorical features, directly create Pool objects
pool_xray_train = Pool(data=X_train_xray, label=y_train_xray)
pool_xray_test = Pool(data=X_test_xray, label=y_test_xray)



# In[ ]:


df_credit.info()


# In[ ]:


df_netflix.info()


# In[ ]:


from sklearn.preprocessing import LabelEncoder

# Encode class labels to ensure they are sequential starting from 0
label_encoder_netflix = LabelEncoder()
y_train_netflix_encoded = label_encoder_netflix.fit_transform(y_train_netflix)
y_test_netflix_encoded = label_encoder_netflix.transform(y_test_netflix)


# In[ ]:


df_netflix.Class.unique()


# In[ ]:


# Function to clean column names
def sanitize_column_names(df):
    df.columns = df.columns.str.replace(r'[^\w\s]', '_', regex=True)  # Replace special characters with underscores
    return df

X_train_netflix = sanitize_column_names(X_train_netflix)
X_test_netflix = sanitize_column_names(X_test_netflix)
X_train_credit = sanitize_column_names(X_train_credit)
X_test_credit = sanitize_column_names(X_test_credit)
X_test_netflix.columns


# In[ ]:


# Check the shapes of the training and testing data

# Ensure the dimensions match
assert X_train_credit.shape[0] == y_train_credit.shape[0], "Mismatch between X_train_credit and y_train_credit"
assert X_test_credit.shape[0] == y_test_credit.shape[0], "Mismatch between X_test_credit and y_test_credit"


# In[ ]:


# Helper function to check and print rows containing NaNs
def check_for_nans(df, df_name):
    nan_rows = df[df.isnull().any(axis=1)]
    if not nan_rows.empty:
        print(f"Dropping {len(nan_rows)} rows containing NaN values from {df_name} dataset.")
        df.dropna(inplace=True)

# Check for object columns and handle them for df_credit
check_for_nans(df_credit, 'df_credit')
for col in df_credit.select_dtypes(include='object').columns:
    try:
        df_credit[col] = df_credit[col].astype(int)  # Attempt to convert to int
    except ValueError:
        print(f"Dropping column '{col}' from df_credit as it cannot be converted to int.")
        df_credit.drop(col, axis=1, inplace=True)

# Check for object columns and handle them for df_netflix
check_for_nans(df_netflix, 'df_netflix')
for col in df_netflix.select_dtypes(include='object').columns:
    try:
        df_netflix[col] = df_netflix[col].astype(int)  # Attempt to convert to int
    except ValueError:
        print(f"Dropping column '{col}' from df_netflix as it cannot be converted to int.")
        df_netflix.drop(col, axis=1, inplace=True)

# Check for object columns and handle them for df_ICR
check_for_nans(df_ICR, 'df_ICR')
for col in df_ICR.select_dtypes(include='object').columns:
    try:
        df_ICR[col] = df_ICR[col].astype(int)  # Attempt to convert to int
    except ValueError:
        print(f"Dropping column '{col}' from df_ICR as it cannot be converted to int.")
        df_ICR.drop(col, axis=1, inplace=True)

# Check for object columns and handle them for df_sptit
check_for_nans(df_sptit, 'df_sptit')
for col in df_sptit.select_dtypes(include='object').columns:
    try:
        df_sptit[col] = df_sptit[col].astype(int)  # Attempt to convert to int
    except ValueError:
        print(f"Dropping column '{col}' from df_sptit as it cannot be converted to int.")
        df_sptit.drop(col, axis=1, inplace=True)

# Check for object columns and handle them for df_mnist
check_for_nans(df_mnist, 'df_mnist')
for col in df_mnist.select_dtypes(include='object').columns:
    try:
        df_mnist[col] = df_mnist[col].astype(int)  # Attempt to convert to int
    except ValueError:
        print(f"Dropping column '{col}' from df_mnist as it cannot be converted to int.")
        df_mnist.drop(col, axis=1, inplace=True)




# <a id="s3"></a>
# <div style="color: black; background-color: #E6E6FA; padding: 10px; border-left: 5px solid purple; border-radius: 5px;">
# <h1>4. First Experiment - Initial Performance</h1>
# </div>
# 
# [Back to table of contents](#TOC)
# 
# Now, we will apply `RandomForestClassifier`, `XGBoostClassifier`, `GradientBoostingClassifier`, `AdaBoostClassifier`, `ExtraTreesClassifier` and `LGBMClassifier` to `df_netflix` and `df_credit`. Then, we will apply `CatBoostClassifier` to the Pool objects.

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from time import time

# 4. First Experiment - Initial Performance

# Define a function to train and evaluate the models
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    t1 = time()  # Start time
    model.fit(X_train, y_train)  # Fit the model
    y_pred = model.predict(X_test)  # Make predictions
    acc = accuracy_score(y_test, y_pred)  # Accuracy metric
    f1 = f1_score(y_test, y_pred, average='weighted')  # F1 score metric
    t = time() - t1  # Compute elapsed time
    print("    Time (seconds) taken for fitting and computing metrics:", t)
    return acc, f1, t

# Reordered classifiers based on speed for Credit dataset
models = {
    'AdaBoost': AdaBoostClassifier(),  # Fastest
    'ExtraTrees': ExtraTreesClassifier(),
    'LightGBM': LGBMClassifier(verbose=-1),
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier()
}


# Netflix dataset (df_netflix is preprocessed as per previous sections)
X_train_netflix, X_test_netflix, y_train_netflix, y_test_netflix = train_test_split(
    df_netflix.drop('Class', axis=1), df_netflix['Class'], test_size=0.2, random_state=RANDOM_SEED
)

# Credit card dataset (df_credit is preprocessed as per previous sections)
X_train_credit, X_test_credit, y_train_credit, y_test_credit = train_test_split(
    df_credit.drop('Class', axis=1), df_credit['Class'], test_size=0.2, random_state=RANDOM_SEED
)

X_train_netflix = sanitize_column_names(X_train_netflix)
X_test_netflix = sanitize_column_names(X_test_netflix)
X_train_credit = sanitize_column_names(X_train_credit)
X_test_credit = sanitize_column_names(X_test_credit)


# Dictionary to store results
results = {}

# Separate execution for XGBoost

# XGBoost for ICR dataset (Binary Classification)
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', objective='binary:logistic')
acc_ICR, f1_ICR, t_ICR = train_and_evaluate_model(model, X_train_ICR, y_train_ICR, X_test_ICR, y_test_ICR)

# XGBoost for Netflix dataset
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
acc_netflix, f1_netflix, t_netflix = train_and_evaluate_model(model, X_train_netflix, y_train_netflix_encoded, X_test_netflix, y_test_netflix_encoded)

# XGBoost for Credit dataset (Binary Classification)
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', objective='binary:logistic')
acc_credit, f1_credit, t_credit = train_and_evaluate_model(model, X_train_credit, y_train_credit, X_test_credit, y_test_credit)

# XGBoost for Spaceship Titanic dataset (Binary Classification)
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', objective='binary:logistic')
acc_sptit, f1_sptit, t_sptit = train_and_evaluate_model(model, X_train_sptit, y_train_sptit, X_test_sptit, y_test_sptit)

# XGBoost for MNIST dataset (Multi-class Classification)
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', objective='multi:softmax')
acc_mnist, f1_mnist, t_mnist = train_and_evaluate_model(model, X_train_mnist, y_train_mnist, X_test_mnist, y_test_mnist)

# XGBoost for Chest X-ray dataset (Binary Classification)
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', objective='binary:logistic')
acc_xray, f1_xray, t_xray = train_and_evaluate_model(model, X_train_xray, y_train_xray, X_test_xray, y_test_xray)

# Store XGBoost results
results["XGBoost"] = {
    'Netflix': {'Accuracy': acc_netflix, 'F1-score': f1_netflix, 'Time': t_netflix},
    'Credit': {'Accuracy': acc_credit, 'F1-score': f1_credit, 'Time': t_credit},
    'ICR': {'Accuracy': acc_ICR, 'F1-score': f1_ICR, 'Time': t_ICR},
    'Spaceship Titanic': {'Accuracy': acc_sptit, 'F1-score': f1_sptit, 'Time': t_sptit},
    'MNIST': {'Accuracy': acc_mnist, 'F1-score': f1_mnist, 'Time': t_mnist},
    'Chest X-ray': {'Accuracy': acc_xray, 'F1-score': f1_xray, 'Time': t_xray}
}

results["XGBoost"]


# In[ ]:


# Evaluate each model on all datasets
X_train_ICR.dropna(inplace=True)
X_test_ICR.dropna(inplace=True)

for model_name, model in tqdm(models.items()):
    print(f"\nTraining and evaluating {model_name} on all datasets...")
    try:
        # Credit card dataset
        print(f"Training and evaluating {model_name} on Credit Card dataset...")
        acc_credit, f1_credit, t_credit = train_and_evaluate_model(model, X_train_credit, y_train_credit, X_test_credit, y_test_credit)
    except Exception as e:
        print(e)
        
    try:
        # Netflix dataset
        print(f"Training and evaluating {model_name} on Netflix dataset...")
        acc_netflix, f1_netflix, t_netflix = train_and_evaluate_model(model, X_train_netflix, y_train_netflix, X_test_netflix, y_test_netflix)
    except Exception as e:
        print(e)
        
    try:
        # ICR dataset
        print(f"Training and evaluating {model_name} on ICR dataset...")
        acc_ICR, f1_ICR, t_ICR = train_and_evaluate_model(model, X_train_ICR, y_train_ICR, X_test_ICR, y_test_ICR)
    except Exception as e:
        print(e)
        
    try:
        # Spaceship Titanic dataset
        print(f"Training and evaluating {model_name} on Spaceship Titanic dataset...")
        acc_sptit, f1_sptit, t_sptit = train_and_evaluate_model(model, X_train_sptit, y_train_sptit, X_test_sptit, y_test_sptit)
    except Exception as e:
        print(e)
        
    try:
        # MNIST dataset
        print(f"Training and evaluating {model_name} on MNIST dataset...")
        acc_mnist, f1_mnist, t_mnist = train_and_evaluate_model(model, X_train_mnist, y_train_mnist, X_test_mnist, y_test_mnist)
    except Exception as e:
        print(e)
        
    try:
        # Chest X-ray dataset (Pneumonia detection)
        print(f"Training and evaluating {model_name} on Chest X-ray dataset...")
        acc_xray, f1_xray, t_xray = train_and_evaluate_model(model, X_train_xray, y_train_xray, X_test_xray, y_test_xray)
    except Exception as e:
        print(e)
        
        
    # Store results for each dataset
    results[model_name] = {
            'Credit': {'Accuracy': acc_credit, 'F1-score': f1_credit, 'Time': t_credit},
            'Netflix': {'Accuracy': acc_netflix, 'F1-score': f1_netflix, 'Time': t_netflix},
            'ICR': {'Accuracy': acc_ICR, 'F1-score': f1_ICR, 'Time': t_ICR},
            'Spaceship Titanic': {'Accuracy': acc_sptit, 'F1-score': f1_sptit, 'Time': t_sptit},
            'MNIST': {'Accuracy': acc_mnist, 'F1-score': f1_mnist, 'Time': t_mnist},
            'Chest X-ray': {'Accuracy': acc_xray, 'F1-score': f1_xray, 'Time': t_xray}
    }

# Print results for all models and datasets
for model_name, result in results.items():
    print(f"\n{model_name} Results:")
    for dataset, metrics in result.items():
        print(f"{dataset} - Accuracy: {metrics['Accuracy']:.4f}, F1-score: {metrics['F1-score']:.4f}, Time: {metrics['Time']:.2f} seconds")


# In[ ]:


from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score
from time import time

# CatBoost Classifier
catboost_model = CatBoostClassifier(silent=True)

# Dictionary to store CatBoost results
catboost_results = {}

# Evaluate CatBoost on Netflix dataset
t1 = time()
catboost_model.fit(pool_netflix_train)
y_pred_netflix_cat = catboost_model.predict(pool_netflix_test)
acc_netflix_cat = accuracy_score(y_test_netflix2, y_pred_netflix_cat)
f1_netflix_cat = f1_score(y_test_netflix2, y_pred_netflix_cat, average='weighted')
t_netflix_cat = time() - t1

# Evaluate CatBoost on Credit dataset
t1 = time()
catboost_model.fit(pool_credit_train)
y_pred_credit_cat = catboost_model.predict(pool_credit_test)
acc_credit_cat = accuracy_score(y_test_credit, y_pred_credit_cat)
f1_credit_cat = f1_score(y_test_credit, y_pred_credit_cat, average='weighted')
t_credit_cat = time() - t1
# Evaluate CatBoost on ICR dataset
t1 = time()
catboost_model.fit(pool_ICR_train)
y_pred_ICR_cat = catboost_model.predict(pool_ICR_test)
acc_ICR_cat = accuracy_score(y_test_ICR, y_pred_ICR_cat)
f1_ICR_cat = f1_score(y_test_ICR, y_pred_ICR_cat, average='weighted')
t_ICR_cat = time() - t1

# Evaluate CatBoost on Spaceship Titanic dataset
t1 = time()
catboost_model.fit(pool_sptit_train)
y_pred_sptit_cat = catboost_model.predict(pool_sptit_test)
acc_sptit_cat = accuracy_score(y_test_sptit, y_pred_sptit_cat)
f1_sptit_cat = f1_score(y_test_sptit, y_pred_sptit_cat, average='weighted')
t_sptit_cat = time() - t1

# Evaluate CatBoost on MNIST dataset
t1 = time()
catboost_model.fit(pool_mnist_train)
y_pred_mnist_cat = catboost_model.predict(pool_mnist_test)
acc_mnist_cat = accuracy_score(y_test_mnist, y_pred_mnist_cat)
f1_mnist_cat = f1_score(y_test_mnist, y_pred_mnist_cat, average='weighted')
t_mnist_cat = time() - t1

# Evaluate CatBoost on Chest X-ray dataset
t1 = time()
catboost_model.fit(pool_xray_train)
y_pred_xray_cat = catboost_model.predict(pool_xray_test)
acc_xray_cat = accuracy_score(y_test_xray, y_pred_xray_cat)
f1_xray_cat = f1_score(y_test_xray, y_pred_xray_cat, average='weighted')
t_xray_cat = time() - t1

# Store CatBoost results
catboost_results['CatBoost'] = {
    'Netflix': {'Accuracy': acc_netflix_cat, 'F1-score': f1_netflix_cat, 'Time': t_netflix_cat},
    'Credit': {'Accuracy': acc_credit_cat, 'F1-score': f1_credit_cat, 'Time': t_credit_cat},
    'ICR': {'Accuracy': acc_ICR_cat, 'F1-score': f1_ICR_cat, 'Time': t_ICR_cat},
    'Spaceship Titanic': {'Accuracy': acc_sptit_cat, 'F1-score': f1_sptit_cat, 'Time': t_sptit_cat},
    'MNIST': {'Accuracy': acc_mnist_cat, 'F1-score': f1_mnist_cat, 'Time': t_mnist_cat},
    'Chest X-ray': {'Accuracy': acc_xray_cat, 'F1-score': f1_xray_cat, 'Time': t_xray_cat}
}

# Print CatBoost results
for dataset_name, metrics in catboost_results['CatBoost'].items():
    print(f"{dataset_name} - Accuracy: {metrics['Accuracy']:.4f}, F1-score: {metrics['F1-score']:.4f}, Time: {metrics['Time']:.2f} seconds")


# In[ ]:


import pandas as pd

# Convert the results dictionary into a DataFrame
def results_to_dataframe(results):
    data = []
    for model_name, model_results in results.items():
        # Loop through all datasets dynamically
        for dataset_name, dataset_results in model_results.items():
            data.append({
                'Model': model_name,
                'Dataset': dataset_name,
                'Accuracy': dataset_results['Accuracy'],
                'F1-score': dataset_results['F1-score'],
                'Time (seconds)': dataset_results['Time']
            })
    
    # Create a DataFrame
    df_results = pd.DataFrame(data)
    return df_results

# Transform the results into a DataFrame
df_results = results_to_dataframe(results)


# In[ ]:


# Assuming df_results is already created and contains the results for other models

# Convert CatBoost results into a DataFrame
def catboost_results_to_dataframe(catboost_results):
    data = []
    for dataset_name, metrics in catboost_results['CatBoost'].items():
        data.append({
            'Model': 'CatBoost',
            'Dataset': dataset_name,
            'Accuracy': metrics['Accuracy'],
            'F1-score': metrics['F1-score'],
            'Time (seconds)': metrics['Time']
        })
    
    # Create a DataFrame for CatBoost results
    df_catboost = pd.DataFrame(data)
    return df_catboost

# Create the CatBoost results DataFrame
df_catboost_results = catboost_results_to_dataframe(catboost_results)

# Append CatBoost results to the existing df_results DataFrame
df_results = pd.concat([df_results, df_catboost_results], ignore_index=True)

# Print updated results


# In[ ]:


# Filepath to save the results
output_file = OUTPUT_DATA_DIR + "model_results.csv"

# Save the DataFrame to a CSV file
df_results.to_csv(output_file, index=False)



# <a id="s4"></a>
# <div style="color: black; background-color: #E6E6FA; padding: 10px; border-left: 5px solid purple; border-radius: 5px;">
# <h1>5. Second Experiment - Optuna Optimization</h1>
# </div>
# 
# [Back to table of contents](#TOC)

# <a id="s5"></a>
# <div style="color: black; background-color: #E6E6FA; padding: 10px; border-left: 5px solid purple; border-radius: 5px;">
# <h1>6. Third Experiment - Data Dependance</h1>
# </div>
# 
# [Back to table of contents](#TOC)

# <a id="s6"></a>
# <div style="color: black; background-color: #E6E6FA; padding: 10px; border-left: 5px solid purple; border-radius: 5px;">
# <h1>7. Fourth Experiment - Evasion Attacks</h1>
# </div>
# 
# [Back to table of contents](#TOC)

# <a id="s7"></a>
# <div style="color: black; background-color: #E6E6FA; padding: 10px; border-left: 5px solid purple; border-radius: 5px;">
# <h1>8. Results and Conclusions</h1>
# </div>
# 
# [Back to table of contents](#TOC)

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gmean

# Function to compute mean of F1-scores across all datasets
def compute_mean(df, method="geometric"):
    if method == "geometric":
        df['Mean_F1'] = df[['F1-score_Credit', 'F1-score_ChestXray', 'F1-score_ICR', 'F1-score_MNIST', 'F1-score_Netflix', 'F1-score_SpaceshipTitanic']].apply(lambda row: gmean(row), axis=1)
    elif method == "arithmetic":
        df['Mean_F1'] = df[['F1-score_Credit', 'F1-score_ChestXray', 'F1-score_ICR', 'F1-score_MNIST', 'F1-score_Netflix', 'F1-score_SpaceshipTitanic']].mean(axis=1)
    return df

# Let the user choose between 'geometric' or 'arithmetic'
mean_method = "geometric"  # Change to 'arithmetic' if you want to compute arithmetic mean

# Pivot the DataFrame to have one row per model, with F1-scores from different datasets as columns
df_pivot = df_results.pivot(index='Model', columns='Dataset', values='F1-score').reset_index()

# Rename columns for clarity
df_pivot.columns.name = None
df_pivot.columns = ['Model', 'F1-score_Credit', 'F1-score_ChestXray', 'F1-score_ICR', 'F1-score_MNIST', 'F1-score_Netflix', 'F1-score_SpaceshipTitanic']

# Apply the function to compute the chosen mean
df_pivot = compute_mean(df_pivot, method=mean_method)

# Sort the DataFrame by Mean F1-scores in descending order
df_sorted = df_pivot.sort_values(by="Mean_F1", ascending=False)

# Add the mean to the x-axis labels
df_sorted['Model_with_Mean'] = df_sorted.apply(lambda row: f"{row['Model']} ({row['Mean_F1']:.4f})", axis=1)

# Set the plot aesthetics
sns.set(style="whitegrid")

# Define a fixed color palette for each dataset
dataset_colors = {
    "F1-score_Netflix": "C2",      # Blue
    "F1-score_Credit": "C2",       # Orange
    "F1-score_ICR": "C2",          # Green
    "F1-score_SpaceshipTitanic": "C2",  # Red
    "F1-score_MNIST": "C2",        # Purple
    "F1-score_ChestXray": "C2"     # Brown
}

# Create the plot
plt.figure(figsize=(16, 10))

A = 0.3

# Plot F1-scores for all datasets with fixed colors
"""barplot_netflix = sns.barplot(x="Model_with_Mean", y="F1-score_Netflix", data=df_sorted, color=dataset_colors["F1-score_Netflix"], label="Netflix", alpha=A)
barplot_credit = sns.barplot(x="Model_with_Mean", y="F1-score_Credit", data=df_sorted, color=dataset_colors["F1-score_Credit"], label="Credit", alpha=A)
barplot_ICR = sns.barplot(x="Model_with_Mean", y="F1-score_ICR", data=df_sorted, color=dataset_colors["F1-score_ICR"], label="ICR", alpha=A)
barplot_sptit = sns.barplot(x="Model_with_Mean", y="F1-score_SpaceshipTitanic", data=df_sorted, color=dataset_colors["F1-score_SpaceshipTitanic"], label="Spaceship Titanic", alpha=A)
barplot_mnist = sns.barplot(x="Model_with_Mean", y="F1-score_MNIST", data=df_sorted, color=dataset_colors["F1-score_MNIST"], label="MNIST", alpha=A)
barplot_xray = sns.barplot(x="Model_with_Mean", y="F1-score_ChestXray", data=df_sorted, color=dataset_colors["F1-score_ChestXray"], label="Chest X-ray", alpha=A)
"""
barplot_netflix = sns.barplot(x="Model_with_Mean", y="F1-score_Netflix", data=df_sorted, color=dataset_colors["F1-score_Netflix"], alpha=A)
barplot_credit = sns.barplot(x="Model_with_Mean", y="F1-score_Credit", data=df_sorted, color=dataset_colors["F1-score_Credit"], alpha=A)
barplot_ICR = sns.barplot(x="Model_with_Mean", y="F1-score_ICR", data=df_sorted, color=dataset_colors["F1-score_ICR"], alpha=A)
barplot_sptit = sns.barplot(x="Model_with_Mean", y="F1-score_SpaceshipTitanic", data=df_sorted, color=dataset_colors["F1-score_SpaceshipTitanic"], alpha=A)
barplot_mnist = sns.barplot(x="Model_with_Mean", y="F1-score_MNIST", data=df_sorted, color=dataset_colors["F1-score_MNIST"], alpha=A)
barplot_xray = sns.barplot(x="Model_with_Mean", y="F1-score_ChestXray", data=df_sorted, color=dataset_colors["F1-score_ChestXray"], alpha=A)

# Add annotations for Netflix bar values
for p in barplot_netflix.patches:
    barplot_netflix.annotate(format(p.get_height(), '.4f'),
                             (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='center', xytext=(0, 9),
                             textcoords='offset points')

# Add annotations for Credit bar values
for p in barplot_credit.patches:
    barplot_credit.annotate(format(p.get_height(), '.4f'),
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 9),
                            textcoords='offset points')

# Add labels and title
plt.title(f"Model F1-Score Comparison on All Datasets with {mean_method.capitalize()} Mean Ranking", fontsize=16)
plt.xlabel(f"Model ({mean_method.capitalize()} Mean F1-score)", fontsize=12)
plt.ylabel("F1-score", fontsize=12)
plt.ylim([0.45, 1.03])

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha="right")

# Add legend
#plt.legend(title="Dataset")

# Save the plot
figure_file = FIGURES_DIR + f"model_f1_score_comparison_{mean_method}_mean_all_datasets.png"
plt.tight_layout()
plt.savefig(figure_file)

# Display the sorted DataFrame for confirmation
df_sorted


# In[ ]:


import matplotlib.pyplot as plt

# Compute geometric mean for F1-scores (already done in previous steps)
df_pivot = compute_mean(df_pivot, method="geometric")

# Aggregating the total execution time for each model across datasets
df_time = df_results.groupby('Model')['Time (seconds)'].sum().reset_index()
df_time.columns = ['Model', 'Total_Time_Seconds']

# Merge the geometric mean F1 scores with the total time data
df_merged_time = df_pivot[['Model', 'Mean_F1']].merge(df_time, on='Model')

# Sort by total execution time in ascending order
df_merged_time_sorted = df_merged_time.sort_values(by="Total_Time_Seconds", ascending=True)

# Plotting the scatter plot
plt.figure(figsize=(10, 6))

# Scatter plot of Geometric Mean F1-score vs Total Time
plt.scatter(df_merged_time_sorted['Total_Time_Seconds'], df_merged_time_sorted['Mean_F1'], color='blue', s=100)

# Add annotations for each point (Model names)
for i in range(df_merged_time_sorted.shape[0]):
    plt.text(df_merged_time_sorted['Total_Time_Seconds'].iloc[i], 
             df_merged_time_sorted['Mean_F1'].iloc[i], 
             df_merged_time_sorted['Model'].iloc[i], 
             fontsize=9, ha='right')

# Add labels and title
plt.title("Geometric Mean F1-score vs Execution Time", fontsize=16)
plt.xlabel("Total Execution Time (Seconds)", fontsize=12)
plt.ylabel("Geometric Mean F1-score", fontsize=12)

# Save the plot
figure_file = FIGURES_DIR + f"model_f1_score_comparison_{mean_method}_mean_all_datasets_with_execution_time.png"
plt.tight_layout()
plt.savefig(figure_file)

# Show the plot
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

# Pivot the DataFrame to have one row per model, with F1-scores for different datasets as columns
df_f1_matrix = df_results.pivot(index='Model', columns='Dataset', values='F1-score')

# Set plot size and aesthetics
plt.figure(figsize=(10, 8))
sns.set(style="whitegrid")

# Create the heatmap using seaborn
sns.heatmap(df_f1_matrix, annot=True, cmap="coolwarm", cbar_kws={'label': 'F1-score'}, linewidths=0.5)

# Add labels and title
plt.title("F1-Score Heatmap for Each Model and Dataset", fontsize=16)
plt.xlabel("Dataset", fontsize=12)
plt.ylabel("Model", fontsize=12)

# Save the plot
figure_file = FIGURES_DIR + f"model_f1_score_comparison_grid_plot.png"
plt.tight_layout()
plt.savefig(figure_file)

# Show the plot
plt.tight_layout()
plt.show()


# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sizes of the individual datasets
dataset_sizes = {
    'Credit': X_train_credit.shape[0],        # Size of Credit Card Fraud dataset
    'Netflix': X_train_netflix.shape[0],      # Size of Netflix dataset
    'ICR': X_train_ICR.shape[0],              # Size of ICR dataset
    'Spaceship Titanic': X_train_sptit.shape[0],  # Size of Spaceship Titanic dataset
    'MNIST': X_train_mnist.shape[0],          # Size of MNIST dataset
    'Chest X-ray': X_train_xray.shape[0]      # Size of Chest X-ray dataset
}

# Convert dataset sizes into a DataFrame and sort by size
df_sizes = pd.DataFrame(list(dataset_sizes.items()), columns=['Dataset', 'Size']).sort_values(by='Size')

# Pivot the F1-score results DataFrame to get F1-scores for each dataset and model
df_f1_scores = df_results.pivot(index='Model', columns='Dataset', values='F1-score').reset_index()

# Sort columns by dataset size
df_f1_scores = df_f1_scores[['Model'] + df_sizes['Dataset'].tolist()]

# Melt the DataFrame to get it into long format for seaborn
df_f1_long = pd.melt(df_f1_scores, id_vars='Model', var_name='Dataset', value_name='F1-score')

# Merge with dataset sizes to get size info in the plot
df_f1_long = pd.merge(df_f1_long, df_sizes, on='Dataset')

# Plotting the line plot
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")

# Line plot for each model
sns.lineplot(data=df_f1_long, x='Dataset', y='F1-score', hue='Model', marker='o', palette='Set2')

# Add dataset sizes as labels on the x-axis
plt.xticks(ticks=range(len(df_sizes)), labels=[f"{dataset} ({size:,})" for dataset, size in zip(df_sizes['Dataset'], df_sizes['Size'])], rotation=45)

# Add labels and title
plt.title("F1-score Across Datasets Ordered by Size", fontsize=16)
plt.xlabel("Dataset (Size in Rows)", fontsize=12)
plt.ylabel("F1-score", fontsize=12)

# Save the plot
figure_file = FIGURES_DIR + f"model_f1_score_size_comparison.png"
plt.tight_layout()
plt.savefig(figure_file)

# Show the plot
plt.tight_layout()
plt.show()


# <p style="padding: 20px;
#           background-color: green;
#           font-family: computermodern;
#           color: white;
#           font-size: 200%;
#           text-align: center;
#           border-radius: 40px 20px;
#           ">Thank you! If you found this useful please like 👍</p>
