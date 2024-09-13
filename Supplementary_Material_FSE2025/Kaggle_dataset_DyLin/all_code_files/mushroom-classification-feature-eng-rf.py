#!/usr/bin/env python
# coding: utf-8

# <h1 style="font-size:30px"> 🍄Mushroom Classification🍄 </h1>

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "/kaggle/input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# <a id="1"></a>
# # <p style="background-color:#8B4513; font-family:calibri; font-size:130%; color:#F2EBE6; text-align:center; border-radius:15px 50px; padding:10px">1 | Import Libraries</p>
# 
# ⬆️ [Table of Contents](#contents_tabel)

# In[ ]:


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

from matplotlib import colors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import linkage, dendrogram

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report, RocCurveDisplay, ConfusionMatrixDisplay

#get_ipython().run_line_magic('matplotlib', 'inline')


# <a id="2"></a>
# # <p style="background-color:#8B4513; font-family:calibri; font-size:130%; color:#F2EBE6; text-align:center; border-radius:15px 50px; padding:10px">2 | Read Dataset</p>
# 
# ⬆️ [Table of Contents](#contents_tabel)

# In[ ]:


# Set the option to display all columns
pd.set_option('display.max_columns', None)


# In[ ]:


train_df = pd.read_csv("/kaggle/input/playground-series-s4e8/train.csv")
train_df.head()


# In[ ]:


test_df = pd.read_csv("/kaggle/input/playground-series-s4e8/test.csv")
test_df.head()


# <a id="3"></a>
# # <p style="background-color:#8B4513; font-family:calibri; font-size:130%; color:#F2EBE6; text-align:center; border-radius:15px 50px; padding:10px">3 | Dataset Overview</p>
# 
# ⬆️ [Table of Contents](#contents_tabel)

# In[ ]:


train_df.info(show_counts=True)


# <div style="background-color: #F2EBE6; padding: 15px; border-radius: 10px; border: 2px solid #8B4513;">
#     <h2 style="color: #8B4513;">Inference 🍄</h2>
#     <ul>
#         <li>The dataset contains <span style="font-weight:600">3116945</span> entries.</li>
#         <li>Each entry represents a hypothetical mushroom with various attributes.</li>
#         <li>There are <span style="font-weight: bold;">21</span> columns (ignoring "id") in the dataset.</li>
#         <li>The columns represent features such as cap-diameter, cap-shape, cap-surface, cap-color, does-bruise-or-bleed, gill-attachment, gill-spacing, gill-color, stem-height, stem-width, stem-root, stem-surface, stem-color, veil-type, veil-color, has-ring, ring-type, spore-print-color, habitat, and season.</li>
#         <li>Most columns have missing data, indicated by non-null counts less than the total number of entries, suggesting the presence of incomplete information.</li>
#         <li>The target variable is '<span style="font-weight: bold;">class</span>', which represents whether the mushroom is poisonous or edible.</li>
#         <li>The features include a mix of numerical (float64) and categorical (object) data types.</li>
#         <li>The dataset requires preprocessing to handle missing values and encoding categorical variables before further analysis or modeling.</li>
#     </ul>
# </div>
# 

# <a id="3.1"></a>
# # <b><span style='color:#333'>3.1 |</span><span style='color:#743089;font-weight:bold'> Summary Statistics for Numerical Variables</span></b>

# In[ ]:


train_df.drop(columns="id").describe().T


# <div style="background-color: #F2EBE6; padding: 15px; border-radius: 10px; border: 2px solid #8B4513;">
#     <h2 style="color: #8B4513;">Inference 🍄</h2>
#     <h3>Cap Diameter:</h3>
#     <ul>
#         <li>The average cap diameter of mushrooms in the dataset is approximately <span style="font-weight: bold;">6.31 cm</span>, with a standard deviation of <span style="font-weight: bold;">4.66 cm</span>.</li>
#         <li>The minimum cap diameter is <span style="font-weight: bold;">0.03 cm</span>, while the maximum cap diameter is <span style="font-weight: bold;">80.67 cm</span>.</li>
#         <li>The majority of mushrooms have a cap diameter between <span style="font-weight: bold;">3.32 cm</span> (25th percentile) and <span style="font-weight: bold;">8.24 cm</span> (75th percentile), with the median being <span style="font-weight: bold;">5.75 cm</span>.</li>
#         <li>Understanding cap diameter distribution is crucial for optimizing harvesting processes, estimating yield, and ensuring product quality in mushroom cultivation businesses.</li>
#     </ul>
#     <h3>Stem Height:</h3>
#     <ul>
#         <li>The average stem height of mushrooms is approximately <span style="font-weight: bold;">6.35 cm</span>, with a standard deviation of <span style="font-weight: bold;">2.70 cm</span>.</li>
#         <li>The minimum stem height is <span style="font-weight: bold;">0.00 cm</span>, while the maximum stem height is <span style="font-weight: bold;">88.72 cm</span>.</li>
#         <li>The majority of mushrooms have a stem height between <span style="font-weight: bold;">4.67 cm</span> (25th percentile) and <span style="font-weight: bold;">7.41 cm</span> (75th percentile), with the median being <span style="font-weight: bold;">5.88 cm</span>.</li>
#         <li>Understanding stem height distribution is important for optimizing planting density, assessing crop health, and predicting resource requirements in mushroom cultivation businesses.</li>
#         <li>It's worth noting that some mushrooms may have a stem height of zero or very close to zero, as not all mushrooms have a distinct stem. Mushroom structures can vary widely.</li>
#     </ul>
#     
# <h3>Stem Width:</h3>
#     <ul>
#         <li>The average stem width of mushrooms is approximately <span style="font-weight: bold;">11.15 mm</span>, with a standard deviation of <span style="font-weight: bold;">8.10 mm</span>.</li>
#         <li>The minimum stem width is <span style="font-weight: bold;">0.00 mm</span>, while the maximum stem width is <span style="font-weight: bold;">102.90 mm</span>.</li>
#         <li>The majority of mushrooms have a stem width between <span style="font-weight: bold;">4.97 mm</span> (25th percentile) and <span style="font-weight: bold;">15.63 mm</span> (75th percentile), with the median being <span style="font-weight: bold;">9.65 mm</span>.</li>
#         <li>Understanding stem width distribution is important for optimizing packaging, transportation, and storage processes in mushroom processing and distribution businesses.</li>
#         <li>Similarly, some mushrooms may have a stem width of zero or very close to zero, depending on their species and morphology.</li>
#     </ul>
# </div>
# 
# 

# <a id="3.2"></a>
# # <b><span style='color:#333'>3.2 |</span><span style='color:#743089;font-weight:bold'> Summary Statistics for Categorical Variables</span></b>

# In[ ]:


train_df.describe(include="object")


# <div style="background-color: #F2EBE6; padding: 15px; border-radius: 10px; border: 2px solid #8B4513;">
#     <h2 style="color: #8B4513;">Inference 🍄</h2>
#     <h3>Mushroom Characteristics:</h3>
#     <ul>
#         <li>There are <span style="font-weight: bold;">2</span> classes in the dataset, representing whether the mushroom is <span style="font-weight: bold;">poisonous (p)</span> or <span style="font-weight: bold;">edible (e)</span>.</li>
#         <li>The most common cap shape is <span style="font-weight: bold;">x</span> (convex), which occurs <span style="font-weight: bold;">1,436,026</span> times.</li>
#         <li>The most prevalent cap surface is <span style="font-weight: bold;">t</span>, which occurs <span style="font-weight: bold;">460,777</span> times.</li>
#         <li>The most frequent cap color is <span style="font-weight: bold;">n</span> (brown), with <span style="font-weight: bold;">1,359,542</span> occurrences.</li>
#         <li>The majority of mushrooms (<span style="font-weight: bold;">2,569,743</span>) do not bruise or bleed, indicated by <span style="font-weight: bold;">f</span> (false).</li>
#         <li>The most common gill attachment is <span style="font-weight: bold;">a</span> , occurring <span style="font-weight: bold;">646,034</span> times.</li>
#         <li>Most mushrooms have close gill spacing, with <span style="font-weight: bold;">1,331,054</span> instances of <span style="font-weight: bold;">c</span> (close).</li>
#         <li>The dominant gill color is <span style="font-weight: bold;">w</span> occurring <span style="font-weight: bold;">931,538</span> times.</li>
#         <li>The most common stem root type is <span style="font-weight: bold;">b</span>, with <span style="font-weight: bold;">165,801</span> occurrences.</li>
#         <li>Many mushrooms have smooth stem surfaces (<span style="font-weight: bold;">327,610</span> instances of <span style="font-weight: bold;">s</span>).</li>
#         <li>The prevalent stem color is <span style="font-weight: bold;">w</span>, occurring <span style="font-weight: bold;">1,196,637</span> times.</li>
#         <li>Most mushrooms have a <span style="font-weight: bold;">veil type</span>, represented by <span style="font-weight: bold;">u</span>, with <span style="font-weight: bold;">159,373</span> occurrences.</li>
#         <li>The majority of mushrooms (<span style="font-weight: bold;">2,368,820</span>) have no rings, indicated by <span style="font-weight: bold;">f</span> (false).</li>
#         <li>Most mushrooms have ring type, represented by <span style="font-weight: bold;">f</span>, occurring <span style="font-weight: bold;">2,477,170</span> times.</li>
#         <li>The most prevalent spore print color is <span style="font-weight: bold;">k</span>, with <span style="font-weight: bold;">107,310</span> occurrences.</li>
#         <li>Regarding habitat, the majority of mushrooms (<span style="font-weight: bold;">2,177,573</span>) are found in <span style="font-weight: bold;">d</span>.</li>
#         <li>In terms of season, the most common season for mushroom growth is <span style="font-weight: bold;">a</span> (autumn), occurring <span style="font-weight: bold;">1,543,321</span> times.</li>
#     </ul>
# </div>
# 

# <a id="4"></a>
# # <p style="background-color:#8B4513; font-family:calibri; font-size:130%; color:#F2EBE6; text-align:center; border-radius:15px 50px; padding:10px">4 | Univariate Analysis</p>
# 
# ⬆️ [Table of Contents](#contents_tabel)

# <a id="4.1"></a>
# # <b><span style='color:#333'>4.1 |</span><span style='color:#743089;font-weight:bold'> Continuous/Numerical Variables</span></b>

# In[ ]:


continuous_features = train_df.drop(columns="id",axis=1).describe().columns
df_continuous = train_df[continuous_features]


# In[ ]:


# Set up the subplot
fig, ax = plt.subplots(ncols=3, figsize=(20, 5))

# Define the mushroom color codes
mushroom_color = '#C9B2A0'  # Mushroom background color
highlight_color = '#fff'  # Highlight color for text and annotations
annotation_color = '#743089'  # Annotation color
grid_color = 'lightgrey'  # Color for grid lines

# Loop to plot histograms for each continuous feature
for i, col in enumerate(df_continuous.columns):
    values, bin_edges = np.histogram(df_continuous[col], 
                                     range=(np.floor(df_continuous[col].min()), np.ceil(df_continuous[col].max())))
    
    # Plot histogram
    graph = sns.histplot(data=df_continuous, x=col, bins=bin_edges, kde=True, ax=ax[i],
                         edgecolor='none', color=mushroom_color, alpha=0.6, line_kws={'lw': 3})
    
    # Set labels and ticks
    ax[i].set_xlabel(col, fontsize=15)
    ax[i].set_ylabel('Count', fontsize=12)
    ax[i].set_xticks(np.round(bin_edges, 1))
    ax[i].set_xticklabels(ax[i].get_xticks(), rotation=45)
    ax[i].grid(color=grid_color)
    
    # Annotate each bar with the count
    for j, p in enumerate(graph.patches):
        ax[i].annotate('{}'.format(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height() + 1),
                       ha='center', fontsize=10, fontweight="bold", color=annotation_color)
    
    # Add mean and standard deviation text annotation
    textstr = '\n'.join((
        r'$\mu=%.2f$' % df_continuous[col].mean(),
        r'$\sigma=%.2f$' % df_continuous[col].std()
    ))
    ax[i].text(0.75, 0.9, textstr, transform=ax[i].transAxes, fontsize=12, verticalalignment='top',
               color=highlight_color, bbox=dict(boxstyle='round', facecolor=annotation_color, edgecolor='white', pad=0.5))

plt.suptitle('Distribution of Continuous Variables', fontsize=20, color=highlight_color)
plt.tight_layout()
plt.show()


# <div style="background-color: #F2EBE6; padding: 15px; border-radius: 10px; border: 2px solid #8B4513;">
#     <h2 style="color: #8B4513;">Inference 🍄</h2>
#     <h3>Cap Diameter:</h3>
#     <ul>
#         <li>The average cap diameter of mushrooms in the dataset is approximately <span style="font-weight:bold">6.31 cm</span>, with a standard deviation of <span style="font-weight:bold">4.66 cm</span>. This indicates considerable variability in cap diameters among the mushrooms.</li>
#         <li>The <b>right-skewed</b> distribution suggests that there are mushrooms with exceptionally small cap diameters, which could be of interest to businesses catering to specialty markets or those requiring large-sized mushrooms for specific culinary purposes</li>
#     </ul>
#     <h3>Stem Height:</h3>
#     <ul>
#         <li>The average stem height of mushrooms is approximately <span style="font-weight:bold">6.35 cm</span>, with a standard deviation of <span style="font-weight:bold">2.7 cm</span>. This indicates moderate variability in stem heights across the dataset.</li>
#         <li>The <b>right-skewed</b> distribution suggests that while most mushrooms have relatively shorter stems, there are mushrooms with shorter stems, which could be of interest for certain culinary or decorative applications</li>
#         <li>businesses involved in the processing and distribution of mushrooms may consider stem height variability when designing packaging and transportation solutions to accommodate mushrooms of different sizes</li>
#     </ul>
#     <h3>Stem Width:</h3>
#     <ul>
#         <li>The average stem width of mushrooms is approximately <span style="font-weight:bold">11.15 mm</span>, with a standard deviation of <span style="font-weight:bold">8.10 mm</span>. This indicates considerable variability in stem widths among the mushrooms.</li>
#         <li>The <b>right-skewed</b> distribution suggests the presence of mushrooms with both thin and medium thick stems, highlighting the diversity in mushroom morphology.</li>
#         <li>Mushroom processors and distributors can use this information to optimize packaging, transportation, and storage processes, ensuring that packaging solutions accommodate mushrooms with varying stem widths effectively.</li>
#     </ul>
# </div>
# 

# <a id="4.2"></a>
# # <b><span style='color:#333'>4.2 |</span><span style='color:#743089;font-weight:bold'> Categorical Variables</span></b>

# In[ ]:


categorical_features = train_df.drop(columns="id",axis=1).columns.difference(continuous_features)
df_categorical = train_df[categorical_features]


# In[ ]:


len(categorical_features)


# In[ ]:


# Set up the subplot for a 9x2 layout (9 rows, 2 columns)
fig, ax = plt.subplots(nrows=9, ncols=2, figsize=(20, 50))

# Loop to plot bar charts for each categorical feature in the 9x2 layout
for i, col in enumerate(categorical_features):
    row = i // 2
    col_idx = i % 2
    
    # Calculate frequency percentages, sort by frequency, and take the top 15 categories
    value_counts = train_df[col].value_counts(normalize=True).mul(100).sort_values(ascending=False).head(15)
    
    # Filter out categories with 0% frequency
    value_counts = value_counts[value_counts > 0]
    
    # Plot bar chart
    value_counts.plot(kind='barh', ax=ax[row, col_idx], width=0.8, color='#C9B2A0')
    
    # Add frequency percentages to the bars
    for index, value in enumerate(value_counts):
        ax[row, col_idx].text(value, index, f'{round(value, 1)}%', fontsize=15, weight='bold', va='center')
    
    # Set axis limits and labels
    ax[row, col_idx].set_xlim([0, 95])
    ax[row, col_idx].set_xlabel('Frequency Percentage', fontsize=12)
    ax[row, col_idx].set_title(col, fontsize=20)

# Set the main title and adjust the layout
plt.suptitle('Distribution of Categorical Variables', fontsize=22)
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()


# In[ ]:


train_df['cap-color'].value_counts()


# <div style="background-color: #f0cccc; padding: 10px 12px; border: 2px solid #cc0000; border-radius: 10px;">
#     <h2>📝 Note</h2>
#     <p>Some features contain erroneous or incorrectly entered values, which should be handled to ensure smoother and more accurate analysis.</p>
# </div>

# <a id="5"></a>
# # <p style="background-color:#8B4513; font-family:calibri; font-size:130%; color:#F2EBE6; text-align:center; border-radius:15px 50px; padding:10px">5 | Handling Noise in Categorical variables</p>
# 
# ⬆️ [Table of Contents](#contents_tabel)

# <a id="5.1"></a>
# # <b><span style='color:#333'>5.1 |</span><span style='color:#743089;font-weight:bold'> Cleaning Features</span></b>

# <div style="background-color: #F2EBE6; padding: 15px; border-radius: 10px; border: 2px solid #8B4513;">
#     <h5>i will create a function to handle this:</h5>
# </div>

# In[ ]:


def clean_categorical(value):
    """
    Cleans the input value by removing all non-alphabetic characters and returns
    a cleaned single-letter category or 'Unknown'.

    This function is designed to process categorical data that may contain
    extraneous numeric characters, multiple words, or NaN values. It removes all
    non-alphabetic characters and checks if the remaining string is a single
    alphabetic character. If it is, the function returns this character. If the
    cleaned string is not exactly one letter (e.g., it's empty, numeric, contains
    multiple letters, or is NaN), the function returns 'Unknown'.

    Parameters:
    value (str): The input value to clean. It can contain a mix of letters, numbers,
                 symbols, or be NaN.

    Returns:
    str: A single alphabetic character if valid, otherwise 'Unknown'.
    """

    # Handle NaN values
    if pd.isna(value):
        return 'Unknown'
    
    # Remove non-alphabetic characters
    cleaned_value = re.sub(r'[^a-zA-Z]', '', str(value))
    
    # If the cleaned value is not a single letter, return 'Unknown'
    if len(cleaned_value) != 1:
        return 'Unknown'
    
    return cleaned_value


# In[ ]:


train_df['cap-color'].unique()


# In[ ]:


test_df['cap-color'].unique()


# In[ ]:


# Applying the function to the 'cap-color' column
train_df['cap-color'] = train_df['cap-color'].apply(clean_categorical)
test_df['cap-color'] = test_df['cap-color'].apply(clean_categorical)

# Displaying unique values after cleaning


# <div style="background-color: #F2EBE6; padding: 15px; border-radius: 10px; border: 2px solid #8B4513;">
#     <h5>Let's apply this function to all categorical features:</h5>
# </div>

# In[ ]:


cat_features = ['cap-color', 'cap-shape', 'cap-surface','does-bruise-or-bleed', 
                'gill-attachment', 'gill-color', 'gill-spacing',
       'habitat', 'has-ring', 'ring-type', 'season', 'spore-print-color',
       'stem-color', 'stem-root', 'stem-surface', 'veil-color', 'veil-type']


# In[ ]:


# Apply the cleaning function to each categorical feature in both train and test datasets
for feature in cat_features:
    train_df[feature] = train_df[feature].apply(clean_categorical)
    test_df[feature] = test_df[feature].apply(clean_categorical)


# In[ ]:


# Set up the subplot for a 4x2 layout
fig, ax = plt.subplots(nrows=9, ncols=2, figsize=(20, 50))

# Loop to plot bar charts for each categorical feature in the 4x2 layout
for i, col in enumerate(categorical_features):
    row = i // 2
    col_idx = i % 2
    
    # Calculate frequency percentages
    value_counts = train_df[col].value_counts(normalize=True).mul(100).sort_values()
    
    # Plot bar chart
    value_counts.plot(kind='barh', ax=ax[row, col_idx], width=0.8, color='#C9B2A0')
    
    # Add frequency percentages to the bars
    for index, value in enumerate(value_counts):
        ax[row, col_idx].text(value, index, str(round(value, 1)) + '%', fontsize=15, weight='bold', va='center')
    
    ax[row, col_idx].set_xlim([0, 95])
    ax[row, col_idx].set_xlabel('Frequency Percentage', fontsize=12)
    ax[row, col_idx].set_title(f'{col}', fontsize=20)

plt.suptitle('Filtered Distribution of Categorical Variables', fontsize=22)
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()


# <a id="5.2"></a>
# # <b><span style='color:#333'>5.2 |</span><span style='color:#743089;font-weight:bold'> Replacing low freq categories with "Other"</span></b>

# In[ ]:


def replace_low_frequency_categories_inplace(df, categorical_features, threshold=0.1):
    """
    Replaces categories with low frequency (based on rounded percentage) in categorical features with 'Other'
    directly in the given DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the categorical features.
    categorical_features (list): List of categorical feature column names.
    threshold (float): The frequency percentage below which categories will be replaced with 'Other'.

    Returns:
    None: The function modifies the DataFrame in-place.
    """
    for col in categorical_features:
        # Calculate frequency percentages and round to one decimal place
        value_counts = df[col].value_counts(normalize=True).mul(100).round(1)
        
        # Identify categories with rounded frequency percentages less than or equal to the threshold
        low_freq_categories = value_counts[value_counts <= threshold].index.tolist()
        
        # Replace low frequency categories with 'Other'
        df[col] = df[col].apply(lambda x: 'Other' if str(x) in low_freq_categories else x)


# In[ ]:


# Now, apply the function to your train_df
replace_low_frequency_categories_inplace(train_df, categorical_features, threshold=0.1)


# In[ ]:


# Now, apply the function to your test_df
replace_low_frequency_categories_inplace(test_df, cat_features, threshold=0.1)


# In[ ]:


# Set up the subplot for a 4x2 layout
fig, ax = plt.subplots(nrows=9, ncols=2, figsize=(20, 50))

# Loop to plot bar charts for each categorical feature in the 4x2 layout
for i, col in enumerate(categorical_features):
    row = i // 2
    col_idx = i % 2
    
    # Calculate frequency percentages
    value_counts = train_df[col].value_counts(normalize=True).mul(100).sort_values()
    
    # Plot bar chart
    value_counts.plot(kind='barh', ax=ax[row, col_idx], width=0.8, color='#C9B2A0')
    
    # Add frequency percentages to the bars
    for index, value in enumerate(value_counts):
        ax[row, col_idx].text(value, index, str(round(value, 1)) + '%', fontsize=15, weight='bold', va='center')
    
    ax[row, col_idx].set_xlim([0, 95])
    ax[row, col_idx].set_xlabel('Frequency Percentage', fontsize=12)
    ax[row, col_idx].set_title(f'{col}', fontsize=20)

plt.suptitle('Filtered Distribution of Categorical Variables', fontsize=22)
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()


# <a id="5.3"></a>
# # <b><span style='color:#333'>5.3 |</span><span style='color:#743089;font-weight:bold'> Unique Categories Consistency Check After Cleaning</span></b>

# In[ ]:


def check_same_categories_across_features(df1, df2, features):
    """
    Checks if all specified categorical features have the same unique categories in two DataFrames.

    Parameters:
    df1 (pd.DataFrame): The first DataFrame.
    df2 (pd.DataFrame): The second DataFrame.
    features (list of str): The names of the categorical features to check.

    Returns:
    bool: True if all features have the same unique categories in both DataFrames, False otherwise.
    dict: A dictionary with feature names as keys and boolean values indicating if they have the same unique categories.
    """
    
    all_same = True
    feature_check = {}
    
    for feature in features:
        # Extract unique categories from each DataFrame
        unique_categories_df1 = set(df1[feature].dropna().unique())
        unique_categories_df2 = set(df2[feature].dropna().unique())
        
        # Check if the unique categories are the same in both DataFrames
        same_categories = unique_categories_df1 == unique_categories_df2
        feature_check[feature] = same_categories
        
        if not same_categories:
            all_same = False
    
    return all_same, feature_check


# In[ ]:


# Check if all categorical features have the same unique categories in both DataFrames
all_same, feature_check = check_same_categories_across_features(train_df, test_df, cat_features)

for feature, same in feature_check.items():
    print(f"{feature}: {'Same' if same else 'Different'}")


# <a id="6"></a>
# # <p style="background-color:#8B4513; font-family:calibri; font-size:130%; color:#F2EBE6; text-align:center; border-radius:15px 50px; padding:10px">6 | Bivariate Analysis</p>
# 
# ⬆️ [Table of Contents](#contents_tabel)

# <div style="background-color: #f0cccc; padding: 10px 12px; border: 2px solid #cc0000; border-radius: 10px;">
#      <h2>📝 Note</h2>
#     <br>
#     For bivariate analysis, we will utilize the <code>train_df</code> dataframe as the <code>test_df</code> dataframe lacks a target feature. This ensures consistency in the analysis and modeling process.
# 
# </div>
# 

# <a id="6.1"></a>
# # <b><span style='color:#333'>6.1 |</span><span style='color:#743089;font-weight:bold'> Numerical Features vs Class</span></b>

# In[ ]:


continuous_features


# In[ ]:


# Set color palette
sns.set_palette(['#c68863', '#743089'])

# Create the subplots
fig, ax = plt.subplots(len(continuous_features), 2, figsize=(15,15), gridspec_kw={'width_ratios': [1, 2]})

# Loop through each continuous feature to create barplots and kde plots
for i, col in enumerate(continuous_features):
    # Barplot showing the mean value of the feature for each target category
    graph = sns.barplot(data=train_df, x="class", y=col, ax=ax[i,0])
    
    # KDE plot showing the distribution of the feature for each target category
    sns.kdeplot(data=train_df[train_df["class"]=='e'], x=col, fill=True, linewidth=2, ax=ax[i,1], label='e')
    sns.kdeplot(data=train_df[train_df["class"]=='p'], x=col, fill=True, linewidth=2, ax=ax[i,1], label='p')
    ax[i,1].set_yticks([])
    ax[i,1].legend(title='Mushroom Classification', loc='upper right')
    
    # Add mean values to the barplot
    for cont in graph.containers:
        graph.bar_label(cont, fmt='         %.3g')
        
# Set the title for the entire figure
plt.suptitle('Continuous Features vs Class(target) Distribution', fontsize=22)
plt.tight_layout()                     
plt.show()


# <div style="background-color: #F2EBE6; padding: 15px; border-radius: 10px; border: 2px solid #8B4513;">
#     <h2 style="color: #8B4513;">Inference 🍄</h2>
#     <ul>
#         <li>On average, edible mushrooms (class 0) tend to have a larger cap diameter (<span style="font-weight:bold">7.14 cm</span>) compared to poisonous mushrooms (class 1) with a mean cap diameter of <span style="font-weight:bold">5.62 cm</span>. This aligns with the distribution analysis, which shows that poisonous mushrooms have a higher peak in the cap diameter distribution within the range of 0-6 cm.</li><br>
#         <li>Edible mushrooms also exhibit slightly greater stem height (<span style="font-weight:bold">6.5 cm</span>) on average compared to poisonous mushrooms (<span style="font-weight:bold">6.23 cm</span>). The overlapping distributions of stem height suggest that this feature alone may not be sufficient for distinguishing between edible and poisonous mushrooms.</li><br>
#         <li>Regarding stem width, edible mushrooms have a higher average stem width (<span style="font-weight:bold">12.7 mm</span>) compared to poisonous mushrooms (<span style="font-weight:bold">9.9 mm</span>). The pronounced peak in the stem width distribution of poisonous mushrooms between 0-9 mm highlights the potential of this feature in distinguishing certain poisonous mushroom species.</li>
#     </ul>
# </div>
# 

# <a id="6.2"></a>
# # <b><span style='color:#333'>6.2 |</span><span style='color:#743089;font-weight:bold'> Categorical Features vs Class</span></b>

# In[ ]:


# Calculate the number of required subplots
num_plots = len(cat_features)
num_rows = (num_plots + 1) // 2  # Add one extra row if the number of plots is odd

# Set up the subplot
fig, ax = plt.subplots(nrows=num_rows, ncols=2, figsize=(20, num_rows * 6))

for i, col in enumerate(cat_features):
    # Calculate the row and column index
    x, y = i // 2, i % 2
    
    # Create a cross tabulation showing the proportion of purchased and non-purchased loans for each category of the feature
    cross_tab = pd.crosstab(index=train_df[col], columns=train_df['class'])
    
    # Using the normalize=True argument gives us the index-wise proportion of the data
    cross_tab_prop = pd.crosstab(index=train_df[col], columns=train_df['class'], normalize='index')   
    
    # Define colormap
    cmp = ListedColormap(['#c68863', '#743089'])
    
    # Plot stacked bar charts
    cross_tab_prop.plot(kind='bar', ax=ax[x, y], stacked=True, width=0.8, colormap=cmp,
                        legend=False, ylabel='Proportion', sharey=True)
    
    # Add the proportions and counts of the individual bars to our plot
    for idx, val in enumerate([*cross_tab.index.values]):
        for (proportion, count, y_location) in zip(cross_tab_prop.loc[val],cross_tab.loc[val],cross_tab_prop.loc[val].cumsum()):
            ax[x, y].text(x=idx-0.3, y=(y_location-proportion)+(proportion/2)-0.03,
                         s = f'    {count}\n({np.round(proportion * 100, 1)}%)', 
                         color = "black", fontsize=9, fontweight="bold")
    
    # Add legend
    ax[x, y].legend(title='class', loc=(0.7,0.9), fontsize=8, ncol=2)
    # Set y limit
    ax[x, y].set_ylim([0,1.12])
    # Rotate xticks
    ax[x, y].set_xticklabels(ax[x, y].get_xticklabels(), rotation=0)
    
# Remove empty subplot if the number of plots is odd
if num_plots % 2 != 0:
    fig.delaxes(ax[num_rows-1, 1])
    
# Set title outside the subplots
plt.suptitle('Categorical Features vs Class(target) Stacked Barplots', fontsize=22, y=0.999)
plt.tight_layout()                     
plt.show()


# <div style="background-color: #F2EBE6; padding: 15px; border-radius: 10px; border: 2px solid #8B4513;">
#     <h2 style="color: #8B4513;">Bias Identification 🎯</h2>
#     <p>Features such as <b style="font-weight:bold">spore-print-color</b>, <b style="font-weight:bold">stem-root</b>, <b style="font-weight:bold">stem-surface</b>, and <b style="font-weight:bold">veil-color</b> exhibit bias towards class poisonous. We will address this in the feature engineering section.</p>
#     <p>Additionally, features like <b style="font-weight:bold">habitat</b> with category " <b style="font-weight:bold">u,w</b>" are directly associated with class edible, while category " <b style="font-weight:bold">p</b>" is associated with class poionous. </p>
#     <p>Moreover, <b style="font-weight:bold">stem-surface, veil-color, stem-root</b> with some categories are directly associated with class edible or poisonous,We will address these issues in the feature engineering section.</p>
# </div>

# <a id="7"></a>
# # <p style="background-color:#8B4513; font-family:calibri; font-size:130%; color:#F2EBE6; text-align:center; border-radius:15px 50px; padding:10px">7 | Data Preprocessing & Feature Engineering</p>
# 
# ⬆️ [Table of Contents](#contents_tabel)

# <a id="7.1"></a>
# # <b><span style='color:#333'>7.1 |</span><span style='color:#743089;font-weight:bold'> Remove Directly Related Features</span></b>

# <div style="background-color: #F2EBE6; padding: 15px; border-radius: 10px; border: 2px solid #8B4513;">
#     <h2 style="color: #8B4513;">Inference 🍄</h2>
#     <p>Features like <b>spore-print-color</b>, <b>stem-root</b>, <b>stem-surface</b>, and <b>veil-color</b> exhibit bias towards class 1 (poisonous), and including them in the model would result in <b>data leakage</b>. Removing these features is essential to build a predictive model that can genuinely forecast toxic mushrooms, rather than retrospectively label them.</p>
# </div>
# 

# In[ ]:


train_df.drop(columns=['spore-print-color', 'stem-root', 'stem-surface', 'veil-color'], inplace=True)
test_df.drop(columns=['spore-print-color', 'stem-root', 'stem-surface', 'veil-color'], inplace=True)


# <a id="7.2"></a>
# # <b><span style='color:#333'>7.2 |</span><span style='color:#743089;font-weight:bold'> Addressing Bias Categories</span></b>

# ## cap-color

# In[ ]:


# Replace biased categories 'e' and 'r' with  category 'Other'
train_df['cap-color'] = train_df['cap-color'].replace(['e','r'], 'Other')
test_df['cap-color'] = test_df['cap-color'].replace(['e','r'], 'Other')


# ## ring-type

# In[ ]:


# Replace biased categories 'z' with  category 'Other'
train_df['ring-type'] = train_df['ring-type'].replace('z', 'Other')
test_df['ring-type'] = test_df['ring-type'].replace('z', 'Other')


# ## stem-color

# In[ ]:


# Replace biased categories 'p' and 'r' with category 'Other'
train_df['stem-color'] = train_df['stem-color'].replace(['p', 'r'], 'Other')
test_df['stem-color'] = test_df['stem-color'].replace(['p', 'r'], 'Other')


# ## cap-surface

# In[ ]:


# Replace biased categories 'i' and 'k' with category 'Other'
train_df['cap-surface'] = train_df['cap-surface'].replace(['i', 'k'], 'Other')
test_df['cap-surface'] = test_df['cap-surface'].replace(['i', 'k'], 'Other')


# <a id="7.3"></a>
# # <b><span style='color:#333'>7.3 |</span><span style='color:#743089;font-weight:bold'> Handling Missing Values</span></b>

# In[ ]:


# Calculate the missing values percentage in the Train dataset
missing_values_percentage = (train_df.isnull().sum() / len(train_df)) * 100
missing_values_percentage[missing_values_percentage > 0]


# In[ ]:


# Calculate the missing values percentage in the Test dataset
missing_values_percentage = (test_df.isnull().sum() / len(test_df)) * 100
missing_values_percentage[missing_values_percentage > 0]


# <div style="background-color: #F2EBE6; padding: 15px; border-radius: 10px; border: 2px solid #8B4513;">
#     <p>we will create a method which will fill missing features values based on the most relevant information(observing the correlation between features).</p>
# </div>

# ## TRAIN: cap-diameter

# In[ ]:


# we will check if this feature is correlated to any other feature in dataset
train_df.drop(columns='id').corr(numeric_only=True)['cap-diameter'].drop('cap-diameter').sort_values().plot.barh()


# In[ ]:


# Calculate the average of 'cap-diameter' for each unique 'stem-width'
stem_width_avg = train_df.groupby(by='stem-width', as_index=False)['cap-diameter'].mean(numeric_only=True)
stem_width_avg = stem_width_avg.set_index('stem-width')


# In[ ]:


# Define the fill_cap_diameter function
def fill_cap_diameter(stem_width, cap_diameter):
    if np.isnan(cap_diameter):
        # Look up the mean value for the given stem_width
        return stem_width_avg.loc[stem_width, 'cap-diameter'].round()
    else:
        return cap_diameter

# Apply the function to fill missing values in 'cap-diameter'
train_df['cap-diameter'] = train_df.apply(lambda x: fill_cap_diameter(x['stem-width'], x['cap-diameter']), axis=1)


# ## TEST: cap-diameter

# In[ ]:


# we will check if this feature is correlated to any other feature in dataset
test_df.drop(columns='id').corr(numeric_only=True)['cap-diameter'].drop('cap-diameter').sort_values().plot.barh()


# In[ ]:


# Calculate the average of 'cap-diameter' for each unique 'stem-width'
stem_width_avg = test_df.groupby(by='stem-width', as_index=False)['cap-diameter'].mean(numeric_only=True)
stem_width_avg = stem_width_avg.set_index('stem-width')

# Define the fill_cap_diameter function
def fill_cap_diameter(stem_width, cap_diameter):
    if np.isnan(cap_diameter):
        # Look up the mean value for the given stem_width
        return stem_width_avg.loc[stem_width, 'cap-diameter'].round()
    else:
        return cap_diameter

# Apply the function to fill missing values in 'cap-diameter'
test_df['cap-diameter'] = test_df.apply(lambda x: fill_cap_diameter(x['stem-width'], x['cap-diameter']), axis=1)


# ## TEST: stem-height

# In[ ]:


# we will check if this feature is correlated to any other feature in dataset
test_df.drop(columns='id').corr(numeric_only=True)['stem-height'].drop('stem-height').sort_values().plot.barh()


# In[ ]:


# Calculate the average of 'stem-height' for each unique 'cap-diameter'
cap_diameter_avg = test_df.groupby(by='cap-diameter', as_index=False)['stem-height'].mean(numeric_only=True)
cap_diameter_avg = cap_diameter_avg.set_index('cap-diameter')

# Define the fill_stem_height function
def fill_stem_height(cap_diameter, stem_height):
    if np.isnan(stem_height):
        # Look up the mean value for the given cap_diamter
        return cap_diameter_avg.loc[cap_diameter, 'stem-height'].round()
    else:
        return stem_height

# Apply the function to fill missing values in 'stem-height'
test_df['stem-height'] = test_df.apply(lambda x: fill_stem_height(x['cap-diameter'], x['stem-height']), axis=1)


# <a id="7.4"></a>
# # <b><span style='color:#333'>7.4 |</span><span style='color:#743089;font-weight:bold'> Handling Duplicate Values</span></b>

# In[ ]:


train_df.duplicated().sum()


# In[ ]:


test_df.duplicated().sum()


# <div style="background-color: #F2EBE6; padding: 15px; border-radius: 10px; border: 2px solid #8B4513;">no duplicate data!!

# <a id="7.5"></a>
# # <b><span style='color:#333'>7.5 |</span><span style='color:#743089;font-weight:bold'> Handling Outliers</span></b>

# In[ ]:


continuous_features


# ## train data

# In[ ]:


Q1 = train_df[continuous_features].quantile(0.25)
Q3 = train_df[continuous_features].quantile(0.75)
IQR = Q3 - Q1
outliers_count_specified = ((train_df[continuous_features] < (Q1 - 1.5 * IQR)) | (train_df[continuous_features] > (Q3 + 1.5 * IQR))).sum()

outliers_count_specified


# <div style="background-color: #F2EBE6; padding: 15px; border-radius: 10px; border: 2px solid #8B4513;">
#     <h2>Outliers Identification(train):</h2>
#     <p>Upon identifying outliers for the specified continuous features, we found the following counts of outliers:</p>
#     <ul>
#         <li><b style="font-weight:bold">cap-diameter:</b> 76124 outliers</li>
#         <li><b style="font-weight:bold">stem-height:</b> 132419 outliers</li>
#         <li><b style="font-weight:bold">stem-width:</b> 66481 outliers</li>
#     </ul>
# </div>
# 

# ## test data

# In[ ]:


Q1 = test_df[continuous_features].quantile(0.25)
Q3 = test_df[continuous_features].quantile(0.75)
IQR = Q3 - Q1
outliers_count_specified = ((test_df[continuous_features] < (Q1 - 1.5 * IQR)) | (test_df[continuous_features] > (Q3 + 1.5 * IQR))).sum()

outliers_count_specified


# In[ ]:


Q1 = train_df[continuous_features].quantile(0.25)
Q3 = train_df[continuous_features].quantile(0.75)
IQR = Q3 - Q1
train_outliers_count_specified = (((train_df[continuous_features] < (Q1 - 1.5 * IQR)) | (train_df[continuous_features] > (Q3 + 1.5 * IQR))).sum())/len(train_df)



Q1 = test_df[continuous_features].quantile(0.25)
Q3 = test_df[continuous_features].quantile(0.75)
IQR = Q3 - Q1
test_outliers_count_specified = (((test_df[continuous_features] < (Q1 - 1.5 * IQR)) | (test_df[continuous_features] > (Q3 + 1.5 * IQR))).sum())/len(test_df)



# <div style="background-color: #F2EBE6; padding: 15px; border-radius: 10px; border: 2px solid #8B4513;">
#     <p><strong>Outlier Ratios Comparison:</strong></p>
#     <p>Comparing the outlier ratios between the train and test datasets, we can see that they are generally similar across features. This suggests that the <strong>outlier counts are balanced</strong> relative to the number of rows in both datasets.</p>
# </div>
# 

# <div style="background-color: #f0cccc; padding: 10px 12px; border: 2px solid #cc0000; border-radius: 10px;">
#      <h3>📝 Sensitivity to Outliers:</h3>
# <br>
# For model evaluations, we are going to use <strong>Tree Based Model</strong>:<br>
#     <strong>Random Forests (RF)</strong>, this tree-based algorithm are generally robust to outliers. They make splits based on feature values, and outliers often end up in leaf nodes, having minimal impact on the overall decision-making process.<br>
# </div>
# 

# <a id="7.6"></a>
# # <b><span style='color:#333'>7.6 |</span><span style='color:#743089;font-weight:bold'>  Encode Categorical Variables</span></b>

# In[ ]:


# List of features to exclude
exclude_features = ['spore-print-color', 'stem-root', 'stem-surface', 'veil-color']

# Create a new list of categorical features excluding the ones in exclude_features
cat_features = [feature for feature in cat_features if feature not in exclude_features]
cat_features


# <div style="background-color: #F2EBE6; padding: 15px; border-radius: 10px; border: 2px solid #8B4513;"><p>Since all categorical features are nominal, we will use <strong>One-Hot Encoding</strong>.
# </p></div>

# In[ ]:


train_df_enc = pd.get_dummies(train_df.drop(columns="id",axis=1), columns=cat_features, drop_first=True,dtype=int)
test_df_enc = pd.get_dummies(test_df.drop(columns="id",axis=1), columns=cat_features, drop_first=True,dtype=int)


# In[ ]:




# In[ ]:


# Encoding Target feature
## Replace class labels
train_df_enc['class'] = train_df_enc['class'].replace({'p': 1, 'e': 0})


# In[ ]:


train_df_enc.head()


# <a id="7.7"></a>
# # <b><span style='color:#333'>7.7 |</span><span style='color:#743089;font-weight:bold'> Checking Imbalanced Data</span></b>

# In[ ]:


percentage = train_df_enc['class'].value_counts(normalize=True) * 100
percentage = percentage.sort_index()

# Plotting the percentage of each class
plt.figure(figsize=(8, 5))
ax = sns.barplot(x=percentage.index, y=percentage)
plt.title('Percentage')
plt.xlabel('Class')
plt.ylabel('Percentage (%)')
plt.xticks(ticks=[1, 0], labels=['Poisonous','Edible'])
plt.yticks(ticks=range(0,80,10))

# Displaying the percentage on the bars
for i, p in enumerate(percentage):
    ax.text(i, p + 0.5, f'{p:.2f}%', ha='center', va='bottom')

plt.show()


# <div style="background-color: #F2EBE6; padding: 15px; border-radius: 10px; border: 2px solid #8B4513;">
#     <p>The bar plot shows the percentage of edible and poisonous mushrooms in the dataset. Approximately <strong>45.29%</strong> of the mushrooms are '<strong>edible</strong>', while <strong>54.71%</strong> of the mushrooms are '<strong>poisonous</strong>'.This indicates that there is some imbalance in the target variable, but it is not highly imbalanced. Generally, a dataset is considered highly imbalanced if one class represents over 80-90% of the data..</p>
# </div>
# 

# <a id="8"></a>
# # <p style="background-color:#8B4513; font-family:calibri; font-size:130%; color:#F2EBE6; text-align:center; border-radius:15px 50px; padding:10px">8 | Train Test Split</p>
# 
# ⬆️ [Table of Contents](#contents_tabel)

# In[ ]:


X = train_df_enc.drop(columns='class', axis=1)
y = train_df_enc['class'] 


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)


# In[ ]:




# <a id="9"></a>
# # <p style="background-color:#8B4513; font-family:calibri; font-size:130%; color:#F2EBE6; text-align:center; border-radius:15px 50px; padding:10px">9 | Correlation Analysis </p>
# 
# ⬆️ [Table of Contents](#contents_tabel)

# In[ ]:


# Define a customized colormap
num_colors = 256
colors = ['darkorange', '#FFEFD5', 'white', '#F5F5F5', 'purple']
my_cmap = LinearSegmentedColormap.from_list('my_colormap', colors, num_colors)

# Calculation of the Spearman correlation
target = 'class'
train_ordered = pd.concat([train_df_enc.drop(target, axis=1), train_df_enc[target]], axis=1)
corr = train_ordered.corr()

# Perform hierarchical clustering on the correlation matrix
linkage_matrix = linkage(corr, method='average')

# Create a clustered heatmap
plt.figure(figsize=(15, 10), dpi=80)
sns.clustermap(corr, method='average', cmap=my_cmap, linewidths=0.2, figsize=(15, 10), dendrogram_ratio=0.2)

plt.show()


# <a id="11"></a>
# # <p style="background-color:#8B4513; font-family:calibri; font-size:130%; color:#F2EBE6; text-align:center; border-radius:15px 50px; padding:10px">11 | Random Forest Model Building </p>
# 
# ⬆️ [Table of Contents](#contents_tabel)

# In[ ]:


# Initialize RandomForestClassifier
rf = RandomForestClassifier(criterion="gini", random_state=0)

# Fit the model
rf.fit(X_train, y_train)

# Predict on test data
y_pred = rf.predict(X_test)

# Calculate Matthews correlation coefficient
MCC_score = matthews_corrcoef(y_test, y_pred)


# <a id="12"></a>
# # <p style="background-color:#8B4513; font-family:calibri; font-size:130%; color:#F2EBE6; text-align:center; border-radius:15px 50px; padding:10px">12 | Competition Submission </p>
# 
# ⬆️ [Table of Contents](#contents_tabel)

# In[ ]:


# Define features(X) and output labels(y)
X = train_df_enc.drop(columns='class', axis=1)
y = train_df_enc['class'] 

# Initialize RandomForestClassifier
rf = RandomForestClassifier(criterion="gini", random_state=0)

# Train the model on the whole dataset
rf.fit(X, y)

# Target prediction for test.csv samples using relavant features
y_pred = rf.predict(test_df_enc)


# In[ ]:


# Map the numerical predictions back to the original labels
inverse_class_mapping = {1: 'p', 0: 'e'}
y_pred_labels = [inverse_class_mapping[pred] for pred in y_pred]

# Create the submission DataFrame
df_submission = pd.DataFrame({
    'id': test_df['id'],
    'class': y_pred_labels
})

df_submission.head()


# In[ ]:


df_submission.to_csv('submission.csv', index = False)


# # Thankyou!
