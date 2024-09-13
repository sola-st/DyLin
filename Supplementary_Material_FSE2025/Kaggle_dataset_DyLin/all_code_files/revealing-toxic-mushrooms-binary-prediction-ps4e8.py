#!/usr/bin/env python
# coding: utf-8

# # <p style="background-color: #ad3931; color: #fcf8f7; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #A57CDB; border: 5px dotted #fcf8f7;">🍄🔮 Mushrooms Danger Unveiled: Binary Prediction of Poisonous Species | PS4E8 🍄🔍</p>
# 

# ![_2705f2c2-59b4-44fd-8f29-686daa114023 (1).jpg](attachment:9ef6a769-596e-4e22-a10d-934cc46748a5.jpg)

# # <p style="background-color: #ad3931; color: #fcf8f7; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #A57CDB; border: 5px dotted #fcf8f7;">About Dataset</p>
# 

# * This is the Binary Prediction of Poisonous Mushrooms Competition Dataset.
# * It consists of three Files such as Train, test and sample submission
# * The Training data consists of 3116945 rows and 21 columns* This is the Binary Prediction of Poisonous Mushrooms Competition Dataset.
# * It consists of three Files such as Train, test and sample submission
# * The Training data consists of 3116945 rows and 21 columns
# * The columns of Training Data are class, cap-diameter, cap-shape, cap-surface, cap-color, does-bruise-or-bleed, gill-attachment, gill-spacing, gill-color, stem-height, stem-width, stem-root, stem-surface, stem-color, veil-type, veil-color, has-ring, ring-type, spore-print-color, habitat, season.
# * The test Data consists of 2077964 rows and 20 columns.
# * The test Data includes these columns such as cap-diameter, cap-shape, cap-surface, cap-color, does-bruise-or-bleed, gill-attachment, gill-spacing, gill-color, stem-height, stem-width, stem-root, stem-surface, stem-color, veil-type, veil-color, has-ring, ring-type, spore-print-color, habitat, season.
# * while the sample submission consists of 2077964 rows and 2 columns.
# * The columns of sample submission includes id and class.
# * The dataset for this competition (both train and test) was generated from a deep learning model trained on the UCI Mushroom dataset.
# * I also took Secondary Mushroom Dataset for little improvement in mcc
# * The Dataset Secondary Mushroom of simulated mushrooms is used for binary classification into edible and poisonous.
# * This dataset includes 61069 hypothetical mushrooms with caps based on 173 species (353 mushrooms per species). 
# * Each mushroom is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended (the latter class was combined with the poisonous class).

# # **Files:**

# 
# * train.csv - the training dataset; class is the binary target (either e or p)
# * test.csv - the test dataset; your objective is to predict target class for each row
# * sample_submission.csv - a sample submission file in the correct format

# # <p style="background-color: #ad3931; color: #fcf8f7; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #A57CDB; border: 5px dotted #fcf8f7;">Aims and Objectives</p>
# 

# * This is the Binary Prediction of Poisonous Mushrooms Competition Data.
# * The aim to take this competition is to predict whether a mushroom is edible or poisonous based on its physical characteristics.
# * For this I firstly concatenate the Train Data of competition with secondary-mushroom-dataset-data-set in order to take the detailed overview about the data and then I create various visualization plots in the form of subplots such as countplot, pairplot, Histogram, Facetgrid Plot, ViolinPlot and piechart to take deep insights about Data.
# * Then I visualize the outliers through Boxplot and apply the LogNormalization Technique to take accuracte and best results.
# * Then I take a look at the Duplicates of the Data. There is no duplicates present in competition data but when i combine the train and original data i get 146.
# * Then I Drop the Duplicates.
# * Then I apply the KFold Cross-Validation and choose the Best Model.
# * Then I apply Hyperparameter Tuning on the best model to take more better results
# * Then I create the submission File

# # <p style="background-color: #ad3931; color: #fcf8f7; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #A57CDB; border: 5px dotted #fcf8f7;">Training Data Columns</p>
# 

# | Column Name            | Description                                                                 |
# |------------------------|-----------------------------------------------------------------------------|
# | class                  | The class of the mushroom (e.g., edible or poisonous).                       |
# | cap-diameter           | The diameter of the mushroom cap (e.g., small, medium, large).               |
# | cap-shape              | The shape of the mushroom cap (e.g., bell, conical, flat).                   |
# | cap-surface            | The surface texture of the mushroom cap (e.g., smooth, scaly).               |
# | cap-color              | The color of the mushroom cap (e.g., brown, yellow, red).                    |
# | does-bruise-or-bleed   | Indicates if the mushroom bruises or bleeds when cut or damaged (e.g., yes, no). |
# | gill-attachment        | How the gills are attached to the stem (e.g., free, attached).               |
# | gill-spacing           | The spacing between the gills (e.g., close, distant).                        |
# | gill-color             | The color of the gills (e.g., white, black, pink).                           |
# | stem-height            | The height of the mushroom stem (e.g., short, medium, tall).                 |
# | stem-width             | The width of the mushroom stem (e.g., thin, medium, thick).                  |
# | stem-root              | The appearance of the stem's base (e.g., bulbous, tapering).                 |
# | stem-surface           | The texture of the mushroom stem (e.g., smooth, rough).                      |
# | stem-color             | The color of the mushroom stem (e.g., white, brown, yellow).                 |
# | veil-type              | The type of veil present (e.g., partial, universal).                         |
# | veil-color             | The color of the veil (e.g., white, yellow, brown).                          |
# | has-ring               | Indicates if the mushroom has a ring on the stem (e.g., yes, no).            |
# | ring-type              | The type of ring present (e.g., single, double).                             |
# | spore-print-color      | The color of the spore print (e.g., black, white, brown).                    |
# | habitat                | The habitat where the mushroom was found (e.g., forest, grassland).          |
# | season                 | The season when the mushroom was found (e.g., spring, summer, autumn, winter). |
# 

# # <p style="background-color: #ad3931; color: #fcf8f7; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #A57CDB; border: 5px dotted #fcf8f7;">About Author</p>
# 

# * Hi Kagglers! I'm Maria Nadeem, a passionate Data Scientist with keen interest in exploring and applying diverse data science techniques.
# * As dedicated to derive meaningful insights and making impactful decisions through data, I actively engage in projects and contribute to Kaggle by sharing detailed analysis and actionable insights.
# * I'm excited to share my latest project on Binary Prediction of Poisonous Mushrooms.
# * In this notebook, I begin by conatenating train data of competition with secondary-mushroom-dataset-data-set, providing detailed overview of the dataset and then dive into creating various visualization plots using subplots to gain deep insights. Following this, I use the KFold Cross Validation, perform model training, and choose the best model to take improve predictions.
# * Then I took the Best Model and then apply Hyperparameter Tuning on it and took more better results.
# 

# | Name               | Email                                               | LinkedIn                                                  | GitHub                                           | Kaggle                                        |
# |--------------------|-----------------------------------------------------|-----------------------------------------------------------|--------------------------------------------------|-----------------------------------------------|
# | **Maria Nadeem**  | marianadeem755@gmail.com | <a href="https://www.linkedin.com/in/maria-nadeem-4994122aa/" style="text-decoration: none; font-size: 16px;"><img src="https://img.shields.io/badge/LinkedIn-%2300A4CC.svg?style=for-the-badge&logo=LinkedIn&logoColor=white" alt="LinkedIn Badge"></a> | <a href="https://github.com/marianadeem755" style="text-decoration: none; font-size: 16px;"><img src="https://img.shields.io/badge/GitHub-%23FF6F61.svg?style=for-the-badge&logo=GitHub&logoColor=white" alt="GitHub Badge"></a> | <a href="https://www.kaggle.com/marianadeem755" style="text-decoration: none; font-size: 16px;"><img src="https://img.shields.io/badge/Kaggle-%238a2be2.svg?style=for-the-badge&logo=Kaggle&logoColor=white" alt="Kaggle Badge"></a> |
# 

# # <p style="background-color: #ad3931; color: #fcf8f7; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #A57CDB; border: 5px dotted #fcf8f7;">Import Libraries</p>
# 

# In[ ]:


#get_ipython().system('pip install xgboost lightgbm catboost optuna ')


# In[ ]:


# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
from IPython.display import display, HTML
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import matthews_corrcoef
from sklearn.base import clone
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import optuna
from optuna.samplers import TPESampler
palette = ['#D96D6A', '#A57CDB', '#4AAB4F', '#F9A1A1', '#72C1F1']
color_palette = sns.color_palette(palette)
#ignore warnings
import warnings
warnings.filterwarnings('ignore')


# # <p style="background-color: #ad3931; color: #fcf8f7; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #A57CDB; border: 5px dotted #fcf8f7;">Datasets Overview</p>
# 

# In[ ]:


# Function to style tables
def style_table(df):
    styled_df = df.style.set_table_styles([
        {"selector": "th", "props": [("color", "white"), ("background-color", "#963c7d")]}  # Updated header color
    ]).set_properties(**{"text-align": "center"}).hide(axis="index")
    return styled_df.to_html()

# Function to generate random shades of color
def generate_random_color():
    color = "#{:02x}{:02x}{:02x}".format(
        random.randint(150, 255),
        random.randint(150, 255)
    )
    return color

# Function to create styled heading with emojis and different colors for main and sub-headings
def styled_heading(text, background_color, text_color='white', border_color=None, font_size='30px', border_style='dashed'):
    border_color = border_color if border_color else background_color
    return f"""
    <div style="
        text-align: center;
        background: {background_color};
        color: {text_color};
        padding: 15px;
        font-size: {font_size};
        font-weight: bold;
        line-height: 1;
        border-radius: 20px 20px 0 0;
        margin-bottom: 20px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
        border: 3px {border_style} {border_color};
    ">
        {text}
    </div>
    """

# Define your color palette
palette = ['#D96D6A', '#A57CDB', '#4AAB4F', '#F9A1A1', '#72C1F1']
color_palette = sns.color_palette(palette)

# Define colors for headings and sub-headings
main_heading_color = '#ad3931'  # Main heading color
sub_heading_color = '#187a24'    # Sub-heading color
headings_border_color = '#963c7d' # Border color for headings

def print_dataset_analysis(dataset, dataset_name, n_top=5, palette_index=0):
    heading_color = color_palette[palette_index]
    
    # Main heading with emoji
    heading = styled_heading(f"📊 {dataset_name} Overview", main_heading_color, 'white', border_color=headings_border_color, font_size='35px', border_style='solid')
    display(HTML(heading))
    
    # Sub-headings with emojis
    display(HTML(f"<h2 style='font-size: 24px; color: {sub_heading_color};'>🔍 Shape of the Dataset</h2>"))
    display(HTML(f"<p>{dataset.shape[0]} rows and {dataset.shape[1]} columns</p>"))
    
    display(HTML(f"<h2 style='font-size: 24px; color: {sub_heading_color};'>👀 First 5 Rows</h2>"))
    display(HTML(style_table(dataset.head(n_top))))
    
    display(HTML(f"<h2 style='font-size: 24px; color: {sub_heading_color};'>📈 Summary Statistics</h2>"))
    display(HTML(style_table(dataset.describe())))
    
    display(HTML(f"<h2 style='font-size: 24px; color: {sub_heading_color};'>🚨 Null Values</h2>"))
    null_counts = dataset.isnull().sum()
    if null_counts.sum() == 0:
        display(HTML("<p>No null values found.</p>"))
    else:
        display(HTML(style_table(null_counts[null_counts > 0].to_frame(name='Null Values'))))
    
    display(HTML(f"<h2 style='font-size: 24px; color: {sub_heading_color};'>🔍 Duplicate Rows</h2>"))
    duplicate_count = dataset.duplicated().sum()
    display(HTML(f"<p>{duplicate_count} duplicate rows found.</p>"))
    
    display(HTML(f"<h2 style='font-size: 24px; color: {sub_heading_color};'>📝 Data Types</h2>"))
    dtypes_table = pd.DataFrame({
        'Data Type': [dataset[col].dtype for col in dataset.columns],
        'Column Name': dataset.columns
    })
    display(HTML(style_table(dtypes_table)))

    display(HTML(f"<h2 style='font-size: 24px; color: {sub_heading_color};'>📋 Column Names</h2>"))
    display(HTML(f"<p>{', '.join(dataset.columns)}</p>"))

    display(HTML(f"<h2 style='font-size: 24px; color: {sub_heading_color};'>🔢 Unique Values</h2>"))
    unique_values_table = pd.DataFrame({
        'Data Type': [dataset[col].dtype for col in dataset.columns],
        'Column Name': dataset.columns,
        'Unique Values': [', '.join(map(str, dataset[col].unique()[:7])) for col in dataset.columns]
    })
    display(HTML(style_table(unique_values_table)))

# Example usage with your dataset (`df_train`, `df_test`, `df_sub`)
# Load datasets
train = pd.read_csv("/kaggle/input/playground-series-s4e8/train.csv", index_col="id")
original = pd.read_csv("/kaggle/input/secondary-mushroom-dataset-data-set/MushroomDataset/secondary_data.csv", sep=";")
test = pd.read_csv("/kaggle/input/playground-series-s4e8/test.csv", index_col="id")
sample_sub = pd.read_csv("/kaggle/input/playground-series-s4e8/sample_submission.csv")

# Use different palette colors for different datasets


# # <p style="background-color: #ad3931; color: #fcf8f7; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #A57CDB; border: 5px dotted #fcf8f7;">Lets Display the countplot for Categorical columns</p>
# 

# ## **About Code:**
# * This code is used to Display the CountPlot of th the categorical columns which show the frequency of each category within the categorical features. Here's a breakdown of the code
# * It visualize the distribution of categorical columns in a dataset, particularly focusing on how the distribution varies based on a target variable which is class. 
# * Then a list of categorical columns is created. so, to create the subplot of these selected columns.
# * Then a grid of subplots is created with 4 rows and 4 columns resulting in 12 subplots are created
# * Then Flattens the 2D array of axes into a 1D array using `axes = axes.flatten()` in order to make it easier to iterate over.
# * Then loop through the columns using enumerate(categorical_features).
# * Then selects the top 10 most frequent categories within the current column. so, that the correct and deep insights about the data can be taken
# * Then  Filters the Data to include only the rows where the column value is one of these top categories to visualize.
# * Then generates the count plot and filtered the data specifies to the plot.
# * Then set the hue=class which is a target variable.
# * Then customize the plot by setting the title and rotates the x-axis labels by 45 degrees to make them more readable.
# * Then adjust the subplots parameter and Displays the plot

# In[ ]:


# List of categorical features
categorical_features = ['cap-shape', 'cap-surface', 'cap-color', 'gill-attachment', 
                         'gill-spacing', 'stem-root', 'stem-surface', 
                         'stem-color', 'has-ring', 
                         'ring-type', 'spore-print-color', 'habitat', 'season']

# Create subplots
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 15))
axes = axes.flatten()

# Plot count plots for each categorical feature
for i, feature in enumerate(categorical_features):
    # Limit the number of categories to the top N most frequent
    top_categories = train[feature].value_counts().index[:10]  # Adjust N as needed
    filtered_data = train[train[feature].isin(top_categories)]
    
    sns.countplot(data=filtered_data, x=feature, hue='class', ax=axes[i], palette=color_palette)
    axes[i].set_title(f'Count of {feature}')
    axes[i].tick_params(axis='x', rotation=45)

# Remove empty subplots if there are fewer features than subplot slots
for j in range(len(categorical_features), len(axes)):
    fig.delaxes(axes[j])

# Adjust layout
plt.tight_layout()
plt.show()


# # <p style="background-color: #ad3931; color: #fcf8f7; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #A57CDB; border: 5px dotted #fcf8f7;">Lets Create FacetGrid Plots for Categorical Columns by Class</p>
# 

# ## **About Code:**
# * This code is used to visualize the distribution of categorical features in a dataset using a FacetGrid for each categorical column.
# * The FacetGrid allow to create a grid of plots, where each plot shows the distribution of a column separated by a specified variable which is being defined.
# * Then a list of categorical columns is created. so, to create the Facetgrid of these selected columns.
# * Then the loop iterates over each categorical column
# * Then selects the top 10 most frequent categories of the selected columns.
# * Then filters the Data to keep only the rows where the columns value is one of the top categories. This filtered dataset of the selected columns is then used for plotting.
# * Then creates a grid of plots using `g = sns.FacetGrid(filtered_data, col='class', col_wrap=4, height=4, sharey=False)`.
# * Then g.map(sns.countplot, feature, palette=color_palette) applies the countplot function to each facet in the grid which is already created.
# * Then set the title, xaxis and yaxis labels to craete the Facetgrid.
# * Then the loop `for ax in g.axes.flat:` iterates through all subplots in the grid.
# * Then adjust the layout and Displays the subplots.

# In[ ]:


# List of categorical features
categorical_features = ['cap-shape', 'cap-surface', 'cap-color', 'gill-attachment', 
                         'gill-spacing','stem-root', 'stem-surface', 
                         'stem-color', 'has-ring', 
                         'ring-type', 'spore-print-color', 'habitat', 'season']

# Create a FacetGrid for each categorical feature
for feature in categorical_features:
    # Limit the number of categories to the top N most frequent
    top_categories = train[feature].value_counts().index[:10]  # Adjust N as needed
    filtered_data = train[train[feature].isin(top_categories)]
    
    # Create the FacetGrid
    g = sns.FacetGrid(filtered_data, col='class', col_wrap=4, height=4, sharey=False)
    g.map(sns.countplot, feature, palette=color_palette)
    g.set_titles(f'{feature} Distribution by Class')
    g.set_axis_labels(feature, 'Count')
    
    # Rotate x-tick labels for readability
    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()


# # <p style="background-color: #ad3931; color: #fcf8f7; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #A57CDB; border: 5px dotted #fcf8f7;">Lets have a Sneakpeak at Histogram</p>
# 

# ## **About Code:**
# * This code is used to visualize the distribution of Numeric Columns in a dataset using histograms.
# * By applying this code  Each numeric feature is plotted in a separate subplot, and the data is grouped by a categorical variable (class) inorder to look that how the data distribution varies between classes.
# * Then a list of Numerical columns is created. so, to visualize the Histogram of the numerical columns.
# * Then the grid of subplots is created using `plt.subplots`.
# * Then specifies the number of Rows (which is equal to1 Here) and specifies the Number of Columns which is equal to number of Numerical columns.
# * Then set the size of figure and bin_edges dictionary where keys are numerical column names, and values are lists of bin edges for those Columns. Bin edges determine the intervals used in the Histogram.
# * Then the code iterates over the numeric columns, and for each Numerical Colunm, it creates a Histogram.
# * Then the data split by the class variable, so that different classes will have different colors in the histogram.
# * And Different classes in Histogram are stacked on top of each other, allowing to compare the distribution across classes.
# * Then specifies the Binedge for the Histogram.
# * Then set the Title, xaxis and yaxis Labels and Positions the legend outside the plot to the upper left.
# * Then adjust the layou and show the subplots.

# In[ ]:


# List of numeric features
numeric_features = ['cap-diameter', 'stem-height', 'stem-width']

# Create subplots
fig, axes = plt.subplots(nrows=1, ncols=len(numeric_features), figsize=(18, 6))

# Define bin edges or range of interest for each feature if needed
bin_edges = {
    'cap-diameter': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    'stem-height': [0, 10, 20, 30, 40, 50, 60, 70, 80],
    'stem-width': [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]
}

for i, feature in enumerate(numeric_features):
    sns.histplot(
        data=train,
        x=feature,
        hue='class',
        multiple='stack',
        ax=axes[i],
        palette=color_palette,
        bins=bin_edges.get(feature, 20)  # Use bin edges defined above or default to 20 bins
    )
    axes[i].set_title(f'Distribution of {feature}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Count')
    axes[i].legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')  # Adjust legend position

# Adjust layout
plt.tight_layout()
plt.show()


# # <p style="background-color: #ad3931; color: #fcf8f7; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #A57CDB; border: 5px dotted #fcf8f7;">Lets have a look at Pairplot</p>
# 

# ## **About Code:**
# 
# * The code is used to genrate a grid of scatter plots and KDE plots that display pairwise relationships between the numeric features such as 'cap-diameter', 'stem-height', and 'stem-width' and to visualize the relationships between multiple Numeric variables in a dataset, with data points colored by the Target variable `class`.
# * Then a list of Numeric columns is created. These are the columns in the Dataset that contain the Numeric data.
# * Then  creates a grid of plots using `sns.pairplot`.
# * Then set `hue=class` , it sets the color to thw class variable.
# * Then Specifies the type of plot to create on  diagonal of the grid. In this case, a kernel density estimate (KDE) plot created on diagonal, which shows the distribution of each individual.
# * Then set the title and positions the title slightly above than default position.
# * Then adjust the layout and Displays the plot

# In[ ]:


# List of numeric features
numeric_features = ['cap-diameter', 'stem-height', 'stem-width']

# Create a pairwise plot with hue for numeric features
g = sns.pairplot(train, vars=numeric_features, hue='class', palette=color_palette, diag_kind='kde')

# Adjust the title and layout
g.fig.suptitle('Pairwise Plot of Numeric Features by Class', y=1.02)
g.fig.tight_layout()
g.fig.subplots_adjust(top=0.9)  # Adjust the top space to make room for the title

plt.show()


# # <p style="background-color: #ad3931; color: #fcf8f7; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #A57CDB; border: 5px dotted #fcf8f7;">Lets Visualize the Distribution of Categorical Data</p>
# 

# ## **About Code:**
# * This code is used to define a function which is `plot_column_distribution` that generates two types of plots First it generates a count plot and 2nd it generates a pie chart to visualize the distribution of a specific categorical column in the Data.
# * This Function is used to display the plots of the column which is taken or defined and it displays the number of top N most frequent categories.
# * Then it filters the original DataFrame to include only the rows where the columns value is one of these top N categories.
# * Then computes the frequency of each category in the filtered DataFrame. 
# * Then creates a figure of subplots with 1 row and 2 columns
# * The count plot is then created showing the frequency of each category in the filtered DataFrame using `sns.countplot`
# * Then adjust the layout and displays the plots

# In[ ]:


def plot_column_distribution(df, column_name, top_n=20, color_palette=color_palette):
    """
    Plots the distribution of a categorical column in the given DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column_name (str): The name of the column to plot.
    top_n (int): Limit the number of categories to the top N most frequent.
    color_palette (list): List of colors for the plots.
    """
    
    # Get the top N most frequent categories
    top_categories = df[column_name].value_counts().nlargest(top_n).index
    df_filtered = df[df[column_name].isin(top_categories)]
    
    # Get value counts for pie chart
    value_counts = df_filtered[column_name].value_counts()
    
    # Create the figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Create a count plot
    sns.countplot(data=df_filtered, x=column_name, ax=axes[0], palette=color_palette)
    axes[0].set_title(f'Count of {column_name}')
    axes[0].set_xlabel(column_name)
    axes[0].set_ylabel('Count')
    
    # Create a pie plot
    axes[1].pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90, colors=color_palette[:len(value_counts)])
    axes[1].set_title(f'Distribution of {column_name}')
    
    plt.tight_layout()
    plt.show()
# Plot distribution for a specific column
plot_column_distribution(train, 'class')


# # <p style="background-color: #ad3931; color: #fcf8f7; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #A57CDB; border: 5px dotted #fcf8f7;">Lets have a look at ViolinPlot</p>
# 

# ## **About Code:**
# * This code creates violin plots for the Numeric Variables, which shows the distribution of these variables across different classes. 
# * numeric_features is a list containing the names of the Numeric columns to visualize. It includes cap-diameter, stem-height, and stem-width.
# * Then the loop iterates through each Numericc variable
# * The violin plot, is generated using `sns.violinplo`t which is used to visualize the distribution of the data and its probability density.
# * Set the Target variable `class` on xaxis and categorical variables on yaxis.
# * Then set the Title, xaxis and yaxis label and Displays the plots.

# In[ ]:


# Create violin plots for numeric features
# List of numeric features
numeric_features = ['cap-diameter', 'stem-height', 'stem-width']
for feature in numeric_features:
    plt.figure(figsize=(8, 5))
    sns.violinplot(data=train, x='class', y=feature, palette=color_palette)
    plt.title(f'Violin Plot of {feature} by Class')
    plt.xlabel('Class')
    plt.ylabel(feature)
    plt.show()


# # <p style="background-color: #ad3931; color: #fcf8f7; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #A57CDB; border: 5px dotted #fcf8f7;">Lets have SneakPeak at HeatMap of Categorical Columns</p>
# 

# In[ ]:


# List of categorical features
categorical_features = ['cap-shape', 'cap-surface', 'cap-color', 'gill-attachment', 
                         'gill-spacing', 'gill-color', 'stem-root', 'stem-surface', 
                         'stem-color', 'veil-type', 'veil-color', 'has-ring', 
                         'ring-type', 'spore-print-color', 'habitat', 'season']
# Create a custom colormap from the color palette
cmap = LinearSegmentedColormap.from_list("custom_cmap", color_palette, N=256)

# Convert categorical features to numeric format
df_cat_encoded = train[categorical_features].apply(lambda x: pd.factorize(x)[0])

# Compute correlation matrix
corr = df_cat_encoded.corr()

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap=cmap, fmt='.2f')
plt.title('Correlation Heatmap of Categorical Features')
plt.show()


# # <p style="background-color: #ad3931; color: #fcf8f7; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #A57CDB; border: 5px dotted #fcf8f7;">Lets have SneakPeak at HeatMap of Numeric Columns</p>
# 

# In[ ]:


numeric_features = ['cap-diameter', 'stem-height', 'stem-width']
# Create a custom colormap from the color palette
cmap = LinearSegmentedColormap.from_list("custom_cmap", color_palette, N=256)

# Convert categorical features to numeric format
df_cat_encoded = train[numeric_features].apply(lambda x: pd.factorize(x)[0])

# Compute correlation matrix
corr = df_cat_encoded.corr()

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap=cmap, fmt='.2f')
plt.title('Correlation Heatmap of Categorical Features')
plt.show()


# In[ ]:


# Combine training datasets
train = pd.concat([train, original], ignore_index=True)


# # <p style="background-color: #ad3931; color: #fcf8f7; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #A57CDB; border: 5px dotted #fcf8f7;">Lets Have a look at Duplicates After combining Train Data and Original Dataset</p>
# 

# In[ ]:


Duplicates=train.duplicated().sum()


# # <p style="background-color: #ad3931; color: #fcf8f7; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #A57CDB; border: 5px dotted #fcf8f7;">Remove Duplicates</p>
# 

# In[ ]:


# Remove duplicates
cols = train.columns.tolist()
cols.remove("class")
train = train.drop_duplicates(subset=cols, keep='first')



# In[ ]:


# Identify columns with more than 70% missing values
missing_threshold = 0.7
missing_train = train.isnull().mean()
columns_to_drop = missing_train[missing_train > missing_threshold].index.tolist()

# Drop columns (except 'class')
columns_to_drop = [col for col in columns_to_drop if col != 'class']
train.drop(columns=columns_to_drop, inplace=True)
test.drop(columns=columns_to_drop, inplace=True)



# # <p style="background-color: #ad3931; color: #fcf8f7; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #A57CDB; border: 5px dotted #fcf8f7;">Lets Have a look at Missing values</p>
# 

# In[ ]:


train.isnull().sum()


# # <p style="background-color: #ad3931; color: #fcf8f7; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #A57CDB; border: 5px dotted #fcf8f7;">Lets Impute the missing values of Numeric Columns</p>
# 

# In[ ]:


# Identify numerical columns
numerical_cols = train.select_dtypes(include=[np.number]).columns.tolist()

# Fill missing numerical values with median
for col in numerical_cols:
    median = train[col].median()
    train[col].fillna(median, inplace=True)
    test[col].fillna(median, inplace=True)


# # <p style="background-color: #ad3931; color: #fcf8f7; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #A57CDB; border: 5px dotted #fcf8f7;">Lets Impute the missing values of Categorical Columns</p>
# 

# In[ ]:


# Identify categorical columns
categorical_cols = train.select_dtypes(include=[object]).columns.tolist()

# Ensure 'class' column is not removed or processed
for col in categorical_cols:
    if col != 'class':
        # Fill missing values with 'Missing'
        train[col].fillna('not_presesnt', inplace=True)
        test[col].fillna('not_presesnt', inplace=True)
        
        # Combine rare categories (frequency less than 1%)
        freq = train[col].value_counts(normalize=True)
        rare_categories = freq[freq < 0.01].index
        train[col] = train[col].replace(rare_categories, 'infrequent')
        test[col] = test[col].replace(rare_categories, 'infrequent')


# # <p style="background-color: #ad3931; color: #fcf8f7; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #A57CDB; border: 5px dotted #fcf8f7;">Lets Encode Categorical Columns</p>
# 

# In[ ]:


# Exclude 'class' from categorical columns
categorical_cols = [col for col in categorical_cols if col != 'class']

# Combine train and test data only for the categorical columns
combined = pd.concat([train[categorical_cols], test[categorical_cols]], axis=0)

# Ordinal Encoding
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
combined_encoded = encoder.fit_transform(combined)

# Split back to train and test
train_encoded = combined_encoded[:len(train)]
test_encoded = combined_encoded[len(train):]

# Replace original categorical columns with encoded ones
train[categorical_cols] = train_encoded
test[categorical_cols] = test_encoded


# # <p style="background-color: #ad3931; color: #fcf8f7; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #A57CDB; border: 5px dotted #fcf8f7;">Lets Select Features and Target Variable</p>
# 

# In[ ]:


# Features and target
X = train.drop('class', axis=1)
y = train['class']

# Label encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Check shapes


# # <p style="background-color: #ad3931; color: #fcf8f7; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #A57CDB; border: 5px dotted #fcf8f7;">Cross-Validation with Evaluation and Predictions</p>
# 

# In[ ]:


# Define Stratified K-Fold
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

def k_fold_model_trainer(model, X, y, skf, test_data):
    """
    Trains the model using Stratified K-Fold cross-validation and evaluates using MCC.
    Returns the average validation MCC and test predictions.
    """
    train_mcc_scores = []
    val_mcc_scores = []
    test_preds = np.zeros(len(test_data))
    
    fold_number = 1
    for train_index, val_index in skf.split(X, y):
        print(f'Fold {fold_number}')
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # Clone the model to ensure fresh parameters
        model_clone = clone(model)
        
        # Fit the model
        model_clone.fit(X_train, y_train)
        
        # Predict on training and validation sets
        y_train_pred = model_clone.predict(X_train)
        y_val_pred = model_clone.predict(X_val)
        
        # Calculate MCC scores
        train_mcc = matthews_corrcoef(y_train, y_train_pred)
        val_mcc = matthews_corrcoef(y_val, y_val_pred)
        
        print(f'Training MCC: {train_mcc:.4f}')
        print(f'Validation MCC: {val_mcc:.4f}\n')
        
        train_mcc_scores.append(train_mcc)
        val_mcc_scores.append(val_mcc)
        
        # Predict on test data
        test_preds += model_clone.predict_proba(test_data)[:, 1]
        
        fold_number += 1
    
    # Average test predictions
    test_preds /= n_splits
    
    # Average MCC scores
    avg_train_mcc = np.mean(train_mcc_scores)
    avg_val_mcc = np.mean(val_mcc_scores)
    
    print(f'Average Training MCC: {avg_train_mcc:.4f}')
    print(f'Average Validation MCC: {avg_val_mcc:.4f}')
    
    return avg_val_mcc, test_preds


# In[ ]:


# Initialize models
lgbm_model = LGBMClassifier(random_state=42)
catboost_model = CatBoostClassifier(verbose=0, random_state=42)
xgboost_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)


# # <p style="background-color: #ad3931; color: #fcf8f7; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #A57CDB; border: 5px dotted #fcf8f7;">Models Training</p>
# 

# In[ ]:


# Dictionary to store model performances
model_performance = {}

# LightGBM
lgbm_val_mcc, lgbm_test_preds = k_fold_model_trainer(lgbm_model, X, y, skf, test)
model_performance['LightGBM'] = {'val_mcc': lgbm_val_mcc, 'test_preds': lgbm_test_preds}

# CatBoost
catboost_val_mcc, catboost_test_preds = k_fold_model_trainer(catboost_model, X, y, skf, test)
model_performance['CatBoost'] = {'val_mcc': catboost_val_mcc, 'test_preds': catboost_test_preds}

# XGBoost
xgboost_val_mcc, xgboost_test_preds = k_fold_model_trainer(xgboost_model, X, y, skf, test)
model_performance['XGBoost'] = {'val_mcc': xgboost_val_mcc, 'test_preds': xgboost_test_preds}


# In[ ]:


# Find the best model based on validation MCC score
best_model_name = max(model_performance, key=lambda k: model_performance[k]['val_mcc'])
best_model_performance = model_performance[best_model_name]



# # <p style="background-color: #ad3931; color: #fcf8f7; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #A57CDB; border: 5px dotted #fcf8f7;">Optimizing the CatBoost with Optuna</p>
# 

# In[ ]:


import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from optuna.samplers import TPESampler

def objective(trial):
    # Define the hyperparameters to tune for CatBoost
    grow_policy = trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide'])
    
    params = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
        'depth': trial.suggest_int('depth', 3, 15),
        'iterations': trial.suggest_int('iterations', 50, 300),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-3, 1e-1),
        'bagging_temperature': trial.suggest_uniform('bagging_temperature', 0, 1),
        'random_strength': trial.suggest_loguniform('random_strength', 1e-3, 1e-1),
        'max_bin': trial.suggest_int('max_bin', 128, 512),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
        'grow_policy': grow_policy
    }

    # Add max_leaves only if grow_policy is 'Lossguide'
    if grow_policy == 'Lossguide':
        params['max_leaves'] = trial.suggest_int('max_leaves', 10, 100)
    
    # Initialize the CatBoost model with the suggested hyperparameters
    model = CatBoostClassifier(
        **params,
        loss_function='Logloss',
        cat_features=[],  # Add categorical features indices if necessary
        verbose=0,
        random_state=42
    )
    
    # Split the training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the validation set
    y_val_pred = model.predict(X_val)
    
    # Calculate the validation MCC
    val_mcc = matthews_corrcoef(y_val, y_val_pred)
    
    return val_mcc


# In[ ]:


# Create a study and optimize the objective
study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=50)

# Get the best hyperparameters
best_params = study.best_params


# In[ ]:


# Retrieve the best parameters from Optuna
best_params = study.best_params

# Initialize and train the best CatBoost model
best_catboost_model = CatBoostClassifier(
    **best_params,
    loss_function='Logloss',
    cat_features=[],  # Add categorical features indices if necessary
    verbose=0,
    random_state=42
)

# Train on the full dataset
best_catboost_model.fit(X, y)

# Predict on the test data
best_catboost_test_preds = best_catboost_model.predict_proba(test)[:, 1]

# Add to model_performance
model_performance['Best CatBoost'] = {'val_mcc': None, 'test_preds': best_catboost_test_preds}


# # <p style="background-color: #ad3931; color: #fcf8f7; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #A57CDB; border: 5px dotted #fcf8f7;">Lets Create the Submission File</p>
# 

# In[ ]:


# Reverse the encoding for the predictions (if applicable)
# Assuming 'label_encoder' was used to encode the target variable 'y' earlier
final_predictions = (best_catboost_test_preds > 0.5).astype(int)

# Convert binary predictions to the original class labels
final_predictions_labels = label_encoder.inverse_transform(final_predictions)

# Load the sample submission file
sample_submission = pd.read_csv('/kaggle/input/playground-series-s4e8/sample_submission.csv')

# Directly replace the values in the target column of the sample submission file
# Assuming the target column is named 'target' in the sample submission
sample_submission.iloc[:, 1] = final_predictions_labels  # Replace the second column with predictions

# Save the submission file
sample_submission.to_csv('submission.csv', index=False)



# In[ ]:


sample_submission.head()


# # <p style="background-color: #ad3931; color: #fcf8f7; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #A57CDB; border: 5px dotted #fcf8f7;">Conclusion</p>
# 

# * This is the Binary Prediction of Poisonous Mushrooms Competition Dataset.
# * It consists of three Files such as Train, test and sample submission
# * The Training data consists of 3116945 rows and 21 columns. This is the Binary Prediction of Poisonous Mushrooms Competition Dataset.
# * It consists of three Files such as Train, test and sample submission
# * The Training data consists of 3116945 rows and 21 columns
# * The test Data consists of 2077964 rows and 20 columns.
# * while the sample submission consists of 2077964 rows and 2 columns.
# * I also took Secondary Mushroom Dataset for little improvement in mcc
# * The Dataset Secondary Mushroom of simulated mushrooms is used for binary classification into edible and poisonous.
# * This dataset includes 61069 hypothetical mushrooms with caps based on 173 species (353 mushrooms per species). 
# * The aim to take this competition Dataset is to predict whether a mushroom is edible or poisonous based on its physical characteristics.
# * For this I firstly concatenate the train data of competition with original Data and then took the detailed overview about the data and then I create various visualization plots in the form of subplots such as countplot, pairplot, Histogram, Facetgrid Plot, ViolinPlot and piechart to take deep insights about Data.
# * Then I visualize the outliers through Boxplot and apply the LogNormalization Technique to take accuracte and best results.
# * Then I take a look at the Duplicates of the Data. There is no duplicates present in competition data but when i combine the train and original data i get 146.
# * Then I Drop the Duplicates.
# * Following this, I use the KFold Cross Validation, perform model training, and choose the best model to take improve predictions.
# * Then I took the Best Model and then apply Hyperparameter Tuning on it and took more better results.
# * After applying the Kfold cross validation and choose the best model I when I do the Hyperparameter Tuning it gives more better or amazing results.
# 

# # <p style="background-color: #ad3931; color: #fcf8f7; font-family: 'Arial', sans-serif; font-size: 30px; text-align: center; padding: 25px 8px; border-radius: 10px; width: 90%; margin: 0 auto; font-weight: bold; text-shadow: 2px 2px 4px #A57CDB; border: 5px dotted #fcf8f7;">Thanks for taking the time to explore my Notebook. If you have any Question Feel Free to Ask</p>
# 
# 
