#!/usr/bin/env python
# coding: utf-8

# # **MUSHROOM PREDICTION(POSIONOUS OR EDIBLE??)**
# * The objective of this competiton is to predict whether the mushroom is poisonous or edible. 
# * There are 22 features like cap diameter, cap-shape, gill-color, gill-shape etc and approx 3116945 rows.
# * Matthews correlation coefficient (MCC) was used as a metric for evaluation in this competition.
# * XGBOOST with Matthews Correlation Coefficient (MCC) of 0.98492 on public leaderboard(20% of test data) and 0.98479 on private leaderboard(80% of test data).
# * Ranked in the top 25% of the competition.
# 
# 
# ![image.png](attachment:fce1c2fb-787a-416b-b2ce-3b72b4e25c7a.png)

# # **LOADING DATA**

# In[ ]:


import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


#Converting to dataframe
df=pd.read_csv('/kaggle/input/playground-series-s4e8/train.csv')
df.head()
df2=pd.read_csv('/kaggle/input/playground-series-s4e8/test.csv')


# In[ ]:


#Droping the id column
df=df.drop(['id'],axis=1)


# In[ ]:


df.head()


# In[ ]:


df.shape


# **CHECKING THE NULL VALUES**
# 

# In[ ]:


df.isnull().sum()


# In[ ]:


null_counts = df.isnull().sum()
total_elements = df.shape[0]  # Number of rows
null_percentage = (null_counts / total_elements) * 100
null_percentage


# # **EXPLORATORY DATA ANALYSIS**

# In[ ]:


#BAR CHART OF MISSING VALUES
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

null_counts = df.isnull().sum()

norm = plt.Normalize(null_counts.min(), null_counts.max())

colors = plt.cm.RdYlGn_r(norm(null_counts))  

# Plot a bar chart
plt.figure(figsize=(12, 8))
plt.bar(null_counts.index, null_counts.values, color=colors, edgecolor='black')
plt.xticks(rotation=45, ha='right')  
plt.xlabel('Columns', fontsize=14)
plt.ylabel('Number of Null Values', fontsize=14)
plt.title('Number of Null Values for Each Column', fontsize=16)


plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{int(x / 1000)}K'))


plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout() 
plt.show()


# * *As you can see there are many null values from the graph which need to be taken care of.*

# In[ ]:


#PIE CHART OF THE TARGET LABELS IN THE DATA
import matplotlib.pyplot as plt

class_counts = df['class'].value_counts()


plt.figure(figsize=(8, 6))  
plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)

plt.title('Class Distribution in the Data')
plt.axis('equal')  
plt.show()


# * *Here the data is not imbalanced.*

# In[ ]:


import warnings

warnings.filterwarnings("ignore")


# In[ ]:


#HISTOGRAMS FOR NUMERICAL DATA
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
sns.histplot(df['cap-diameter'], kde=True, color='purple')
plt.title('Cap Diameter Distribution')

plt.subplot(1, 3, 2)
sns.histplot(df['stem-height'], kde=True, color='purple')
plt.title('Stem Height Distribution')

plt.subplot(1, 3, 3)
sns.histplot(df['stem-width'], kde=True, color='purple')
plt.title('Stem Width Distribution')

plt.tight_layout()
plt.show()


# * *Here the numerical features are somewhat right skewed.*

# In[ ]:


#Pair Plot
sns.pairplot(df[['cap-diameter', 'stem-height', 'stem-width']], plot_kws={'color': '#17becf'})
plt.suptitle('Pairwise Scatter Plots', y=1.02)
plt.show()


# In[ ]:


#SUMMARY STATITICS
summary_stats = pd.DataFrame({
    'Mean': [df['cap-diameter'].mean(), df['stem-height'].mean(), df['stem-width'].mean()],
    'Median': [df['cap-diameter'].median(), df['stem-height'].median(), df['stem-width'].median()],
    'Standard Deviation': [df['cap-diameter'].std(), df['stem-height'].std(), df['stem-width'].std()]
}, index=['Cap Diameter', 'Stem Height', 'Stem Width'])

# Plot
summary_stats.plot(kind='bar', figsize=(10, 6), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.title('Summary Statistics for Numerical Features')
plt.ylabel('Value')
plt.show()


# In[ ]:


#VIOLIN PLOT
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
sns.violinplot(y=df['cap-diameter'], color='purple')
plt.title('Cap Diameter Violin Plot')

plt.subplot(1, 3, 2)
sns.violinplot(y=df['stem-height'], color='purple')
plt.title('Stem Height Violin Plot')

plt.subplot(1, 3, 3)
sns.violinplot(y=df['stem-width'], color='purple')
plt.title('Stem Width Violin Plot')

plt.tight_layout()
plt.show()


# * *These plots help to visualise where the majority of data lies for these 3 features.* 

# In[ ]:


#DISTRIBUTION OF CATEGORICAL FEATURES OF TOP 5 CATEGORIES 
target_column = 'class'

categorical_columns = df.select_dtypes(include=['object']).columns
categorical_columns = [col for col in categorical_columns if col != target_column]

fig, axes = plt.subplots(nrows=len(categorical_columns), ncols=1, figsize=(8, 4 * len(categorical_columns)))

for i, col in enumerate(categorical_columns):
    top_categories = df[col].value_counts().nlargest(5).index

    filtered_df = df[df[col].isin(top_categories)]

    cross_tab = pd.crosstab(filtered_df[col], filtered_df[target_column])

    ax = axes[i]
    cross_tab.plot(kind='bar', stacked=True, colormap='viridis', ax=ax)
    ax.set_title(f'Top 5 {col} Distribution by Class')
    ax.set_ylabel('Count (K)')
    ax.set_xlabel(col)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    
    ticks_loc = ax.get_yticks().tolist()
    ax.set_yticklabels(['{:.0f}K'.format(x/1000) for x in ticks_loc])

plt.tight_layout()
plt.show()


# **DATA CLEANING**

# In[ ]:


#DROPPING THE VEIL-TYPE COLUMN AS THERE WERE MANY NULL VALUES AMD WAS NOT IMPROVING THE ACCURACY
df=df.drop(['veil-type'],axis=1)
df2=df2.drop(['veil-type'],axis=1)


# In[ ]:


df.head()


# # **HANDLING NUMERICAL DATA**
# 
# * I replaced NaN(null values) with median.
# * I applied log normal transformation as these features were not normally distributed as they were right skewed.
# * I have also added code to remove the outliers i.e. the data points which are 3 std dev away, but surprisingly removing the outliers  was not advantageous in this case.

# **HANDLING CAP DIAMETER**

# In[ ]:


#Imputing null values with median
median_value = df['cap-diameter'].median()
df['cap-diameter'].fillna(median_value, inplace=True)
df2['cap-diameter'].fillna(median_value, inplace=True)


# In[ ]:




import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))  # Set figure size
sns.kdeplot(df['cap-diameter'], shade=True, color='blue')
plt.title('KDE Distribution of Cap Diameter')
plt.xlabel('Cap Diameter')
plt.ylabel('Density')
plt.grid(True)
plt.show()


# In[ ]:


#Applying log normal transformation to get gaussian distribution
df['cap-diameter'] = df['cap-diameter'] + 1e-6  # Adding a small constant to avoid log(0)
df2['cap-diameter'] = df2['cap-diameter'] + 1e-6  # Adding a small constant to avoid log(0)

# Apply log-normal transformation
df['cap-diameter'] = np.log(df['cap-diameter'])
df2['cap-diameter'] = np.log(df2['cap-diameter'])


# In[ ]:





plt.figure(figsize=(10, 6))  # Set figure size
sns.kdeplot(df['cap-diameter'], shade=True, color='blue')
plt.title('KDE Distribution of Cap Diameter')
plt.xlabel('Cap Diameter')
plt.ylabel('Density')
plt.grid(True)
plt.show()


# In[ ]:





# Plotting the boxplot for 'cap-diameter'
plt.figure(figsize=(10, 6))
plt.boxplot(df['cap-diameter'])
plt.title('Boxplot of Cap Diameter')
plt.ylabel('Cap Diameter')
plt.grid(True)
plt.show()


# In[ ]:


# #IMP
# #Removing the outliers which is 3 standard dev away
# # Calculate mean and standard deviation
# mean_diameter = df['cap-diameter'].mean()
# std_diameter = df['cap-diameter'].std()

# # Define the boundaries: mean ± 3 standard deviations
# lower_boundary = mean_diameter - 3 * std_diameter
# upper_boundary = mean_diameter + 3 * std_diameter

# # Filtering the DataFrame to remove values outside the boundaries
# df = df[(df['cap-diameter'] >= lower_boundary) & (df['cap-diameter'] <= upper_boundary)]

# print(f'Lower Boundary: {lower_boundary}')
# print(f'Upper Boundary: {upper_boundary}')
# df.shape


# In[ ]:


df.head()


# **HANDLING STEM HEIGHT**

# In[ ]:


#Imputing null values with median
median_value = df['stem-height'].median()
df['stem-height'].fillna(median_value, inplace=True)
df2['stem-height'].fillna(median_value, inplace=True)


# In[ ]:



#Checking the distribution
plt.figure(figsize=(10, 6))  # Set figure size
sns.kdeplot(df['stem-height'], shade=True, color='blue')
plt.title('KDE Distribution of Stem Height')
plt.xlabel('Stem Height')
plt.ylabel('Density')
plt.grid(True)
plt.show()


# In[ ]:


#Applying log normal transformation to get gaussian distribution
df['stem-height'] = df['stem-height'] + 1e-6  # Adding a small constant to avoid log(0)
df2['stem-height'] = df2['stem-height'] + 1e-6  # Adding a small constant to avoid log(0)

# Apply log-normal transformation
df['stem-height'] = np.log(df['stem-height'])
df2['stem-height'] = np.log(df2['stem-height'])


# In[ ]:




#Checking the distribution
plt.figure(figsize=(10, 6))  # Set figure size
sns.kdeplot(df['stem-height'], shade=True, color='blue')
plt.title('KDE Distribution of stem height')
plt.xlabel('Stem Height')
plt.ylabel('Density')
plt.grid(True)
plt.show()


# In[ ]:


# Plotting the boxplot for 'stem-height'
plt.figure(figsize=(10, 6))
plt.boxplot(df['stem-height'])
plt.title('Boxplot of Stem Height')
plt.ylabel('stem height')
plt.grid(True)
plt.show()


# In[ ]:


df.shape


# In[ ]:


# #Removing the outliers which is 3 standard dev away
# # Calculate mean and standard deviation
# mean_diameter = df['stem-height'].mean()
# std_diameter = df['stem-height'].std()

# # Define the boundaries: mean ± 3 standard deviations
# lower_boundary = mean_diameter - 3 * std_diameter
# upper_boundary = mean_diameter + 3 * std_diameter

# # Filter the DataFrame to remove values outside the boundaries
# df = df[(df['stem-height'] >= lower_boundary) & (df['stem-height'] <= upper_boundary)]

# print(f'Lower Boundary: {lower_boundary}')
# print(f'Upper Boundary: {upper_boundary}')
# df.shape


# **HANDLING STEM WIDTH**

# In[ ]:


#Imputing null values with median
median_value = df['stem-width'].median()
df['stem-width'].fillna(median_value, inplace=True)
df2['stem-width'].fillna(median_value, inplace=True)


# In[ ]:



#Checking the distribution
plt.figure(figsize=(10, 6))  # Set figure size
sns.kdeplot(df['stem-width'], shade=True, color='blue')
plt.title('KDE Distribution of Stem Width')
plt.xlabel('Stem Width')
plt.ylabel('Density')
plt.grid(True)
plt.show()


# In[ ]:


#Applying log normal transformation to get gaussian distribution
df['stem-width'] = df['stem-width'] + 1e-6  # Adding a small constant to avoid log(0)
df2['stem-width'] = df2['stem-width'] + 1e-6  # Adding a small constant to avoid log(0)

# Apply log-normal transformation
df['stem-width'] = np.log(df['stem-width'])
df2['stem-width'] = np.log(df2['stem-width'])


# In[ ]:




#Checking the distribution
plt.figure(figsize=(10, 6))  # Set figure size
sns.kdeplot(df['stem-width'], shade=True, color='blue')
plt.title('KDE Distribution of stem Width')
plt.xlabel('Stem Width')
plt.ylabel('Density')
plt.grid(True)
plt.show()


# In[ ]:



##IMP


# # Plotting the boxplot for 'stem-width'
# plt.figure(figsize=(10, 6))
# plt.boxplot(df['stem-width'])
# plt.title('Boxplot of Stem width')
# plt.ylabel('stem width')
# plt.grid(True)
# plt.show()


# In[ ]:


df.shape


# In[ ]:


# #Removing the outliers which is 3 standard dev away
# # Calculate mean and standard deviation
# mean_diameter = df['stem-width'].mean()
# std_diameter = df['stem-width'].std()

# # Define the boundaries: mean ± 3 standard deviations
# lower_boundary = mean_diameter - 3 * std_diameter
# upper_boundary = mean_diameter + 3 * std_diameter

# # Filter the DataFrame to remove values outside the boundaries
# df = df[(df['stem-width'] >= lower_boundary) & (df['stem-width'] <= upper_boundary)]

# print(f'Lower Boundary: {lower_boundary}')
# print(f'Upper Boundary: {upper_boundary}')
# df.shape


# In[ ]:


df.head()


# # **HANDLING CATEGORICAL FEATURES**
# * Imputed the missing value with the mode.
# * I cleaned the columns by replacing erroneous categories with the most frequent categories. 
# * I applied one-hot encoding to the most frequent categories and created a new column to indicate whether the numerical feature contains missing values.

# In[ ]:


#FUNCTION FOR ONE HOT ENCODING
def one_hot_encode(df, column_name, categories):
    # Creating a new DataFrame with the specified categories
    df_encoded = pd.DataFrame(0, index=df.index, columns=[f"{column_name}_{cat}" for cat in categories])

    # Performing one-hot encoding for the specified categories
    for category in categories:
        df_encoded[f"{column_name}_{category}"] = (df[column_name] == category).astype(int)
    
    # Dropping the original column and concatenate the new encoded columns
    df = df.drop(column_name, axis=1)
    df = pd.concat([df, df_encoded], axis=1)

    return df


# **HANDLING SPORE COLOR**

# In[ ]:




# In[ ]:




# In[ ]:


spore_print_color_mode = df['spore-print-color'].mode()[0]
df['spore_print_color_missing'] = df['spore-print-color'].isnull().astype(int)
df['spore-print-color'].fillna(spore_print_color_mode, inplace=True)


# In[ ]:


spore_print_color_mode = df['spore-print-color'].mode()[0]
df2['spore_print_color_missing'] = df2['spore-print-color'].isnull().astype(int)
df2['spore-print-color'].fillna(spore_print_color_mode, inplace=True)


# In[ ]:



categories = ['k','p','n','w','r','u','g']

df = one_hot_encode(df, 'spore-print-color', categories)
df2 = one_hot_encode(df2, 'spore-print-color', categories)




# **HANDLING VEIL COLOR**

# In[ ]:




# In[ ]:




# In[ ]:


df['veil-color'].isnull().sum()


# In[ ]:




veil_color_mode = df['veil-color'].mode()[0]

df['veil_color_missing'] = df['veil-color'].isnull().astype(int)

df['veil-color'].fillna(veil_color_mode, inplace=True)



# In[ ]:


veil_color_mode = df['veil-color'].mode()[0]
df2['veil_color_missing'] = df2['veil-color'].isnull().astype(int)
df2['veil-color'].fillna(veil_color_mode, inplace=True)


# In[ ]:



categories = ['w','y','n','u','k','e']

df = one_hot_encode(df, 'veil-color', categories)
df2 = one_hot_encode(df2, 'veil-color', categories)




# **HANDLING STEM ROOT**

# In[ ]:




# In[ ]:




# In[ ]:


df['stem-root'].isnull().sum()


# In[ ]:


stem_root_mode = df['stem-root'].mode()[0]
df['stem_root_missing'] = df['stem-root'].isnull().astype(int)
df['stem-root'].fillna(veil_color_mode, inplace=True)


# In[ ]:


stem_root_mode = df['stem-root'].mode()[0]
df2['stem_root_missing'] = df2['stem-root'].isnull().astype(int)
df2['stem-root'].fillna(veil_color_mode, inplace=True)


# In[ ]:


categories = ['b','s','r','c']

df = one_hot_encode(df, 'stem-root', categories)
df2 = one_hot_encode(df2, 'stem-root', categories)




# In[ ]:


df2.shape


# **HANDLING CAP SURFACE**

# In[ ]:




# In[ ]:




# In[ ]:


df['cap-surface'].isnull().sum()


# In[ ]:



cap_surface_mode = df['cap-surface'].mode()[0]

df['cap-surface_missing'] = df['cap-surface'].isnull().astype(int)

df['cap-surface'].fillna(cap_surface_mode, inplace=True)



# In[ ]:


df2['cap-surface'].isnull().sum()


# In[ ]:


df2['cap-surface_missing'] = df2['cap-surface'].isnull().astype(int)

df2['cap-surface'].fillna(cap_surface_mode, inplace=True)



# In[ ]:



categories = ['t', 'y', 's', 'h', 'g', 'd', 'k','e','i','w','l']

df = one_hot_encode(df, 'cap-surface', categories)
df2 = one_hot_encode(df2, 'cap-surface', categories)




# **HANDLING GILL ATTACHMENT**

# In[ ]:




# In[ ]:




# In[ ]:


df2['gill-attachment'].isnull().sum()


# In[ ]:



gill_attachment_mode = df['gill-attachment'].mode()[0]

df['gill-attachment_missing'] = df['gill-attachment'].isnull().astype(int)

df['gill-attachment'].fillna(gill_attachment_mode, inplace=True)



# In[ ]:



gill_attachment_mode = df['gill-attachment'].mode()[0]
df2['gill-attachment_missing'] = df2['gill-attachment'].isnull().astype(int)
df2['gill-attachment'].fillna(gill_attachment_mode, inplace=True)


# In[ ]:



categories = ['a', 'd', 's', 'e', 'x', 'p', 'f']
df = one_hot_encode(df, 'gill-attachment', categories)
df2 = one_hot_encode(df2, 'gill-attachment', categories)




# **HANDLING GILL SHAPE**

# In[ ]:




# In[ ]:




# In[ ]:


df['gill-spacing'].isnull().sum()


# In[ ]:


gill_spacing_mode = df['gill-spacing'].mode()[0]
df['gill_spacing_missing'] = df['gill-spacing'].isnull().astype(int)
df['gill-spacing'].fillna(gill_spacing_mode, inplace=True)


# In[ ]:


gill_spacing_mode = df['gill-spacing'].mode()[0]
df2['gill_spacing_missing'] = df2['gill-spacing'].isnull().astype(int)
df2['gill-spacing'].fillna(gill_spacing_mode, inplace=True)


# In[ ]:


categories = ['c','d','f']


df = one_hot_encode(df, 'gill-spacing', categories)
df2 = one_hot_encode(df2, 'gill-spacing', categories)




# **HANDLING CAP SHAPE**

# In[ ]:




# In[ ]:



df.loc[:, 'cap-shape'] = df['cap-shape'].fillna('x')
df2.loc[:, 'cap-shape'] = df2['cap-shape'].fillna('x')


# In[ ]:


df['cap-shape'].unique()


# In[ ]:




# In[ ]:




# In[ ]:


df.shape


# In[ ]:


df['cap-shape'].unique()


# In[ ]:


df['cap-shape'].isnull().sum()


# In[ ]:


categories = ['x', 'f', 's', 'b', 'c', 'p', 'o']
df = one_hot_encode(df, 'cap-shape', categories)
df2 = one_hot_encode(df2, 'cap-shape', categories)




# **HANDLING STEM SURFACE**

# In[ ]:




# In[ ]:




# In[ ]:


df['stem-surface'].isnull().sum()


# In[ ]:



stem_surface_mode = df['stem-surface'].mode()[0]

df['stem_surface_missing'] = df['stem-surface'].isnull().astype(int)

df['stem-surface'].fillna(stem_surface_mode, inplace=True)



# In[ ]:



stem_surface_mode = df['stem-surface'].mode()[0]

df2['stem_surface_missing'] = df2['stem-surface'].isnull().astype(int)

df2['stem-surface'].fillna(stem_surface_mode, inplace=True)



# In[ ]:



categories = ['s', 'y', 'i', 't', 'g', 'k', 'h']


df = one_hot_encode(df, 'stem-surface', categories)
df2 = one_hot_encode(df2, 'stem-surface', categories)




# **HANDLING CAP COLOR**

# In[ ]:


df['cap-color'].fillna('n', inplace=True)
df2['cap-color'].fillna('n', inplace=True)


# In[ ]:


df['cap-color'].unique()


# In[ ]:




# In[ ]:




# In[ ]:


df['cap-color'].isnull().sum()


# In[ ]:


df['cap-color'].unique()


# In[ ]:


categories = ['n','w','y','g','e','o','p','r','u','b','k','l']

df = one_hot_encode(df, 'cap-color', categories)
df2 = one_hot_encode(df2, 'cap-color', categories)




# **HANDLING DOES BRUISE OR BLEED**

# In[ ]:


df['does-bruise-or-bleed'].isnull().sum()


# In[ ]:


df['does-bruise-or-bleed'].unique()


# In[ ]:


df['does-bruise-or-bleed'].value_counts()


# In[ ]:



df['does-bruise-or-bleed'].fillna('f', inplace=True)
df2['does-bruise-or-bleed'].fillna('f', inplace=True)


# In[ ]:


df['does-bruise-or-bleed'].isnull().sum()


# In[ ]:


df.shape


# In[ ]:



categories = ['t', 'f']




df = one_hot_encode(df, 'does-bruise-or-bleed', categories)
df2 = one_hot_encode(df2, 'does-bruise-or-bleed', categories)




# **HANDLING GILL COLOR**

# In[ ]:


df['gill-color'].value_counts()


# In[ ]:


df['gill-color'].fillna('w', inplace=True)
df2['gill-color'].fillna('w', inplace=True)


# In[ ]:


df['gill-color'].isnull().sum()


# In[ ]:




# In[ ]:




# In[ ]:



categories = ['w', 'n', 'y', 'p', 'g', 'o', 'k', 'f', 'r', 'e', 'b', 'u']
df = one_hot_encode(df, 'gill-color', categories)
df2 = one_hot_encode(df2, 'gill-color', categories)




# **HANDLING STEM COLOR**

# In[ ]:


df['stem-color'].value_counts()


# In[ ]:


df2['stem-color'].value_counts()


# In[ ]:



df['stem-color'].fillna('w', inplace=True)
df2['stem-color'].fillna('w', inplace=True)


# In[ ]:


categories = [ 'w', 'n', 'y', 'g', 'o', 'e', 'u', 'p', 'k', 'r', 'l', 'b']

df = one_hot_encode(df, 'stem-color', categories)
df2 = one_hot_encode(df2, 'stem-color', categories)




# **HANDLING HAS RING**

# In[ ]:


df['has-ring'].isnull().sum()


# In[ ]:



df['has-ring'].fillna('f', inplace=True)
df2['has-ring'].fillna('f', inplace=True)


# In[ ]:


df['has-ring'].value_counts()


# In[ ]:


categories = [ 't', 'f']

df = one_hot_encode(df, 'has-ring', categories)
df2 = one_hot_encode(df2, 'has-ring', categories)




# ## 

# **HANDLING RING TYPE**

# In[ ]:


df['habitat'].isnull().sum()


# In[ ]:


df['habitat'].value_counts()


# In[ ]:


df2['habitat'].value_counts()


# In[ ]:


df['habitat'].fillna('d', inplace=True)
df2['habitat'].fillna('d', inplace=True)


# In[ ]:


categories = [ 'd', 'g', 'l', 'm', 'h', 'w', 'p', 'u']


df = one_hot_encode(df, 'habitat', categories)
df2 = one_hot_encode(df2, 'habitat', categories)




# **HANDLING SEASON**

# In[ ]:


df['season'].isnull().sum()


# In[ ]:


df['season'].fillna('a', inplace=True)
df2['season'].fillna('a', inplace=True)


# In[ ]:


df['season'].value_counts()


# In[ ]:



categories = [ 's', 'w', 'a', 'u']
df = one_hot_encode(df, 'season', categories)
df2 = one_hot_encode(df2, 'season', categories)




# **HANDLING RING TYPE**

# In[ ]:


df['ring-type'].value_counts()


# In[ ]:


df2['ring-type'].value_counts()


# In[ ]:


df['ring-type'].isnull().sum()


# In[ ]:


ring_type_mode = df['ring-type'].mode()[0]
df['ring_type_missing'] = df['ring-type'].isnull().astype(int)
df['ring-type'].fillna(ring_type_mode, inplace=True)


# In[ ]:


ring_type_mode = df['ring-type'].mode()[0]
df2['ring_type_missing'] = df2['ring-type'].isnull().astype(int)
df2['ring-type'].fillna(ring_type_mode, inplace=True)


# In[ ]:



categories = [ 'f','e','z','l','r','p','g','m']

df = one_hot_encode(df, 'ring-type', categories)
df2 = one_hot_encode(df2, 'ring-type', categories)




# In[ ]:


df.head()


# In[ ]:


df2.head()


# # **SPLITTING THE DATASET**
# * Splitting the data into train and validation as 95:5 because the data is very large and there was plenty of it for testing.

# In[ ]:


y = df.iloc[:, 0] 
X = df.iloc[:, 1:]  


# In[ ]:


# Label encoding the target variable y: 'e' -> 0 and 'p' -> 1
y = y.map({'e': 0, 'p': 1})


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=50,stratify=y)



# **SCALING THE NUMERICAL FEATURES**

# In[ ]:


from sklearn.preprocessing import StandardScaler

# List of numerical columns to scale
numerical_columns = ['cap-diameter', 'stem-height', 'stem-width']
standard_scaler = StandardScaler()
X_train[numerical_columns] = standard_scaler.fit_transform(X_train[numerical_columns])
X_val[numerical_columns] = standard_scaler.transform(X_val[numerical_columns])


# In[ ]:


# Counting the number of 0 and 1 labels in y_train and y_val
train_label_counts = y_train.value_counts()
val_label_counts = y_val.value_counts()




# # **MODEL TRAINING AND EVALUATION**
# * I tried 3 models XGBOOST, CATBOOST and DEEP NEURAL NETWORKS and found XGBOOST the best one
# * Used optuna for tuning the hyperparameters.
# * Optuna is an open-source hyperparameter optimization framework designed to automate the process of hyperparameter tuning in machine learning models. It provides an efficient and flexible way to find the best hyperparameters for algorithms by using sophisticated optimization techniques.
# ![image.png](attachment:cd66a734-005b-4a17-99c6-df2e2d7e2668.png)
# 

# **XGBOOST**

# In[ ]:


# #Hyperparameter tuning with OPTUNA
# import optuna
# import xgboost as xgb
# from sklearn.metrics import accuracy_score, matthews_corrcoef
# from sklearn.model_selection import train_test_split

# # Define your objective function for Optuna
# def objective(trial):
#     # Define the hyperparameter search space
#     params = {
#         'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
#         'max_depth': trial.suggest_int('max_depth', 3, 15),
#         'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
#         'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
#         'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.3, 1.0),
#         'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
#         'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
#         'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
#         'tree_method': 'gpu_hist',  # Use GPU
#         'gpu_id': 0
#     }

#     # Initialize XGBoost Classifier with current trial's hyperparameters
#     xgb_clf = xgb.XGBClassifier(**params)

#     # Train the classifier
#     xgb_clf.fit(X_train, y_train)

#     # Predict on the validation set
#     y_pred = xgb_clf.predict(X_val)

#     # Calculate the Matthews correlation coefficient (MCC)
#     mcc = matthews_corrcoef(y_val, y_pred)

#     return mcc  # Optuna will maximize this value

# # Create a study object and specify that we want to maximize the MCC
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=50)  # You can adjust the number of trials

# # Output the best trial results
# print(f"Best trial: MCC {study.best_trial.value:.4f}, Parameters: {study.best_trial.params}")





# In[ ]:


import xgboost as xgb
from sklearn.metrics import accuracy_score, matthews_corrcoef

# HYPERPARAMETERS ARE TUNED USING OPTUNA
xgb_clf = xgb.XGBClassifier(
    learning_rate=0.03866478165917683,
    max_depth=15,
    n_estimators=537,
    subsample=0.5308198191299447,
    colsample_bytree=0.39780439228131836,
    gamma=0.00317117984524131,
    reg_alpha=0.00013252041342870703,
    reg_lambda=8.600311922560845e-07,
    tree_method='gpu_hist', 
    gpu_id=0,
    objective='binary:logistic'
)

xgb_clf.fit(X_train, y_train)
y_pred = xgb_clf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
mcc = matthews_corrcoef(y_val, y_pred)





# In[ ]:


from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_val, y_pred)
# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[ ]:


#PRECISION, RECALL, F1 SCORE
from sklearn.metrics import precision_score, recall_score, f1_score


precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

# Print the results


# **CATBOOST**

# In[ ]:



##CATBOOST



# #Catboost
# from catboost import CatBoostClassifier
# from sklearn.metrics import accuracy_score, matthews_corrcoef

# # Initialize the CatBoost Classifier with some basic parameters
# catboost_clf = CatBoostClassifier(iterations=1500,  # Number of boosting iterations
#                                   depth=13,          # Depth of the tree
#                                   learning_rate=0.05, # Learning rate
#                                   verbose=100,      # Logging frequency
#                                   task_type='GPU',
#                                   devices=[0,1],
# #                                   eval_metric='Logloss',
#                                   max_bin=300
# #                                   scale_pos_weight=0.825,
# #                                   grow_policy='Lossguide'
# #                                   bagging_temperature=0.6,
# #                                   random_strength=0.6,
# #                                   border_count=300
                            
#                                   )     

# # Train the classifier on the training data
# catboost_clf.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)

# # Predict on the validation set
# y_pred = catboost_clf.predict(X_val)

# # Calculate accuracy and Matthews correlation coefficient
# accuracy = accuracy_score(y_val, y_pred)
# mcc = matthews_corrcoef(y_val, y_pred)

# # Print the results
# print(f"CatBoost Classifier - Accuracy: {accuracy:.4f}, MCC: {mcc:.4f}")



#TUNING CATBBOST WITH OPTUNA
# import optuna
# from catboost import CatBoostClassifier
# from sklearn.metrics import accuracy_score, matthews_corrcoef

# def objective(trial):
#     # Define the hyperparameter search space
#     params = {
#         'iterations': trial.suggest_int('iterations', 100, 2000),
#         'depth': trial.suggest_int('depth', 4, 16),
#         'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
#         'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1, 100),
#         'border_count': trial.suggest_int('border_count', 32, 350),
#         'random_strength': trial.suggest_loguniform('random_strength', 1e-5, 1e+3),
#         'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
# #         'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Lossguide', 'Depthwise']),
# #         'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'Poisson'])
#     }

#     # Initialize CatBoost Classifier with current trial's hyperparameters
#     catboost_clf = CatBoostClassifier(
#         iterations=params['iterations'],
#         depth=params['depth'],
#         learning_rate=params['learning_rate'],
#         l2_leaf_reg=params['l2_leaf_reg'],
#         border_count=params['border_count'],
#         random_strength=params['random_strength'],
#         min_data_in_leaf=params['min_data_in_leaf'],
#         #grow_policy=params['grow_policy'],
#         #bootstrap_type=params['bootstrap_type'],  # Use supported bootstrap types
#         task_type='GPU',
#         devices=[0,1]  # Adjust if you want to use multiple GPUs
#     )

#     # Train the classifier
#     catboost_clf.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=0)

#     # Predict on the validation set
#     y_pred = catboost_clf.predict(X_val)

#     # Calculate the Matthews correlation coefficient (MCC)
#     mcc = matthews_corrcoef(y_val, y_pred)

#     return mcc  # Optuna will maximize this value

# # Create a study object and specify that we want to maximize the MCC
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=50)  # You can adjust the number of trials

# # Output the best trial results
# print(f"Best trial: MCC {study.best_trial.value:.4f}, Parameters: {study.best_trial.params}")

# # Train the CatBoost model with the best parameters found by Optuna
# best_params = study.best_trial.params
# best_catboost_clf = CatBoostClassifier(
#     iterations=best_params['iterations'],
#     depth=best_params['depth'],
#     learning_rate=best_params['learning_rate'],
#     l2_leaf_reg=best_params['l2_leaf_reg'],
#     border_count=best_params['border_count'],
#     random_strength=best_params['random_strength'],
#     min_data_in_leaf=best_params['min_data_in_leaf'],
#     #grow_policy=best_params['grow_policy'],
#     #bootstrap_type=best_params['bootstrap_type'],  # Ensure this matches
#     task_type='GPU',
#     devices=[0, 1]  # Adjust if you want to use multiple GPUs
# )
# best_catboost_clf.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=0)

# # Predict on the validation set with the best model
# y_pred_best = best_catboost_clf.predict(X_val)

# # Calculate accuracy and Matthews correlation coefficient for the best model
# accuracy_best = accuracy_score(y_val, y_pred_best)
# mcc_best = matthews_corrcoef(y_val, y_pred_best)

# # Print the results for the best model
# print(f"Best CatBoost Classifier - Accuracy: {accuracy_best:.4f}, MCC: {mcc_best:.4f}")


# **Testing the model on test data and creating submission file**

# In[ ]:


X_test = df2.drop('id', axis=1)



X_test[numerical_columns] =standard_scaler.transform(X_test[numerical_columns])
y_pred_test = xgb_clf.predict(X_test)

# Map numeric predictions to original classes 'e' and 'p'
label_mapping = {0: 'e', 1: 'p'}
predicted_classes = pd.Series(y_pred_test).map(label_mapping)

submission = pd.DataFrame({
    'id': df2['id'].values,   
    'class': predicted_classes 
})


num_ids_submission = submission['id'].notnull().sum()

# Save the submission DataFrame to a CSV file
submission.to_csv('submission1111.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




