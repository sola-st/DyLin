#!/usr/bin/env python
# coding: utf-8

# # Effects of NaN (missing) values in your data on scaling, normalization, encoding, and model training:
# 
# ### 1. **Scaling and Normalization:**
#    - **Impact**: Scaling and normalization methods such as Min-Max scaling, standardization, and other transformations cannot handle NaN values. These methods will either fail or ignore NaN values, which can lead to incorrect scaling or normalization of your data.
#    - **Consequence**: The presence of NaN values can lead to biased transformations, as these missing entries are not accounted for in the scaling calculations, potentially skewing the scale of the feature.
# 
# ### 2. **Encoding:**
#    - **Impact**: Encoding methods, such as One-Hot Encoding, Label Encoding, and Probability Ratio Encoding, do not handle NaN values well. Many encoding functions will either fail or skip the rows with missing values during the encoding process.
#    - **Consequence**: Missing values may result in incomplete encoding, causing data inconsistencies and potentially reducing the predictive power of categorical features.
# 
# ### 3. **Model Training:**
#    - **Impact**: Most machine learning algorithms cannot handle NaN values directly. Models such as Linear Regression, SVM, and many tree-based models (e.g., Decision Trees, Random Forest) will raise errors if NaN values are present in the dataset.
#    - **Consequence**: NaN values can lead to model training failures, biased estimations, or inaccurate predictions, as the model cannot learn patterns from incomplete data.
# 
# ### **Specific Effects on Different Models:**
#    - **Linear Models (e.g., Linear Regression, Logistic Regression)**: They will fail to fit the data and throw errors due to missing values.
#    - **Tree-Based Models (e.g., Decision Trees, Random Forests)**: Although some tree-based models can handle missing values by treating them as a separate category, this can still reduce the model’s performance as it may lead to suboptimal splits.
#    - **Neural Networks (e.g., ANN, CNN, RNN)**: These models do not inherently handle NaN values, leading to gradient issues and potential training failures.
# 
# ### **Summary:**
# Leaving NaN values untreated in your data can disrupt scaling, encoding, and model training, leading to errors, biased models, or incorrect interpretations. It is essential to handle missing values appropriately, such as through imputation, before proceeding with data transformation and model training.

# # Encoding
# #### **The Importance of Encoding in Feature Engineering**
# 
# Encoding is a vital step in feature engineering, which occurs during data preprocessing, enabling the conversion of raw data into informative features for machine learning models. This process is essential because machine learning algorithms can only process numerical data, and encoding allows us to transform categorical data into a numerical format that can be understood by these algorithms.
# 
# #### **Types of Encoding**
# 
# There are two primary methods of encoding:
# 
# #### **Nominal Encoding**
# 
# Nominal encoding converts categorical variables into numerical variables where ***the categories do not have a natural order.*** For example, encoding colors such as red, blue, and green into numerical values would be an example of nominal encoding. In this case, there is no inherent order or hierarchy between the colors, and the numerical values assigned to them are arbitrary. ***This encoding is best suited for features(column) with a small number of categories.***
# 
# #### **Ordinal Encoding**
# 
# Ordinal encoding converts categorical variables into numerical variables where ***the categories do have a natural order.*** For example, encoding education levels such as high school, bachelor's degree, and master's degree into numerical values would be an example of ordinal encoding. In this case, there is a clear hierarchy between the education levels, with higher levels indicating greater educational attainment. ***This encoding is best suited for features(column) with many categories.***

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Reading "Titanic" dataset
data = pd.read_csv("/kaggle/input/titanic/train.csv")

data['title'] = data.Name.str.split(",", expand=True)[1].str.split('.', expand=True)[0].str.strip()
data['Cabin'] = data['Cabin'].fillna('Missing')
data['Cabin Level'] = data['Cabin'].astype('str').str[0]
data.head()


# In[ ]:


value_df = data.copy(deep=True)
for col in value_df.columns:
    value_df[col] = value_df[col].astype('str')
    print(f"There are {value_df[col].nunique()} Unique valuse in Column '{col}'.")
    if data[col].nunique() < 150:
        print(f"\nUnique values in '{col}' column: {sorted(value_df[col].unique().tolist())}")
    print()
    print("--"*50)


# ## Nominal encoding:
# 
# * **Frequency Encoding or Count Encoding**: Count encoding involves replacing each category with the frequency or count of that category in the dataset. This method is useful when the frequency of a category is more important than its order or sequence. For example, encoding colors 'Red': 3, 'Blue': 2, 'Green': 1.
# 
# * **Target or Mean Encoding**: This method involves encoding categorical variables into numerical variables using the mean of the target variable. Use mean encoding for categorical variables with a large number of categories and a target variable. For example, encoding cities as the mean house price for New Delhi, Mumbai, and Chandigarh.
# 
# * **Probability Ratio Encoding**: This method involves encoding categorical variables into numerical variables based on the probability ratio of each category against the target variable. If a feature "Weather" has categories like "Sunny," "Rainy," and "Cloudy," Probability Ratio Encoding would assign numerical values to each based on their probability of occurrence with the target variable, such as a high chance of success.
# 
# * **Label Encoding**: This is a simple and intuitive method of encoding where each category is assigned a unique numerical value. Use label encoding when the categories are mutually exclusive and there is no natural order or ranking between them. For example, encoding colors as 0 for red, 1 for blue, and 2 for green.
# 
# * **One-Hot Encoding**: This method involves creating a binary vector for each category, where all values are 0 except for one value that is 1. Use one-hot encoding for categorical variables with a large number of categories. For example, encoding colors as [1, 0, 0] for red, [0, 1, 0] for blue, and [0, 0, 1] for green.
# 
# * **Top N One-Hot Encoding**: This method involves creating a binary vector for Top N (like 5, 10, 15) category, where all values are 0 except for one value that is 1. Use one-hot encoding for categorical variables with a large number of categories. For example, encoding colors as [1, 0, 0] for red, [0, 1, 0] for blue, and [0, 0, 0] for green (Not in Top N).
# 
# * **Dummy Encoding (also known as k-1 encoding)**: This method is similar to one-hot encoding except we create k-1 columns, where k is the number of categories. We drop one column, usually the first or last, to avoid multicollinearity and reduce the dimensionality of the data. For instance, encoding colors as [1, 0] for red, [0, 1] for blue, and [0, 0] for green.
# 
# * **Binary Encoding**: This method involves encoding categorical variables into binary vectors, where each category is represented by a unique binary code. Use binary encoding for categorical variables with a small number of categories. For example, encoding colors as 00 for red, 01 for blue, and 10 for green.
# 
# * **Hashing**: This method involves encoding categorical variables into numerical variables using a hash function. Use hashing for categorical variables with a very large number of categories. For example, encoding citie's pincode as hash values.

# I will be using Column `title` for implementation purpose.

# ### Frequency Encoding 
# This involves replacing each category with the frequency or count of that category.\
# Here, we have replaced orginal title with their frequence.
# 
#     Mr              517
#     Miss            182
#     Mrs             125
#     Master           40
#     Dr                7
#     Rev               6
#     Mlle              2
#     Major             2
#     Col               2
#     the Countess      1
#     Capt              1
#     Ms                1
#     Sir               1
#     Lady              1
#     Mme               1
#     Don               1
#     Jonkheer          1
#     
# **One disadvantage of this approach is that when categories have the same frequency, they will be assigned the same weight. This can make it difficult for the model to distinguish between these categories.**

# In[ ]:


df_freq = data.title.value_counts().to_dict()
freq_encod = data["title"].map(df_freq)
data.assign(Frequence_Encoding = freq_encod).head()


# ### Target Encording
# 
# Similar to frequency encoding, mean encoding maps each category to the mean of the target variable, rather than just the count.
# 
# * **Captures Information**: It captures the relationship between the categories and the target variable, potentially providing more predictive features.
# * **Monotonic Relationship**: It creates a monotonic relationship between the categorical variable and the target variable.
# * **Risk of Overfitting**: There is a risk of overfitting the model, as the encoding can lead to high variance if the target variable has many unique values or if the sample size is small.
# 
# More Info at:
# kaggle Tutorial: https://www.kaggle.com/code/ryanholbrook/target-encoding \
# Target class: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html#targetencoder

# In[ ]:


data.groupby(['title'])['Survived'].mean()


# In[ ]:


mean_df = data.groupby(['title'])['Survived'].mean().to_dict()
mean_encord = data['title'].map(mean_df)
data.assign(Mean_Encording = mean_encord).head()


# Or Use **category_encoders** lib

# In[ ]:


import category_encoders as ce

encoder = ce.TargetEncoder(cols=['title'])
df = encoder.fit_transform(data['title'],data['Survived'])

data.assign(target_encoding=df.apply(lambda x: ''.join(str(x[col]) for col in df.columns), axis=1)).head()


# ### Probability Ratio Encoding 
# This method is particularly useful when dealing with binary classification problems, as it helps to encode categories based on their relationship with the target variable.
# 
# **Advantages:**
# * **Captures Target Relationship**: It captures the strength and direction of the relationship between the categorical variable and the target, making it highly informative for predictive modeling.
# * **Prevents Overfitting**: By using probability ratios, it often avoids the pitfalls of overfitting compared to other encoding methods like One-Hot Encoding.
# * **Handles Rare Categories**: Categories that do not provide significant information about the target are naturally assigned low importance, helping the model focus on more predictive features.

# In[ ]:


prob_df = data.groupby(['Cabin Level'])['Survived'].mean()
prob_df = pd.DataFrame(prob_df)
prob_df['died'] = 1 - prob_df['Survived']
prob_ratio = (prob_df['Survived']/prob_df['died']).to_dict()
data.assign(Probability_Ratio_Encoding = data['Cabin Level'].map(prob_ratio)).head()


# ### Label Encoding
# 
# Encode target labels with value between 0 and n_classes-1.
# 
# This transformer should be used to encode target values, i.e. y, and not the input X.\
# Note: **I am using title column for implementation purpose.** As the scikit-learn documentation states, the LabelEncoder should be used to encode target values (y) and not the input features (X). This is because the input features may have different categorical variables that require different encoding schemes. Using the LabelEncoder on the input features could lead to incorrect encoding and affect the performance of the machine learning model.
# 
# Link for sklearn class: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder
# 

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
label_encord = le.fit_transform(data.title)
data.assign(Label_Encording = label_encord).head()


# ### One-Hot Encoding
# Encode categorical features as a one-hot numeric array.
# 
# The input to this transformer should be an array-like of integers or strings, denoting the values taken on by categorical (discrete) features. The features are encoded using a one-hot (aka ‘one-of-K’ or ‘dummy’) encoding scheme.
# 
# sklearn class: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
encoded_array = enc.fit_transform(data[['Sex', 'title']]).toarray()

## OR

pd.get_dummies(data, columns=['title'], prefix='One-Hot').head(5)


# ### Top N One-Hot Encoding 
# 
# By limiting the One-Hot Encoding to the top N categories, you can significantly reduce the dimensionality of the encoded matrix, making it more compact and efficient. This approach is especially useful when handling rare or infrequent categories, such as the "Countess" title in your dataset, which has only a single occurrence.
# 
#     Mr              517
#     Miss            182
#     Mrs             125
#     Master           40
#     Dr                7
#     Rev               6
#     Mlle              2
#     Major             2
#     Col               2
#     the Countess      1
#     Capt              1
#     Ms                1
#     Sir               1
#     Lady              1
#     Mme               1
#     Don               1
#     Jonkheer          1

# In[ ]:


top_N = 4
top_4 = [x for x in data.title.value_counts().sort_values(ascending=False).head(top_N).index]
top_df = data.copy()
for label in top_4:
    top_df[label] = np.where(data["title"]==label,1,0)
top_df.head()


# ### or through the parameters of *OneHotEncoder* in scikit-learn

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore', min_frequency=7, max_categories=5)
encoded_array = enc.fit_transform(data[['Sex', 'title']]).toarray()
enc.categories_, data.iloc[886][['Sex', 'title']], encoded_array[886], enc.inverse_transform([encoded_array[886]]), enc.get_feature_names_out(['Sex', 'title'])


# ### Dummy Encoding
# 
# Both One-Hot encoding and pd.get_dummies have build-in parameter for droping column.
# 
# * ***preprocessing.OneHotEncoder(drop='first').fit(X)***
# * ***pd.get_dummies(data, drop_first=True)***
# 

# In[ ]:


dummy = pd.get_dummies(data['title'], drop_first=False)
dummy.drop(dummy.index)


# In[ ]:


pd.get_dummies(data['title'], drop_first=True).head(5)


# ### Binary Encoding
# 
# Binary encoding is a number system that uses the binary digit, or bit, as the fundamental unit of information. A bit can only be a "0" or a "1". Binary encoding is used in many applications, including: Face recognition, Fingerprint identification, Other classification problems, and Simultaneous data reduction and pattern matching.
# 
# The `category_encoders` library provides an API implementation for various encoding techniques, which are not directly available in pandas or scikit-learn.
# 
# https://contrib.scikit-learn.org/category_encoders/binary.html

# In[ ]:


import category_encoders as ce

encoder = ce.BinaryEncoder(cols=['title'])
df = encoder.fit_transform(data['title'])

data.assign(binary_encoding=df.apply(lambda x: ''.join(str(x[col]) for col in df.columns), axis=1)).head()


# ### Hashing
# 
# This is useful for columns with many categorical values where the categories have no inherent meaning or order. For example, the Ticket column, which can have a large number of unique values without specific order or meaning, is an ideal candidate for hashing. Hashing can reduce dimensionality and efficiently handle high-cardinality features by converting them into a fixed-size representation.
# 
# * **High Cardinality**: Hashing is effective when dealing with categorical variables that have a large number of unique values.
# * **No Inherent Order**: It is especially useful for features where the categories do not have any meaningful ordinal relationship.
# * **Dimensionality Reduction**: Hashing transforms categorical values into a numeric space of fixed size, which helps in managing memory and computational efficiency.

# In[ ]:


import category_encoders as ce

encoder = ce.HashingEncoder(cols=['Ticket'])
df = encoder.fit_transform(data['Ticket'])
data.assign(hashing_encoding=df.apply(lambda x: ''.join(str(x[col]) for col in df.columns), axis=1)).head()


# ## Ordinal Encoding
# 
# * **Basic Ordinal Encoding**: Description: Assigns integer values to categories based on their order. For example, in a feature with categories "Low", "Medium", and "High", you might encode "Low" as 1, "Medium" as 2, and "High" as 3.
# 
# * **Sequeance Encoding**: This method involves converting categorical variables into numerical variables, preserving the order or sequence of the categories. Use sequence encoding when the categories have a natural order or sequence, such as dates or times. For example, encoding months of the year as 1 for January, 2 for February, ..., 12 for December.
# 
# * **Rank Encording**: It assigns numerical values to categories based on their relative order or ranking. This method is specifically designed for ordinal features, where the categories have a meaningful order or hierarchy.For instance, if you have categories like "Poor", "Fair", "Good", and "Excellent", you might assign ranks such that "Poor" is 1, "Fair" is 2, "Good" is 3, and "Excellent" is 4.
# 
# * **Target Guided Ordinal Encoding**: method of encoding categorical variables by ordering the categories based on the target variable and assigning ordinal numbers to them. This approach creates a meaningful relationship between the encoded values and the target variable, making it particularly useful for predictive modeling.
# 
# * **Custom Mapping**: Description: Manually specify the mapping of categories to numerical values based on domain knowledge or other considerations. This approach is useful when you want to control how categories are represented numerically.

# ### Basic Ordinal Encoding
# 
# The Scikit-learn library provides an OrdinalEncoder class for ordinal encoding. You can find the documentation here:https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html
# 
# > The OrdinalEncoder class in Scikit-learn assigns unique integer values to categories based on the order they appear in the data when fitting the encoder. This process essentially follows a "first-come, first-served" approach.
# > The categories parameter allows explicit control over the order of categories.
# 

# In[ ]:


from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(categories='auto', handle_unknown='use_encoded_value',
                         unknown_value=np.nan, encoded_missing_value=-1)

data.assign(Pclass_Ordinal = encoder.fit_transform(data[['Cabin']])).head()


# ### Sequeance Encoding
# 
# This encoding technique is generally used for cyclical features such as dates and days of the week. For implementation purposes, I will apply this technique to the `Age` column. By encoding age in this manner, we can capture patterns that might be useful for modeling.

# In[ ]:


seq_encod = data["Ticket"].rank(method='dense', ascending=False, na_option='top').astype(int).to_dict()
data.assign(Age_sequeance_encoding=seq_encod).tail()


# ### Rank Encoding
# 
# In this dataset, the `Pclass` (passenger class) column is a good example of rank encoding. `Higher classes` are denoted by `1`, and `lower classes` are denoted `3` (Models don't multiply the values, meaning higher numbers don't necessarily imply higher importance; instead, they understand the relationship between features.). This encoding reflects the historical context where passengers in higher classes often had more resources and status, which could have influenced their chances during the rescue operations. Since the `Pclass` column is already encoded in this way, we will repeat this decode for implementation purposes.

# In[ ]:


rank_dict = {
    1: 'Upper class',
    2: 'Middle class',
    3: 'Lower class'
}
data.assign(Pclass = data["Pclass"].map(rank_dict))


# ### Target Guided Ordinal Encoding
# 
# * **Based on Target Variable**: The encoding process uses the target variable (often denoted as y) to guide how the categories are ordered and assigned numerical values.
# * **Creation of Ordinal Relationships**: Unlike standard ordinal encoding, where categories are assigned arbitrary numbers, Target Guided Ordinal Encoding creates a meaningful order based on how each category relates to the target variable. For example, categories that are associated with higher target values are assigned lower ranks and vice versa.
# * **Monotonic Relationship**: This encoding creates a monotonic relationship between the categorical variable and the target variable, which can help improve the performance of machine learning models by capturing inherent patterns in the data.
# 
# ***Steps:***
# * **Compute Target Statistics**: For each category, compute a target statistic, such as the mean of the target variable (e.g., the average target value for each category).
# * **Rank Categories**: Rank the categories based on the computed statistic, typically in ascending or descending order.
# * **Assign Ordinal Values**: Map each category to an ordinal number based on its rank. The category with the lowest target value might be encoded as 1, the next as 2, and so on.

# In[ ]:


mean_df = data.groupby(["title"])["Survived"].mean().sort_values()
mean_df = mean_df.rank(method="dense", na_option="top")
mean_df


# In[ ]:


data.assign(Target_encoding=data['title'].map(mean_df))


# In[ ]:


mean_dict = data.groupby(["title"])["Survived"].mean().sort_values().to_dict()
mean_dict = {k: idx for idx, (k, v) in enumerate(mean_dict.items())}
mean_dict


# In[ ]:


data.assign(Target_Ordinal=data["title"].map(mean_dict))


# ### Custom Mapping
# 
# In this approach, categories are ranked based on domain expertise or specific criteria. For implementation, I will use the `Sex` column, assigning a rank of `1` to `female` and `0` to `male`. This ranking reflects the historical preference given to females and children during rescue operations.

# In[ ]:


cus_dict = {
    "female": 1,
    "male": 0
}
data.assign(Custom_Mapping=data["Sex"].map(cus_dict))


# ### For more detailed explanations about encoding techniques and how to use functions, such as .fit(X) and .categories_, please refer to the Scikit-learn page on preprocessing: https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features Alternatively, you can check out the category_encoders library at https://contrib.scikit-learn.org/category_encoders/, which provides various encoding methods

# # **Transformation of Features: Why Is It Required?**
# 
# Feature transformation is crucial in data preprocessing because it helps optimize the performance of machine learning algorithms. Here’s why feature transformation is needed for various models:
# 
# * **Linear Regression & Gradient Descent (Global Minima):**
#    - **Reason**: Linear Regression and Gradient Descent methods rely on finding the optimal point (Global Minima) where the cost function is minimized. Feature transformation (like scaling) helps the algorithm converge faster and more accurately by ensuring that all features contribute equally.
# 
# 
# * **Algorithms like KNN, K-Means, Hierarchical Clustering (Euclidean Distance):**
#    - **Reason**: These algorithms use distance metrics (like Euclidean Distance) to calculate similarity between data points. If features are on different scales, some features might dominate the distance calculation. Scaling ensures that no single feature disproportionately influences the outcome.
# 
# 
# * **Deep Learning Techniques (Standardization, Scaling):**
#    - **Artificial Neural Networks (ANN)**: Scaling helps ANN achieve better gradient descent convergence and avoids issues like vanishing or exploding gradients.
#    - **Convolutional Neural Networks (CNN)**: Often used for image data where pixel values ranging from 0 to 255 are often normalized to a range of 0 to 1 or standardized to have zero mean and unit variance, improving model learning, scaling helps standardize input values and improves model performance.
#    - **Recurrent Neural Networks (RNN)**: Feature scaling ensures stable and faster training, which is crucial for time-series or sequential data.

# ## **Feature Scaling and Transformation Techniques**
# 
# - **Normalization**: Scales features to a specific range, often [0, 1]. Useful for models that are sensitive to the scale of input features.
#     * **Minimum and Maximum Scaler:** Rescales feature values to lie within a specific range, typically [0, 1]. This is achieved by subtracting the minimum value and then dividing by the range.
#     
#     
# - **Standardization**: Centers features around zero and scales based on standard deviation, making features have a mean of 0 and a standard deviation of 1.
#     * **Robust Scaler:** Scales features based on their median and quantiles. This method is robust to outliers and useful for data with skewed distributions.
# 
# 
# - **Gaussian(Normal) Transformation:** Applies transformations to make data more normally distributed. Useful for algorithms that assume a Gaussian distribution of features. Linear Regression, Logistic Regression, Gaussian Naive Bayes, Linear Discriminant Analysis (LDA), Principal Component Analysis (PCA), K-Nearest Neighbors (KNN) (in some cases, especially with kernel methods).
# 
#     * **Logarithmic Transformation:** Applies a logarithmic function to features to reduce the impact of extreme values and make data more normally distributed.
# 
#     * **Reciprocal Transformation:** Uses the reciprocal of feature values to compress large values and expand smaller values. Useful for data with a wide range of values.
# 
#     * **Square Root Transformation:** Applies the square root function to features to stabilize variance and reduce skewness in the data.
# 
#     * **Exponential Transformation:** Applies an exponential function to features to model data that grows rapidly. It can be used to emphasize differences in smaller values.
# 
#     * **Box-Cox Transformation:** A family of power transformations that can stabilize variance and make the data more normally distributed. The transformation depends on a parameter that is estimated from the data.
# 
# These techniques help preprocess data for machine learning models by adjusting feature scales, reducing skewness, and improving model performance.\
# [Compare the effect of different scalers on data with outliers](https://scikit-earn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#results)

# In[ ]:


columns = ['Pclass','Age','Fare','Survived']
scale_df = data[columns].copy()
scale_df['Age'] = data['Age'].fillna(data.Age.median())
scale_df.head()


# ### Normalization 
# Normalizer class is used to normalize samples (rows) of the dataset rather than individual features (columns). 
# This is the process of scaling individual samples to have unit norm. This process can be useful if you plan to use a quadratic form such as the dot-product or any other kernel to quantify the similarity of any pair of samples.\
# This assumption is the base of the Vector Space Model often used in text classification and clustering contexts.
# 
# * **L1 Norm (Manhattan Norm)**: The sum of the absolute values of the vector components.
# * **L2 Norm (Euclidean Norm)**: The square root of the sum of the squares of the vector components.
# * **Max Norm**: The maximum absolute value among the vector components.

# In[ ]:


from sklearn.preprocessing import Normalizer
normalized_df = Normalizer(norm='l2').fit_transform(scale_df)
normalized_df = pd.DataFrame(normalized_df, columns=columns)
normalized_df.head()


# ### **Min-Max Scaling:**
# 
# Min-Max Scaling is a feature scaling technique that transforms the values of a feature into a specified range, typically between 0 and 1. It adjusts the data values by shifting and rescaling them according to the minimum and maximum values of the feature. This transformation is often used in machine learning to bring different features to the same scale, helping algorithms converge faster and perform better.
# 
# **Formula:**
# 
# X_scaled = (X - X_min)/(X_max - X_min)
# 
# - **( X )**: The original value.
# - **( X_min )**: The minimum value of the feature.
# - **( X_max )**: The maximum value of the feature.
# 
# **How It Works:**
# - Each value of \( X \) is transformed to a value between 0 and 1 by subtracting the minimum value and dividing by the range (difference between maximum and minimum values).
# - This method preserves the shape of the original distribution of the data while scaling it.
# 
# **Example Calculation:**
# * For a value of 20: Scaled value = (20 - 10) / (50 - 10) = 10 / 40 = 0.25
# 
# * For a value of 50: Scaled value = (50 - 10) / (50 - 10) = 40 / 40 = 1
# 
# 
# Min-Max Scaling is commonly used when the data does not have significant outliers, as outliers can heavily impact the scaling process.
# 
# Sklearn link: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
MinMax_scaler = MinMaxScaler()
MinMax_scaled = MinMax_scaler.fit_transform(scale_df)
scaled_df = pd.DataFrame(MinMax_scaled, columns=columns)
scaled_df.head()


# [MaxAbsScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html) works in a very similar fashion, but scales in a way that the training data lies within the range [-1, 1] by dividing through the largest maximum value in each feature. It is meant for data that is already centered at zero or sparse data.

# In[ ]:


from sklearn.preprocessing import MaxAbsScaler
MinMax_scaler = MinMaxScaler()
MinMax_scaled = MinMax_scaler.fit_transform(scale_df)
scaled_df = pd.DataFrame(MinMax_scaled, columns=columns)
scaled_df.head()


# ### **Standardization**
# 
# Standardization is a feature transformation technique where all variables or features are scaled to have a mean of zero and a standard deviation of one. This process ensures that all features contribute equally to the model, preventing features with larger scales from dominating the learning process.
# 
# #### **Formula for Standardization (Z-Score):**
# 
# **z = (x - x_mean)\std**
# 
# - **\(x\)**: The individual data point.
# - **\(x\_mean\)**: The mean of the feature values.
# - **\(std\)**: The standard deviation of the feature values.
# 
# #### **Key Points:**
# - **Purpose**: Standardization centers the data around zero and scales it based on the standard deviation, making it easier for algorithms to learn and find optimal solutions, especially in gradient-based methods.
# - **When to Use**: This method is particularly useful for algorithms that assume data is normally distributed or that are sensitive to the scale of data, such as Linear Regression, Logistic Regression, KNN, and SVM.
# 
# #### **Benefits of Standardization:**
# - Ensures features are on a comparable scale, improving model performance.
# - Speeds up convergence in optimization algorithms, particularly gradient descent.
# - Reduces the risk of features with larger scales dominating the model training process.
# 
# Standardization is a key step in data preprocessing, enhancing the effectiveness and efficiency of many machine learning models.
# 
# Sklearn Link:https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler

# In[ ]:


# We use the Standardscaler from sklearn library
from sklearn.preprocessing import StandardScaler
Stand_Scaler=StandardScaler()
stand_scale=pd.DataFrame(Stand_Scaler.fit_transform(scale_df), columns=columns)
stand_scale.head()


# It is possible to disable either centering or scaling by either passing `with_mean=False` or `with_std=False` to the constructor of `StandardScaler`.

# ### Robust Scaler
# RobustScaler scales features using statistics that are robust to outliers, specifically the median and the interquartile range (IQR).
# 
# Scaling Formula: `𝑋_scaled = (𝑋 − median) / IQR`
#  
# X is the original feature value.\
# median is the median of the feature.\
# IQR (Interquartile Range) is the difference between the 75th percentile (Q3) and the 25th percentile (Q1) of the feature.
# 
# * **Advantages**:
# 
#     * **Robust to Outliers**: By using the median and IQR, RobustScaler is less affected by outliers compared to other scalers like MinMaxScaler or StandardScaler.
#     * **Centering**: It centers the data around the median, which is less sensitive to extreme values compared to the mean.
# 
# 
# * **When to Use RobustScaler**:
#     * Presence of Outliers: If your feature contains outliers and you prefer not to remove them, RobustScaler is suitable because it scales features based on the median and interquartile range (IQR), which are less sensitive to extreme values. In some cases, outliers may carry important information (e.g., age in a dataset where older individuals might be a crucial part of the analysis). Removing these outliers could result in the loss of valuable information. When you want to maintain the integrity of your data and do not want to alter or remove outliers, RobustScaler provides a way to scale your features while minimizing the impact of these extreme values.

# In[ ]:


from sklearn.preprocessing import RobustScaler
robust_scaler=RobustScaler()
df_robust_scaler=pd.DataFrame(robust_scaler.fit_transform(scale_df),columns=columns)
df_robust_scaler.head()


# In[ ]:


import scipy.stats as stat
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import pylab 
#### If you want to check whether feature is guassian or normal distributed
#### Q-Q plot
def plot_data(df,feature):
    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    df[feature].hist()
    plt.subplot(1,2,2)
    stat.probplot(df[feature],dist='norm',plot=pylab)
    plt.show()


# A normal distribution of data is often required for linear models such as Linear Regression and Logistic Regression because these models assume that the input features are normally distributed. Transformations help reduce skewness in the data, making it more symmetric and closer to a normal distribution, which can improve model performance. Here are some common transformation methods used to achieve this:
# 
# - Logarithmic Transformation
# - Square Root Transformation
# - Reciprocal Transformation
# - Box-Cox Transformation
# - Yeo-Johnson Transformation
# 
# These methods help in stabilizing variance and enhancing the normality of the features, making them better suited for linear modeling.

# ### Logarithmic Transformation
# We take log of each value

# In[ ]:


scale_df['Age_log'] = np.log(scale_df['Age'])
plot_data(scale_df,'Age_log')


# ### Reciprocal Trnasformation

# In[ ]:


scale_df['Age_reciprocal']=1/scale_df.Age
plot_data(scale_df,'Age_reciprocal')


# ### Square Root Transformation

# In[ ]:


scale_df['Age_sqaure']=scale_df.Age**(1/2)
plot_data(scale_df,'Age_sqaure')


# ### Exponential Transdormation

# In[ ]:


scale_df['Age_exponential']=scale_df.Age**(1/1.2)
plot_data(scale_df,'Age_exponential')


# ### Box-Cox Transformation
# The Box-Cox transformation is a statistical technique used to stabilize variance and make data more closely approximate a normal distribution. This transformation is particularly useful when data exhibits non-normality, heteroscedasticity, or skewness, which can negatively impact statistical modeling.
# 
# The Box-Cox transformation is defined by the formula:
# T(Y)=(Y exp(λ)−1)/λ
# 
#     where Y is the response variable and λ is the transformation parameter. λ varies from -5 to 5. In the transformation, all values of λ are considered and the optimal value for a given variable is selected.
#     
# **Box-Cox requires the input data to be strictly positive; negative or zero values will cause errors.**
# 

# In[ ]:


from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='box-cox') ##with method='yeo-johnson' is a modification of the Box-Cox transformation that works for both positive and negative values.
df = pt.fit_transform(scale_df[["Age"]])
df = pd.DataFrame(df, columns=["Age"])
plot_data(df, 'Age')


# # Acknowledgments:
# 
# **The materials and insights in this notebook are compiled from various sources, including educational content by Krish Naik, official Scikit-learn documentation, and guidance from ChatGPT.**
