#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


datat = pd.read_csv("/kaggle/input/titanic/train.csv")  #datat = data titanic
datat


# In[ ]:


datat.columns #name of all columns


# In[ ]:


datat.describe() #statistical data of data titanic such as mean


# In[ ]:


age_np = np.array(datat["Age"])
fare_np = np.array(datat["Fare"])
age_np = age_np[~np.isnan(age_np)] #erasing the NaNs
fare_np = fare_np[~np.isnan(fare_np)] #erasing the NaNs
age_np[:10] #printing first 10


# In[ ]:


agep5 = np.array(datat["Age"]+5)   #adding 5 to ages - agep5 age plus 5
faret2 = np.array(datat["Fare"] * 2) #multiplying with 2 with ticket prices - faret2 fare times 2
agep5[:10], faret2[:10]


# In[ ]:


datat.info() #getting information about data


# In[ ]:


datat.head(5)


# In[ ]:


datat.isnull().sum()


# In[ ]:


datat.dropna(axis=0, inplace=True)
datat  #dropping  NaNs in the columns Age, Cabin, Embarked


# In[ ]:


datat.isnull().sum()


# In[ ]:


datat[(datat.Survived == 1) & (datat.Sex == "female")] #Only FEMALE survivors


# In[ ]:


datat[(datat.Survived == 1) & (datat.Sex == "male")] #Only MALE survivors


# In[ ]:


datat[(datat.Fare >50)] #only ticket prices more than 50, passengers


# In[ ]:


datat.groupby("Survived")[["Age", "Fare"]].mean() #comparing survivors with their ages & ticket prices


# In[ ]:


datat.groupby("Sex")[["Survived"]].mean() #comparing survivors females and males 


# In[ ]:


datat["FareTaxes"] = datat["Fare"] * 0.10 #adding new column about tciket prices' tax
datat


# In[ ]:


#adding new column about Adultness
datat["Adultness"] = datat["Age"].apply(lambda x: 
                                        "Adult" if x >= 18
                                        else "Child")

datat


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
palette = sns.color_palette("Spectral") #changing colors on the plots -spectral is one of he seaborn palettes


# In[ ]:


#Line Chart
sns.set_palette(palette, n_colors=None, desat=None, color_codes=False)
sns.lineplot(x=datat["Age"], y=datat["Fare"])
plt.title('Ticket Prices Sort by Age')
plt.grid(True)


# In[ ]:


#Scatter Plot
sns.set_palette(palette, n_colors=None, desat=None, color_codes=False)
sns.scatterplot(data= datat, x="Age", y="Fare", hue="Sex")
sns.color_palette("flare")
plt.title('Ticket Prices Sort by Age and Sex')


# In[ ]:


#Histogram Plot showing Ages
sns.set_palette(palette, n_colors=None, desat=None, color_codes=False)
sns.histplot(data=datat, x="Age", kde=True)


# In[ ]:


#Histogram Plot showing Ticket Prices
sns.set_palette(palette, n_colors=None, desat=None, color_codes=False)
sns.histplot(data=datat, x="Fare", kde=True)


# In[ ]:


#Bar Plot comparing survivings by sex
sns.set_palette(palette, n_colors=None, desat=None, color_codes=False)
sns.barplot(datat, x="Sex", y="Survived", hue="Survived" , 
           estimator=lambda x: len(x))


# In[ ]:


#Three Plots in One 
sns.set_palette(palette, n_colors=None, desat=None, color_codes=False)
fig, (ax1, ax2, ax3)= plt.subplots(3, 1, figsize=(8, 6))

sns.lineplot(x=datat["Age"], y=datat["Fare"], ax=ax1)
ax1.set_title('Ages sort by Ticket Prices')

sns.histplot(datat["Age"], bins=30, ax=ax2)
ax2.set_title('Age Histogram')

sns.barplot(datat, x="Sex", y="Survived", hue="Survived" , 
           estimator=lambda x: len(x), ax=ax3)
ax3.set_title('Surviving sort by Sex')



plt.tight_layout()

plt.show()


# In[ ]:


#Heatmap

