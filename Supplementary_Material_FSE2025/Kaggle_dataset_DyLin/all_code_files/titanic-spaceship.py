#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df=pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')
df


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.describe()


# In[ ]:


df['Age'].value_counts()
    


# In[ ]:


df.hist(bins=20,figsize=(10,5),color='brown')


# In[ ]:


# Melting the DataFrame
test_df_melted = df.melt(id_vars='HomePlanet', value_vars='VIP', var_name='Variable', value_name='Value')

# Create the countplot using the melted DataFrame
sns.countplot(x='Value', hue='HomePlanet', data=test_df_melted, palette='dark')

# Set labels and title
plt.xlabel('VIP Status')
plt.ylabel('Count')
plt.title('Count Plot of VIP Status by Home Planet')
plt.show()


# In[ ]:



plt.figure(figsize=(7,5))

plt.pie([25, 30, 45],labels=('Age', 'Spa', 'VRDeck'),explode=[0.1, 0.1, 0.1],autopct='%1.1f%%',shadow=True,labeldistance=1.1) 

plt.title('Titanic')

plt.show()


# In[ ]:


# Select only the numerical columns
numeric_data = df.select_dtypes(include='number')

# Plotting the heatmap with correlation values
plt.figure(figsize=(11, 7))
sns.heatmap(numeric_data.corr(), annot=True, cmap='Blues')

# Display the plot
plt.show()


# In[ ]:


plt.figure(figsize=(7,5))

sns.scatterplot(x='Age', y='PassengerId', data=df)

plt.ylabel('Frequency', color='r')

plt.show()


# In[ ]:


# Set the figure size to 7x5 inches for the plot
plt.figure(figsize=(7,5))

# Plot a histogram of the 'age' column in the DataFrame 'df' with a cyan color
df['Age'].plot(kind='hist', color='c')

# Label the x-axis as 'Age' and set the label color to red
plt.xlabel('Age', color='r')

# Label the y-axis as 'Frequency' and set the label color to red
plt.ylabel('Frequency', color='r')

# Set the title of the plot and color it red
plt.title('The Age Number Of Data', color='r')

# Display the plot
plt.show()


# In[ ]:


plt.figure(figsize=(7, 5))

# Plotting the 'children' column of the dataframe with different styles and colors

# First plot: solid red line
df['Age'].plot(color='r', linestyle='solid')

# Second plot: dashed black line
df['Age'].plot(color='k', linestyle='dashed')

# Third plot: dashdot yellow line
df['Age'].plot(color='y', linestyle='dashdot')

# Fourth plot: dotted cyan line
df['Age'].plot(color='r', linestyle='dotted')

# Adding labels to the x and y axes
plt.xlabel('children')  # Label for x-axis
plt.ylabel('Frequency')  # Label for y-axis


# In[ ]:


x=df.Age
y=df.RoomService
w=df.Age
z=df.FoodCourt

plt.scatter(x, y, s=w, c=z)

plt.colorbar()

# Displaying the plot
plt.show()


# In[ ]:


def f(x, y):
    e = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)  # Compute the function value using sine and cosine
    return e

x = np.linspace(0, 5, 50)  
y = np.linspace(0, 5, 40)  

# Create a meshgrid of x and y values
X, Y = np.meshgrid(x, y)

# Compute Z values using the function f
Z = f(X, Y)

# Create a filled contour plot of the function with 20 contour levels and a 'Blues' color map
plt.contourf(X, Y, Z, 20, cmap='hot')

plt.colorbar()


plt.show()


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure() #adjust 3D figure
ax = fig.add_subplot(111, projection='3d')
# X and Y values from DataFrame
X = df['RoomService']
Y = df['FoodCourt']
X,Y = np.meshgrid(X,Y) # combine them
R = np.sqrt(X**2 + Y**2) # function
Z = np.cos(R) # Z Value

ax.plot_surface(X,Y,Z,cmap='Greens') # Draw them
plt.show()


# In[ ]:


# Remove NaN values from 'Age' and 'VRDeck'
df_clean = df[['Age', 'VRDeck']].dropna()

a = df_clean['Age']
b = df_clean['VRDeck']

# Create the 2D histogram
plt.hist2d(a, b, cmap='Blues', bins=10)

plt.colorbar()

# Adding labels for clarity
plt.xlabel('Age')
plt.ylabel('VRDeck')
plt.title('2D Histogram of Age and VRDeck')

# Display the plot
plt.show()

