#!/usr/bin/env python
# coding: utf-8

# # Task1-Iris flower classification ML project

# # Imporing the libraries

# In[68]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from pandas.plotting import scatter_matrix
from sklearn.datasets import load_iris


# # Loading the dataset 

# In[55]:


head=['sepal length','sepal width','petal length','petal width','species']
iris=pd.read_csv("iris.csv")
print(iris)


# # Exploring the dataset

# In[31]:


iris.head()


# In[32]:


iris.tail()


# # Find the size of dataset 

# In[33]:


iris.shape


# In[34]:


iris.dtypes


# # Checking for null values in dataset

# In[35]:


iris.isnull()


# In[36]:


iris.isnull().sum()


# # Correlation between the dataset

# In[37]:


plt.figure(figsize=(13,12))
sns.heatmap(iris.corr(), annot=True, cmap='cubehelix_r')


# # Exploratory data anaysis

# In[38]:


iris.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False)
plt.show()


# # Data Visualisation

# In[39]:


iris.hist()
plt.show()


# # Pair plot of data

# In[40]:


from pandas.plotting import scatter_matrix
scatter_matrix(iris)
plt.show()


# # Scatter plot data visualisation

# In[41]:


iris = load_iris()
features = iris.data.T


# In[42]:


sepal_length_label=iris.feature_names[0]
sepal_width_label=iris.feature_names[1]
petal_length_label=iris.feature_names[2]
petal_width_label=iris.feature_names[3]


# In[43]:


sepal_length = features[0]
sepal_width = features[1]
petal_length = features[2]sepal_length_label=iris.feature_names[0]
sepal_width_label=iris.feature_names[1]
petal_length_label=iris.feature_names[2]
petal_width_label=iris.feature_names[3]
petal_width = features[3]


# In[44]:


plt.scatter(sepal_length, sepal_width, c=iris.target)
plt.xlabel(sepal_length_label)
plt.ylabel(sepal_width_label)
plt.show()


# In[45]:


plt.scatter(petal_length, petal_width, c=iris.target)
plt.xlabel(petal_length_label)
plt.ylabel(petal_width_label)
plt.show()


# # Data prepration

# In[60]:


x=iris.iloc[:,0:4]
x.head()


# In[69]:


y=iris['species'].to_frame
y


# In[70]:


from sklearn import preprocessing
std = preprocessing.StandardScaler()
x = std.fit_transform(x)
x[0:4]

