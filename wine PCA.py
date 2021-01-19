# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 16:49:46 2021

@author: Shubham
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
wine = pd.read_csv('D:\\Data Science study\\Documents\\Assignments\\PCA\\wine.csv')
wine
wine.describe()
wine.head()
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
wine.data = wine.iloc[:,1:]
wine.data.head()

# Normalizing the data using the inbuilt function
wine_norm = scale(wine.data)


# Now creating PCA's with given data
pca = PCA(n_components = 13)
pca_values = pca.fit_transform(wine_norm)

# The amount of variance each pca explaains is 
var = pca.explained_variance_ratio_
var
pca.components_[0]

#Cumuative variance

var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1

#Variance plot for pca components is obtained as
plt.plot(var1, color = "red")

# Let's plot the plot between PCA1 and PCA2

x = pca_values[:,0]
y = pca_values[:,1]
plt.plot(x,y,"bo")

# From the graph we can see that there is no relationaship between the PCA1 and PCA2 at all

#####################Clustering#####################

# Now we will use clustering on the pca

# First we will use the Hierarchical clustering 

# We have been asked to create the clusters based on the only first 3 principle components scores

wine_pca = pd.DataFrame(pca_values[:,:3])
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z = linkage(wine_pca,method='complete', metric = 'euclidean')
sch.dendrogram(
        z,
        leaf_rotation = 0.,
        leaf_font_size = 8.
)

# From the dendrogram we can say that 4 clusters will be optimum for our data

# Now lets use the agglomerative clustering
from sklearn.cluster import AgglomerativeClustering

h_complete_pca = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'complete').fit(wine_pca)

h_complete_pca.labels_

# Converting h_complete_pca.labels into series

cluster_labels_pca = pd.Series(h_complete_pca.labels_)

# Creating the new  column clust and assigning the labels to it
h_wine_pca = wine.copy()

h_wine_pca['clust'] = cluster_labels_pca

h_wine_pca

# Let's shift the position of the clust column to initial

h_wine_pca = h_wine_pca.iloc[:,[14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]]

h_result_pca = h_wine_pca.groupby(h_wine_pca.clust).mean()

h_result_pca

# Creating final csv file of the dataframe

import os        # Importing os

os.getcwd()   # getting current working directory

os.chdir("D:\\Data Science study\\assignment\\Sent\\8")   # Changing current working directory

h_wine_pca.to_csv("h_wine_pca", index = False)

# Now we will use kmeans clustering
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
# First create skree plot or elbow curve plot for the desicion of the K value

k = list(range(2,15))
k
TWSSPCA = [] # For storing the total within sum squared value

# We are creating the function for deciding the K value

type(wine_pca)

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(wine_pca)
    WSSPCA = []  # variable for storing within sum of squared values of clusters 
    for j in range (i):
        WSSPCA.append(sum(cdist(wine_pca.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,wine_pca.shape[1]),"euclidean")))
    TWSSPCA.append(sum(WSSPCA))

# plotting scree plot 

plt.plot(k,TWSSPCA,'ro-');plt.xlabel("No. of Clusters");plt.ylabel("Total within SS");plt.xticks(k)

# According to the plot 3 clusters should be optimum

pca_model = KMeans(n_clusters = 3)  # Creating model 

#  But we need to take only first three principle component scores 
# Hence we will use wine_pca which we have created earlier 

pca_model.fit(wine_pca)

pca_model.labels_   # Getting labels 

labels = pd.Series(pca_model.labels_) # converting labels into series
k_wine_pca = wine.copy()   #creating new dataframe

k_wine_pca['clust'] = labels   # assigning clust column to the k_wine with labels values 

k_wine_pca.head()

k_wine_pca = k_wine_pca.iloc[:,[14,1,2,3,4,5,6,7,8,9,10,11,12,13]] #shifting clust to the position of the 1st column

k_wine_pca.iloc[:,1:14].groupby(k_wine_pca.clust).mean()  # Taking clusterwise mean of all the columns

# Creating csv file 
k_wine_pca.to_csv("k_wine_pca", index  = False)

##############################-------------------------------------------------#####################################################

# Now we will also apply clustering on the same datasets without PCA and see if there are any major differences.

# First we will use the Hierarchical clustering 

z = linkage(wine_norm,method='complete', metric = 'euclidean')
sch.dendrogram(
        z,
        leaf_rotation = 0.,
        leaf_font_size = 8.
)

# From the dendrogram we can say that 5 clusters will be optimum for our data

# Now lets use the agglomerative clustering

h_complete = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'complete').fit(wine_norm)

h_complete.labels_

# Converting h_complete.labels into series

cluster_labels = pd.Series(h_complete.labels_)

# Creating the new  column clust and assigning the labels to it

h_wine = wine.copy()

h_wine['clust'] = cluster_labels

h_wine

# Let's shift the position of the clust column to initial

h_wine = h_wine.iloc[:,[14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]]

h_result = h_wine.groupby(h_wine.clust).mean()

h_result

# Creating final csv file of the dataframe

h_wine.to_csv("h_wine", index = False)

# Now we will use kmeans clustering

# First create skree plot or elbow curve plot for the desicion of the K value

k = list(range(2,15))
k
TWSS = [] # For storing the total within sum squared value

# We are creating the function for deciding the K value

type(wine_norm)

# In this case the pca_values is np.array format which does not support following function
# Hence we will convert it into dataframe 

dataframe = pd.DataFrame(wine_norm)

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(dataframe)
    WSS = []  # variable for storing within sum of squared values of clusters 
    for j in range (i):
        WSS.append(sum(cdist(dataframe.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,dataframe.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))

# plotting scree plot 

plt.plot(k,TWSS,'ro-');plt.xlabel("No. of Clusters");plt.ylabel("Total within SS");plt.xticks(k)

# According to the plot 3 clusters should be optimum

model = KMeans(n_clusters = 3)  # Creating model 

#  But we need to take only first three principle component scores 
# Hence we will use wine_pca which we have created earlier 

model.fit(wine_norm)

model.labels_   # Getting labels 

labels = pd.Series(model.labels_) # converting labels into series
k_wine = wine.copy()   #creating new dataframe

k_wine['clust'] = labels   # assigning clust column to the k_wine with labels values 

k_wine.head()

k_wine = k_wine.iloc[:,[14,1,2,3,4,5,6,7,8,9,10,11,12,13]] #shifting clust to the position of the 1st column

k_wine.iloc[:,1:14].groupby(k_wine.clust).mean()  # Taking clusterwise mean of all the columns

# Creating csv file 
k_wine.to_csv("k_wine", index  = False)


# we can conclude that only heirerchical clustering creates variations but KMeans clustering isn't affected.