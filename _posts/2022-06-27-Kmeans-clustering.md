---
title: "Post: Kmeans clustering"
categories:
  - Blog
tags:
  - machine learning
  - algorithms
  - AI
classes: 
  - wide
toc:
  - True
---

# 1. Introduction

K-means is one of the most popular, unsupervised clustering methods. The algorithm is used to group data into different clusters, where each datapoint is assigned to its nearest cluster centroid. K-means has many applications, like e.g. language clustering, image segmentation or anomaly detection.


# 2 How it works...

Clustering splits N samples $$x_1, ... , x_N$$ into K disjoint sets or clusters using the following steps:

- *Choose a value for K, the number of clusters.*
- *Randomly initialize K cluster centroids.*
- *Assign each datapoint in the dataset to its nearest cluster centroid based on the Euclidean distance.*

The Euclidean distance between two points is given by:

$$d(p,q) =\sqrt{(q_1 - p_1)^2 + (q_2 - p_2)^2} $$

To assign a datapoint $$x$$ to a new cluster, we use:

$$\operatorname*{arg\,min}_{c_i \in C} \quad dist(c_i, x)^2$$

where $$c_i$$ is the centroid of cluster $$i$$ and $$dist()$$ the euclidean distance function. Note that the euclidean distance function can be replaced by a different one, like e.g. the Manhatten distance.

- *Compute new cluster centroids using the mean of all datapoints assigned to a specific cluster.*

To compute a new centrod $$c_i$$, we use:

$$c_i =\frac{1}{S_i}\sum_{x_i\in S_i } x_i $$

where $$S_i$$ is the set of all points assigned to cluster $$i$$.

- *Repeat step 3-4 until a pre-defined stopping criteria is met.*


## 2.1 Cluster centroid initialization

There are several different techniques to initialize the centroids of the k-means algorithm:

- random datapoints
- k-means ++
- native sharding

In this tutorial we will use random initialization.

## 2.2 Stopping criteria

- Cluster centroids do not change anymore.
- The sum of the distances is minimized.
- A maximum number of iterations is reached.

## 2.3 Chosing K

A common method for choosing the number of clusters K is called the "Elbow method". Here we run the K-means algorithm several times with different values for K and plot K against the Within Cluster Sum of Squares (WCSS) error, or sometimes called distortion. The best value of K is where the plot leads to a sudden drop in the WCSS error.

<figure>
	<img src="/assets/images/post_5/elbow.png" style="width: 600px;">
	<figcaption>Figure 1. Choosing the number of K clusters using the elbow method.</figcaption>
</figure>

## 2.4 Data preparation before cluster analysis

The data we use should be prepared before clustering. One reason for this is that data might have different units that aren't comparible. In addition, even if variables have the same unit, they might have a totally different scale.<br>
Different scales and units might introduce bias into the k-means cluster algorithm. In example, clusters might be heavily dependent on one variable.
To avoid this, we *normalize* the data. Often a process called z-score normalization is used, where we subtract the mean from a datapoint and divide by the standard deviation:

$$Z = \frac{x -\mu }{\sigma} $$

## 2.5 Limitations of k-means clustering

- **How to find the right number of clusters?** We saw that the elbow method is one way to determine the number of clusters. Unfortunately, the elbow method does not always work. E.g., when the data is evenly distributed, the elbow method fails.
- **Impact of seeds:** changing the seed of the algorithm, might lead too different clusters.
- **Biased towards equal sized clusters:** Because we try to minimize distortion, k-means clustering might result in non-intuitive clusters. The clustering result are clusters with similar areas but not necessarily the same number of datapoints.
- **Sensitive to outliers:** Centroids can be dragged in the wrong direction due to outliers (or noise) in the dataset.

# 3. K-means in Python 

One advantage with the K-means algorithm is that it is easy to imlpement in Python. As an example we will use the Iris dataset. <br />
Lets import the dataset from the scikit-image library.

```python
from sklearn.datasets import load_iris

dataset = load_iris()
X = dataset.data
y = dataset.target
```

Now we can define a function that randomly initializes the cluster centroids. If we want to be sure that the randomly chosen center points lie within the range of our dataset, we can make use of the datasets mean and standard deviation.

```python
def init_clusters():

    mean = np.mean(self.X, axis = 0)
    std = np.std(self.X, axis = 0)
    centers = np.random.randn(self.K,self.X.shape[1])*std + mean

    return centers

```

Lets implement the rest of the algorithm...
<br> 
The Iris dataset has three different classes, so a good choice for K might be K = 3.
To compute the distance between datapoints and the cluster centers we will use the Euclidean distance.

```python
import numpy as np
from copy import  deepcopy


N = self.X.shape[0]
K = 3

centers = init_clusters()

centers_old = np.zeros(centers.shape) # to store old centers

centers_new = deepcopy(centers) # Store new centers

clusters = np.zeros(N)
distances = np.zeros((N,K))

error = np.linalg.norm(centers_new - centers_old)

# When, after an update, the estimate of that center stays the same, exit loop
while error != 0:
    
    # Measure the distance to every center
    for i in range(K):
        distances[:,i] = np.linalg.norm(X - centers_new[i], axis=1)
    
    # Assign all training data to closest center
    clusters = np.argmin(distances, axis = 1)
    
    centers_old = deepcopy(centers_new)
    # Calculate mean for every cluster and update the center
    for i in range(K):
        centers_new[i] = np.mean(X[clusters == i], axis=0)
    error = np.linalg.norm(centers_new - centers_old)

```

Finally we can plot the result using a scatter plot.

```python
import matplotlib.pyplot as plt
# Plot the data
colors=['orange', 'blue', 'green']
for i in range(N):
    plt.scatter(X[i, 0], X[i,1], s=7, color = colors[int(y[i])])
plt.scatter(centers_new[:,0], centers_new[:,1], marker='*', c='g', s=150)        

```

<figure>
	<img src="/assets/images/post_5/kmeans.png" style="width: 600px;">
	<figcaption>Figure 2. Result of the K-means algorithm.</figcaption>
</figure>

