---
title: "Post: Hierarchical clustering"
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

Hierarchical clustering is an unsupervised algorithm of cluster analysis, used to group similar datapoints into clusters. 
<br>
<br>
In general, we can divide hierarchical clustering algorithms into two different types:

- agglomerative: also called for *"bottom-up"* approach. In agglomerative hierarchical clustering, each datapoint starts in its own cluster. Pairs of clusters are then merged while moving up the hierarchy. 
- divisive: *"top-down"* approach. In divisive hierarchical clustering, all observations start in a single cluster. The cluster is split as we move down the hierarchy.

Like k-means clustering, hierarchical clustering uses unlabeled datapoints. The clustering hierarchy is developed using a so called **dendrogram**, which can be seen as a tree-shaped structure, showing the relationship between datapoints.
<br>
<br>
One huge advantage of hierarchical clustering when compared to k-means clustering, is that we don't have to decide the number of clusters in advance. The number of clusters can be choosen using the dendrogram and a *threshold* value. 

# 2 How it works...

To show how hierarchical clustering works, we will focus on the agglomerative method in this tutorial. That means, we start with single datapoints, merge those that are close together, until we end up with a single cluster.
<br>
<br>
The algorithm can be summarized as follows:

- Generate $$N$$ clusters, where $$N$$ is the number of datapoints.
- Take the two closest datapoints (clusters) and merge them so that we end up with $$N-1$$ clusters.
- Repeat this previous step (merging the nearest two clusters), until only a single cluster is left.
- Generate a dendrogram.
- Choose an appropriate threshold value.
- Extract cluster labels.

## 2.1 Measure cluster distances

To measure the distance between two clusters, hierarchical clustering uses so called **linkage methods**. There are several different ones, each leading to a different dendogram and with that a different clustering outcome:

- **Single linkage:** uses the shortest distance between the closest points of clusters computed using a distance metric (e.g. the Euclidean distance). Single linkage is useful to separate shapes between two clusters that are non-elliptical. Nevertheless, if there is noise between the two clusters, single linkage will fail.

<figure>
	<img src="/assets/images/post_6/single_linkage.PNG" style="width: 650px;">
	<figcaption>Figure 1. Single linkage in hierarchical clustering..</figcaption>
</figure>

- **Complete linkage:** computes the longest distance between two points of two different clusters. This method usually leads to closer clusters than the single linkage method and separates two clusters even if there is noise between them. A disadvantage with complete linkage is that large clusters can be broken and the method might be biased towards large clusters.

<figure>
	<img src="/assets/images/post_6/max_linkage.PNG" style="width: 600px;">
	<figcaption>Figure 2. Complete linkage that uses the longest distance between two points of two clusters.</figcaption>
</figure>

- **Centroid linkage:** uses the distance between the centroid of two clusters.

<figure>
	<img src="/assets/images/post_6/center_linkage.PNG" style="width: 600px;">
	<figcaption>Figure 3. Linkage using the centroids of two clusters.</figcaption>
</figure>


- **Average linkage:** computes the average of the similarity of all pairs of points. Average linking can be biased toawards large clusters but does well even when there is noise between them.

<figure>
	<img src="/assets/images/post_6/average_linking.PNG" style="width: 600px;">
	<figcaption>Figure 4. Average linkage in hierarchical clustering.</figcaption>
</figure>

- **Ward linkage:** based on the sum of squares. Wards method does, like average linking, well when there is noise between the clusters, but might also be biased against larger clusters.


## 2.2 Dendrogram

The dendrogram is a tree-shaped structure that stores every step of the hierarchical clustering process. The *x-axis* of the dendrogram shows the datapoints, while the *y-axis* indicates the distance between them.
<br>
<br>
The dendrogram is used to determine the number of clusters using a threshold value. 


## 2.3 Data preparation before cluster analysis

The data we use should be prepared before clustering. One reason for this is that data might have different units that aren't comparible. In addition, even if variables have the same unit, they might have a totally different scale.<br>
Different scales and units might introduce bias into the k-means cluster algorithm. In example, clusters might be heavily dependent on one variable.
To avoid this, we *normalize* the data. Often a process called z-score normalization is used, where we subtract the mean from a datapoint and divide by the standard deviation:

$$Z = \frac{x -\mu }{\sigma} $$

An alternative is called whitening, where the normalized data is given by:

$$Z = \frac{x}{\sigma} $$

## 2.4 Limitations of hierarchical clustering

- **Not feasible for large datasets:** The runtime of a hierarchical cluster algorihm increases with the number of datapoints. For large datasets, hierarchical clustering is therfore not feasible.

# 3. Hierarchical clustering in Python 

The dataset we will use in this tutorial comprises kernels belonging to three different varieties of wheat: *Kama*, *Rosa* and *Canadian*, 70 elements each, randomly selected for the experiment. High quality visualization of the internal kernel structure was detected using a soft X-ray technique. The data set can be used for the tasks of classification and cluster analysis.
<br>
Seven different geometric parameters of wheat kernels were measured:

1. area $$A$$,
2. perimeter $$P$$,
3. compactness $$C =\frac{4\pi A}{P^2}$$
4. length of kernel,
5. width of kernel,
6. asymmetry coefficient
7. length of kernel groove.

We will implement hierarchical clustering using the scipy python library.
<br>
Lets start by importing the needed packages for both the hierarchical clustering algorithm as well as data visualization.

```python
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import pandas as pd
```

Now we read the dataset using pandas.

```python
seeds = pd.read_csv("seeds.csv", header = None)
seeds.columns = ['Area', 'Perimeter', 'Compactness', 'Kernel length', 'Kernel width', 'Assymetry coefficient', 'Kernel groove length', 'Label']
seeds['Label'] = seeds['Label'].map({1:'Kama wheat', 2:'Rosa wheat', 3:'Canadian wheat'})
seeds.head()
```
<figure>
	<img src="/assets/images/post_6/dataset.PNG" style="width: 800px;">
	<figcaption>Figure 5. The first entries of the grain dataset.</figcaption>
</figure>


```python
#extract data and labels
samples = seeds.iloc[:, :-1].values
varieties = seeds.iloc[:, -1].values
```

Now that our data is in the right format, we can use the hierarchical clustering algorithm from the scipy package. We will use the *complete linkage* or *max linkage* method here.

```python
# Calculate the linkage: mergings
mergings = linkage(samples, method = 'complete')
```

Finally, we create a dendrogram...

```python
plt.figure(figsize=(20, 10))

# Plot the dendrogram, using varieties as labels
dendrogram(mergings,
           labels=varieties,
           leaf_rotation=90,
           leaf_font_size=7,
)
plt.show()
```

<figure>
	<img src="/assets/images/post_6/dendrogram.PNG" style="width: 800px;">
	<figcaption>Figure 6. The dendrogram produced by the hierarchical clustering algorithm.</figcaption>
</figure>

Using the *fcluster()* function, we can exract the labels of the hierarchical clustering algorithm. The only thing we need to do is to choose an appropriate threshold. 
<br>
From the dendogram we can see that $$t = 8$$ might be a good value.

```python
labels = fcluster(mergings, 8, criterion='distance')
```

Lets create a scatterplot of the kernel length feature vs. the kernel widdth feature.

```python
plt.figure(figsize=(5,5))
plt.scatter(samples[:, 3],samples[:,4],c=labels)
```


<figure>
	<img src="/assets/images/post_6/scatterplot.PNG" style="width: 800px;">
	<figcaption>Figure 7. Scatterplot showing the kernel length plotted against the kernel width.</figcaption>
</figure>

From the above scatterplot, we can see that we were able to separate our data into 3 distinct clusters.
<br>

Finally, extract the cluster labels for this intermediate clustering, and compare the labels with the grain varieties using a cross-tabulation.

```python
# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
pd.crosstab(df['labels'], df['varieties'])
```

<figure>
	<img src="/assets/images/post_6/crosstab.PNG" style="width: 400px;">
	<figcaption>Figure 8. Cross-tabulation analysis to compare the clustering result with the original (*ground truth*) labels.</figcaption>
</figure>


A jupyter notebook with the complete code can be found here :[Hierarchical clustering Jupyter Notebook](https://github.com/MLRadfys/mlradfys.github.io/blob/main/assets/post_6/hierarchical_clustering.ipynb) 