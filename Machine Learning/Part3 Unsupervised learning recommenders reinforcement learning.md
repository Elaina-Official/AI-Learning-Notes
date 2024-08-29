# Unsupervised Learning, Recommender Systems and Reinforcement Learning

## Week 1

### Clustering

A clustering algorithm looks at the number of data points and automatically finds data points that are related or similar to each other.

#### K-means Algorithm

K-Means clustering is a popular unsupervised learning algorithm used for partitioning a dataset into K distinct, non-overlapping subsets (clusters). The goal of K-Means is to **organize data points into clusters** such that points within the same cluster are more similar to each other than to those in other clusters. 

##### Process

The basic steps of the K-Means algorithm are as follows:

- Initialize centroids: Randomly select K points as the initial centroids

- Assign each point to its closest centroid: Each data point is assigned to the nearest centroid based on a distance metric, usually the Euclidean distance
- Recompute the centroids: For each cluster, compute the mean of all data points assigned to it, and update the centroid to this mean value
- Repeat the assignment and update steps

If a cluster has zero training examples assigned to it, the third step would be trying to compute the average of zero points, and that's not well defined. The most common thing to do is to eliminate the cluster or randomly reinitialize the cluster centroid. 

##### Loss Function

In K-means algorithm, we use distortion function as loss function. 
$$
J(c^{(1)}, \cdots, c^{(m)}, \mu_1, \cdots, \mu_k) = \frac{1}{m}\sum_{i=1}^{m}||x^{(i)}-\mu_c(i)||^2
$$

- $c^{(i)}$ = index of cluster $(1, 2, \cdots, K)$ to which example $x^{(i)}$ is currently assigned
- $\mu_k$ = cluster centroid $k$
- $\mu_c(i)$ = cluster centroid of cluster to which example $x^{(i)}$ has been assigned

By minimizing the distortion function, K-Means is able to generate cluster divisions that cluster well.

##### Initialize K-means

Selecting the initialization centroid for the K-Means algorithm is an important step in the quality of clustering. The choice of the centroid of mass has a large impact on the final clustering results, as the K-Means algorithm is susceptible to the position of the initial centroid of mass, which may cause the algorithm to fall into a local optimum solution.

- Random initialization

  Random initialization is the simplest initialization method for K-means algorithm. It randomly choose $K$ training examples as the initial centroid. It is easy to implement, but prone to poor cluster results. 

##### Choosing the Number of Clusters

We usually find the point at which the decrease in the distortion function decreases significantly as $K$ continues to increase. The value of $K$ corresponding to this point is usually chosen as the optimal number of clusters because it provides a better balance between clustering effectiveness and computational complexity.

But for some situations, decrease of the distortion function is very slow, and this method does not perform well. So we can evaluate $K$ based on how well it performs for that downstream purpose. 
