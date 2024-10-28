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
J(c^{(1)}, \cdots, c^{(m)}, \mu_1, \cdots, \mu_k) = \frac{1}{m}\sum_{i=1}^{m}\vert\vert x^{(i)}-\mu_c(i)\vert\vert^2
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

### Finding unusual events

#### Anomaly Detection Algorithm

Anomaly detection algorithms look at an unlabeled dataset of normal events and thereby learns to detect or to raise a red flag for if there is an unusual or anomalous event. A anomaly detection algorithm will always have following steps.

- Choose $n$ features $x_i$ that you think might be indicative of anomalous examples.
- Fit parameters $\mu_1, \cdots, \mu_n, \sigma_1^2, \cdots, \sigma_n^2$

$$
\mu_j = \frac{1}{m}\sum_{i=1}^m{x_j^{(i)}}\ \ \ \ \ \ \sigma_j^2 = \frac{1}{m}\sum_{i=1}^m{(x_j^{(i)})-\mu_j^2}
$$

- Given new example $x$, compute $p(x)$

$$
p(x) = \prod_{j=1}^n{p(x_j;\mu_j;\sigma_j^2)} = \prod_{J=1}^n{\frac{1}{\sqrt{2\pi}\sigma_j}\exp(-\frac{(x_j-\mu_j)^2}{2\sigma_j^2})}
$$



Assume that $\epsilon$ is small enough, then anomaly if $p(x) < \epsilon$. 

#### Developing and Evaluating Anomaly Detection System

we often evaluate anomaly detection system on a cross validation set to choose parameter $\epsilon$ . Here are some possible evaluation metrics

- True Positive(TP), False Positive(FP), False Negative(FN), True Negative(TN)
- Precision/Recall
- F1-score

Anomaly detection is suitable for scenarios dealing with extremely unbalanced data, difficult to label, or where the type of anomaly is unknown.

#### Choosing What Features to Use

In anomaly detection, Gaussianization of part of the data is a common preprocessing step to transform the data into a form that more closely matches a Gaussian distribution. By Gaussianising the data, it is possible to simplify the distributional properties of the data to more closely approximate these assumptions, thus improving the effectiveness of the algorithm.

At the same time, Gaussian distribution has good mathematical properties for further analysis and calculations. Since the probability density function (PDF) of the Gaussian distribution has a closed form, which facilitates the calculation of statistics such as probabilities and confidence intervals.

## Week 2

### Recommender System

If we want to build a recommender system, we need to build a model for each user and analyze their preferences.

#### Collaborative Filtering 

##### Cost Function

Suppose we are building a movie recommender system, and now we have following parameters

- $r(i, j) = 1$ if user $j$ has rated movie $i$ ($0$ otherwise)
- $y^{(i, j)} = $ rating given by user $j$ on movie $i$ (if defined)
- $w^{(j)}, b^{(j)} = $ parameters for user $j$
- $x^{(i)} = $ feature vector for movie $i$

For user $j$ and movie $i$, predict rating: $w^{(j)}\cdot x^{(i)} + b^{(j)}$

$m^{(j)} = $ no. of movies rated by user $j$

To learn parameters $w^{(j)}, b^{(j)}$ for user $j$, we have cost function

$$
J(w^{(j)}, b^{(j)})) = \frac{1}{2}\sum_{i:r(i,j)=1}(w^{(j)}\cdot x^{(i)} + b^{(j)} - y^{(i,j)})^2 + \frac{\lambda}{2}\sum_{k=1}^{n}(w_k^{(j)})^2
$$

To learn parameters $w^{(1)}, b^{(1)}, w^{(2)}, b^{(2)}, \cdots, w^{(n_u), b^{(n_u)}}$ for all users, we have cost function

$$
J\left(
\begin{matrix}
w^{(1)}, \cdots, w^{(n_u)} \\
b^{(1)}, \cdots, b^{(n_u)} 
\end{matrix}
\right) 
= \frac{1}{2}\sum_{j=1}^{n_u}\sum_{i:r(i,j)=1}(w^{(j)}\cdot x^{(i)} + b^{(j)} - y^{(i,j)})^2 + \frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^{n}(w_k^{(j)})^2
$$

Then we can minimize the cost function and find the best value for the parameters.

##### Collaborative Filtering Algorithm

Sometimes we do not know the feature vector for the movies, and we want to inference them from the rating given by users. This method analyzes the similarities between users and then calculate the possible feature vector for the movies. 

We already have the cost function to learn $w^{(1)}, b^{(1)}, \cdots, w^{(n_u), b^{(n_u)}}$

$$
J(w, b) = \frac{1}{2}\sum_{j=1}^{n_u}\sum_{i:r(i,j)=1}(w^{(j)}\cdot x^{(i)} + b^{(j)} - y^{(i,j)})^2 + \frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^{n}(w_k^{(j)})^2
$$

Similarly, we can know the cost function to learn $x^{(1)}, \cdots, x^{(n_m)}$

$$
J(x) = \frac{1}{2}\sum_{i=1}^{n_m}\sum_{j:r(i,j)=1}(w^{(j)}\cdot x^{(i)} + b^{(j)} - y^{(i,j)})^2 + \frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^{n}(x_k^{(i)})^2
$$

Integrate the two cost function above, we have 

$$
J(w,b,x) = \frac{1}{2}\sum_{(i,j):r(i,j)=1}(w^{(j)}\cdot x^{(i)} + b^{(j)} - y^{(i,j)})^2 + \frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^{n}(w_k^{(j)})^2 + \frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^{n}(x_k^{(i)})^2
$$

To get the best parameters for $w,b,x$, we need to minimize $J(w,b,x)$. 

One simple method is using gradient descent, and we can implement the gradient descent algorithm like this

$$
w_i^{(j)} := w_i^{(j)} - \alpha\frac{\partial}{\partial w_i^{(j)}}J(w,b,x) \\
b^{(j)} := b^{(j)} - \alpha\frac{\partial}{\partial b^{(j)}}J(w,b,x) \\
x_k^{(i)} := x_k^{(i)} - \alpha\frac{\partial}{\partial x_k^{(i)}}J(w,b,x) \\
$$

##### Binary Labels

Many important applications of recommended systems or collective filtering algorithms involved binary labels instead of multiple labels. 

We need a new loss function which is more appropriate for binary labels. And the loss function is 

$$
y^{(i,j)}:f_{(w,b,x)}(x) = g(w^{(j)}\cdot x^{(i)}+b^{(j)}) \\
L\left(f_{(w,b,x)}(x),y^{(i,j)}\right) = -y^{(i,j)}\log\left(f_{(w,b,x)}(x)\right) - (1-y^{(i,j)})\log\left(1-f_{(w,b,x)}(x)\right) \\
J(w,b,x) = \sum_{(i,j):r(i,j)=1}L\left(f_{(w,b,x)}(x),y^{(i,j)}\right)
$$

So with this loss function, we can take the linear regression like collaborative filtering algorithm and generalize it to work with binary labels. 

##### Mean Normalization

If we first carry out mean normalization, the model will perform better. For a new user who has not rate any movie, we can use mean normalization to make the prediction more reasonable. We take all of the values from the users and put them into a two dimensional matrix. And then calculate the average rating for each movie to become a vector $\mu$. For each column in the matrix, it should minus the vector $\mu$ and get a new matrix. Now we can use this new matrix to predict the rating from new users. And the new predict result will be
$$
w^{(j)}\cdot x^{(i)} + b^{(j)} + \mu_i
$$
Normalization makes the algorithm tun a bit faster, but even more important, it makes the algorithm give much better, much more reasonable when there are users that rated very few movies or even no movies at all. 

#### Content-based Filtering 

##### Introduction

We have learned the collaborative filtering so far, and we will start to develop a second type of recommender system called content-based filtering algorithm. Instead of depending on rating of users who gave similar ratings, this system is based on features of user and item to find good match. 

We also use $r(i, j)$ to denote whether user $j$ has rated item $i$, and use $y(i,j)$ to denote the rating given by user $j$ on item $i$. Suppose $x_u^{(j)}$ is the user feature for user $j$ and $x_m^{(i)}$ is the movie features for movie $i$. 

Then we predict rating of user $j$ on movie $i$ as 
$$
v_u^{(j)}\cdot v_m^{(i)}
$$
where $v_u^{(j)}$ is computed from $x_u^{(j)}$ and $v_m^{(i)}$ is computed from $x_m^{(i)}$. Apparently, vector $v_u$ represents the user's preferences and vector $v_m$ represents the movie features. 

##### Deep Learning for Content-Based filtering

A good way to develop a content based filtering algorithm is to use deep learning. We can build a user network which takes $x$ input the list of features of the user $x_u$, such as the age, gender, the country of the user, and so on. Then using a few layers, say dense neural network layers, it will output this vector $v_u$ that describe the user. Notice that in this neural network, the output layer has several units, and so $v_u$ is actually a list of 32 numbers. Similarly, to compute $v_m$ for a  movie,  we can have a movie network which takes $x$ input features of the movie and through a few layers of a neural network ends up outputting $v_m$, the vector that describes the movie. Finally, we will predict the rating of this user on that movie as $v_u\cdot v_m$. We can also use a sigmoid function to get the input between $0$ and $1$, like $g(v_u\cdot v_m)$.

##### Cost Function

For the neural network above, the cost function is
$$
J = \sum_{(i,j):r(i.j)=1}(v_u^{(j)}\cdot v_m^{i}-y^{(i,y)})^2+\text{NN regularization term}
$$

##### Recommending form a Large Catalogue

For a website who has more than $10$ million users, we cannot compute the neural network for each user. Many large scaled recommend system are implemented as two steps, which are called **Retrieval** and **Ranking** steps. The retrieval step will generate a large list of plausible item candidates that tries to cover a lot of possible things you might recommend to user. Then we combine retrieved items into list, removing duplicates and items that we do not want to recommend for the user anymore. The second step is ranking step. in the ranking step, we take list retrieved and rank using learn model and display ranked items to user. 

Retrieving more items results in better performance, but slower recommendations.

To analyze/optimize the trade-off carry out offline experiments to see if retrieving additional items results in more relevant recommendations (i.e., $p(y(^{i.j})) = 1$ of items displayed to user are higher).

## Week 3

### Reinforcement Learning

#### What is Reinforcement Learning

**Reinforcement Learning (RL)** is a branch of machine learning where an agent learns to make decisions by interacting with an environment. The agent performs actions, observes the results, and receives rewards or penalties based on the outcome. The goal of RL is to train the agent to maximize cumulative rewards over time by learning the best strategies (also known as policies) through trial and error. RL is widely used in areas like robotics, gaming, autonomous systems, and more, as it is well-suited for problems that require sequential decision-making in dynamic environments.

#### The Return in Reinforcement Learning

The return in reinforcement learning is the sum if the rewards that the system gets, but weighted by the discount factor, where rewards in the far future are weighted by the discount factor raised to a higher power. If there are any rewards are negative, then the discount factor actually incentivizes the system to push out the negative rewards as fat into the future as possible. 

#### Making Decisions: Policies in Reinforcement Learning 

In reinforcement learning, our goal is to come up with a function which is called a policy $\pi$, whose job is to take as input, any state $s$ and map it to some action $a$ that it wants us to take. The goal of reinforcement learning is to find a policy $\pi$ or $\pi(s)$ that tells you what action to take in every state so as to maximize the return. 

### State-Action Value Function

#### State-Action Value Function Definition

The state-action value function is a function typically denoted by the letter $Q$, and it is a function of a state you might be in, as well as the action you might choose to take in that state. The Q-value $Q(s,a)$ represents the return you get if you start in state $s$, take action $a$ once, and then follow the optimal policy thereafter. The best possible return from state $s$ is $\underset{a}{\max}Q(s,a)$. The best possible action in state $s$ is the action $a$ that gives $\underset{a}{\max}Q(s,a)$. 

#### Bellman Equation

The **Bellman equation** is a fundamental recursive relationship used in dynamic programming and reinforcement learning to describe the value of a decision problem. It breaks down the value (or expected reward) of a state into two components: the immediate reward from taking an action in the current state, and the expected value of future rewards from the subsequent state. The Bellman equation helps define the optimal policy by relating the value of a state to the values of the possible subsequent states, forming the basis for algorithms like value iteration and Q-learning.

To better understand Bellman equation, let's suppose 

- $s$ is the current state
- $a$ is the current action
- $R(s)$ is the reward of current state 
- $\gamma$ is the discount factor 
- $s'$ is the state we get to after taking action $a$
- $a'$ is the action that we take in state $s'$

Then the Bellman Equation is 
$$
Q(s,a) = R(s)+\gamma\underset{a'}{\max}Q(s',a')
$$

#### Random Environment

In some applications, when we take an action, the outcome is not always completed reliable. Since the goal of reinforcement learning is to choose a policy $\pi(s)= a$ that will tell us what action a to take in state s so as to maximize the expected return, then the new Bellman equation will be
$$
Q(s,a) = R(s)+\gamma E[\underset{a'}{\max}Q(s',a')]
$$
In the equation, $E$ means the average value of $\underset{a'}{\max}Q(s',a')$. 

#### State-Value Function

For the Bellman Equation, firstly we initialize neural network randomly as guess of $Q(s,a)$. 

Then we repeat the following steps:

- Take action in the lunar lander. Get $(s,a,R(s),s')$.
- Store $10000$ most recent $(s,a,R(s),s')$ tuples.

After these steps, we can train the neural network by creating training set of $10000$ examples using $x=(s,a)$ and $y=R(s)+\gamma \underset{a'}{\max}Q(s',a')$. 
