# Supervised Machine Learning Regression and Classification

## Week 1

### Introduction

#### What is Machine Learning

#### Supervised Learning 

Data comes with both inputs and outputs. Algorithm has to classify new inputs into corresponding outputs. 

- Regression

  Predict a infinite value based on some given inputs and outputs. 

- Classification

  Label something into some limited classes.

#### Unsupervised Learning 

Data only comes with inputs data, but not output their corresponding labels. Algorithms has to find **structure** in the data. 

- Clustering

  Group similar data points together.

- Anomaly detection

  Find unusual data points.

- Dimensionality reduction

  Compress data using fewer numbers.

### Linear Regression with One Variable (Univariate Linear Regression)

#### Model Representation

We provide a dataset to train our algorithm which called **Training Set**, and then feed them to our **Learning Algorithm**. In this part, the training set is composed by one variable inputs and outputs. They are usually marked as  $x^i$  and  $y^i$ , like  $(x^i, y^i = 2104, 460)$ .

The learning algorithm will produce a function  $f$ , also called **model**. This function accept  $x$  as **feature** and produce a  $\hat{y}$  as **prediction**. When the result is  $y$ , it means it is the actual value from the training set (also called **target**). In contrast,  $\hat{y}$  is a estimated value of y instead of a true value.

Model  $f(x)$  is actually  $f_{w, b}(x) = wx+b = \hat{y}$ , where  $w$  and  $b$  are parameters (or coefficients and weights). In machine learning, the parameters of the model are the variables you can adjust during training in order to improve the model.  

#### Cost Function (Squared Error Cost Function)

Cost function is used to calculate the error of the model  $f$ . The smaller value of the cost function means the model is more accurate. The cost function is like

$$
J(w, b) = \frac{1}{m}\sum_{i=1}^m\left(\hat{y}^{(i)} - y^{(i)}\right)^2
$$

This is the initial cost function which uses the squared error to compute the cost.

In order to compute easily, machine learning people always use the following function as the cost function.

$$
\begin{aligned}
J(w, b) &= \frac{1}{2m}\sum_{i=1}^m\left(\hat{y}^{(i)} - y^{(i)}\right)^2 \\
&= \frac{1}{2m}\sum_{i=1}^m\left(f_{w, b}(x^{(i)}) - y^{(i)}\right)^2
\end{aligned}
$$

### Gradient Descent

#### Gradient Descent Algorithm

Gradient descent algorithm is an common method to find the minimum of cost function. Firstly we choose a random parameters combination and compute the value of cost function, then we look for the next parameter combination that reduces the value of cost function the most until we find a **local minimum**. But a local minimum is not always a global minimum, so we will choose different parameters combination to find the **global minimum**. 

Suppose that we have a combination of parameters  $(w_1, b)$ , and if we want to find the global minimum, we can start with some  $w, b$ , like  $w = b = 0$ . Then we keep changing  $w, b$  to reduce the  $J(w, b)$  until we settle at or near a minimum.

The function of batch gradient descent algorithm is

$$
w := w - \alpha\frac{\partial}{\partial{w}}J(w, b) \\
b := b - \alpha\frac{\partial}{\partial{w}}J(w, b) \\
$$

Variable  $\alpha$  is called **learning rate**, which determines how large of a step we take downward in the direction that minimize the cost function. Its value should in  $[0, 1]$ .

Be careful that these parameters should be updated simultaneously. If we calculate these  $w, b$  sequentially, then we find that the new value of  $w$  will be used to compute  $b$ , instead of the original value, and that is **wrong**. Therefore, we can compute the parameters like following

$$
\begin{aligned}
temp_w &:= w - \alpha\frac{\partial}{\partial{w}}J(w, b) \\
temp_b &:= b - \alpha\frac{\partial}{\partial{w}}J(w, b) \\
w &:= temp_w \\
b &:= temp_b \\
\end{aligned}
$$

Then we can generalize this method to more parameters. Suppose that we have a combination of parameters  $(w_1, w_2, \cdots, w_n, b)$ , we can implement the gradient descent algorithm like this

$$
\begin{aligned}
temp_i &:= w_i - \alpha\frac{\partial}{\partial{w_i}}J(w_0, w_2, \cdots, w_n), i = 0, 1, \cdots, n \\
w_i &:= temp_i \\
\end{aligned}
$$

#### Learning Rate

Learning rate( $\alpha$ ) is a significant parameter in the expression of the cost function which determines **how large we take one step**. 

If we choose a very small value for the learning rate, we will take many small steps until we reach the minimum of the cost function. Otherwise, if we choose a very large value for it, we will take a large step each time, and then we may go far away from the minimum of the cost function. So, it is significant to choose a appropriate value for the learning rate. 

#### Gradient Descent for Linear Regression

We can calculate the partial derivative of the Cost function and then replace their expressions.

$$
\begin{aligned}
\frac{\partial}{\partial{w}}J(w, b) &= \frac{1}{m}\sum_{i=1}^{m}{f_{w, b}(x^{(i)} - y^{(i)})}x^{(i)} \\
\frac{\partial}{\partial{b}}J(w, b) &= \frac{1}{m}\sum_{i=1}^{m}{f_{w, b}(x^{(i)} - y^{(i)})} \\
\end{aligned}
$$

 Therefore, we can rewrite the function of batch gradient descent algorithm like following

$$
\begin{aligned}
w &:= w - \alpha\frac{1}{m}\sum_{i=1}^{m}{f_{w, b}(x^{(i)} - y^{(i)})}x^{(i)} \\
b &:= b - \alpha\frac{1}{m}\sum_{i=1}^{m}{f_{w, b}(x^{(i)} - y^{(i)})} \\
\end{aligned}
$$

## Week 2

### Multiple Features

We have already know what is linear regression with one variable and its model  $f_{w, b}(x) = wx+b$  .

And now let's talking about the linear regression with two or more than two variables. 

Let $n$ denotes the number of features, $x_j$ denote the $j^{th}$ feature, $x^{(i)}$ denotes the features of $i^{th}$ training example, and $x_j^{(i)}$ denotes the value of feature $j$ in $i^{th}$ training example. Then we can get a model with $n$ features. 

$$
\begin{aligned}
f_{\vec{w}, b}(x) &= \vec{w}\cdot\vec{x} + b \\
&= [w_1, w_2, w_3, \cdots, w_n]\cdot\vec{x} + b \\
&= w_1x_1 + w_2x_2 + w_3x_3 + \cdots + w_nx_n + b
\end{aligned}
$$

In this expression, $\vec{w}$ is a row vector and $\vec{x}$ is a column vector. 

#### Gradient Descent for Multiple Regression

In Week 1, we have implemented the gradient descent algorithm. 

$$
w_i := w_i - \alpha\frac{\partial}{\partial{w_i}}J(w_0, w_2, \cdots, w_n), i = 0, 1, \cdots, n \\
$$

This expression can be rewrited as

$$
w_1 := w_1 - \alpha\frac{1}{m}\sum_{i=1}^{m}{f_{\vec{w}, b}(x^{(i)} - y^{(i)})}x_1^{(i)} \\
w_2 := w_2 - \alpha\frac{1}{m}\sum_{i=1}^{m}{f_{\vec{w}, b}(x^{(i)} - y^{(i)})}x_2^{(i)} \\
\vdots \\
w_n := w_n - \alpha\frac{1}{m}\sum_{i=1}^{m}{f_{\vec{w}, b}(x^{(i)} - y^{(i)})}x_n^{(i)} 
\\
b := b - \alpha\frac{1}{m}\sum_{i=1}^{m}{f_{\vec{w}, b}(x^{(i)} - y^{(i)})}
\\
$$

### Feature Scaling

When we face multi-dimensional feature problems, we need to ensure that these features have similar scales, which will help the gradient descent algorithm converge faster. 

Suppose that we have two features, one ranges from $0$ to $5$ ,  and the other one ranges from $0$ to $2000$ . If we use these two parameter to draw a diagram of the cost function, then we will find the image will more like a **ellipses**, and the gradient descent algorithm requires many iterations to converge. 

The solution is to make sure features are on a similar scale. In the example above, we have $x_1 \in [0, 5]$ , $x_2 \in [0, 2000]$ . We can transfer their ranges to a similar scale like $[0, 1]$ . So we have

$$
x_1 := \frac{x_1}{5} \\
x_2 := \frac{x_2}{2000} \\
$$

There are also some useful scaling methods. 

- Mean Normalization

  For parameter $x_i$, we have 
  
$$
x_i := \frac{x_i - \mu_i}{max-min}
$$

  $\mu_i$ is the mean value of $x$ , namely, $\mu_i = \bar{x}$. $max$ and $min$ are the max value and min value in $x$ . 

- Z-score normalization

  For parameter $x_i$, we have 
  
$$
x_i := \frac{x_i - \mu_i}{\sigma_1}
$$

  In this expression, $\mu_i$ has the same meaning in mean normalization, and $\sigma_1$ denotes the standard deviation of $x$ . So $\sigma_1^2$ is the variance of $x$ . 

**Notice**

If $x$ is already appropriate to run gradient descent, then we don't need to scale it. Here are some acceptable intervals: $[0, 3], [-2, 0.5], [-3, 3], [-0.3, 2]$. 

### Checking Gradient Descent for Convergence

We need to know that if the gradient descent is converging when running gradient descent. Let's draw a diagram whose horizonal axis is the number of iterations of gradient descent that you've run so far, and vertical axis is a number of cost function $J(\vec{w}, b)$ . When the cure is no longer decreasing much, then we know the gradient descent converges. We can also use an automatic convergence test to judge if the gradient descent is converging. Let $\epsilon = 0.001$ or any number that small enough, if $J(\vec{w}, b)$ decreases by $\leqslant \epsilon$ in one iteration, that declare the convergence. 

### Choosing the Learning Rate

In the former class, we learned that if we choose a very big learning rate, the cost function $J(\vec{w} , b)$ will be hard to converge or even can not converge. And if we choose a very small learning rate, it will take much more steps to be convergent. So choosing an appropriate learning rate is very important. 

If out cost function is not monotonic decreasing, it may caused by choosing a too big learning rate or there are some bugs in our code. To find out the reason, we can first set the learning rate with a very small value like $0.001$. Then run the gradient descent to find the variation of the cost function. If the cost function still does not decrease monotonically, it means that there are some bugs in our code. Otherwise, we can replace the learning rate with current value times $3$ or $10$, or some other appropriate value until we find a just right value. 

### Feature Engineering

Suppose we have a house with length $x_1$ and width $x_2$, we may write down a model like this

$$
f_{\vec{w}, b}(x) = w_1x_1 + w_2x_2 + b
$$

But let's think about it seriously, when purchasing a house, we not only consider the width and length of it, but also the area. So, this model is not very appropriate and that means linear regression does not work for all data. We can add a parameter $x_3 = x_1\cdot x_2$ to denotes the area of the house. The new model is

$$
f_{\vec{w}, b}(x) = w_1x_1 + w_2x_2 + w_3x_3 + b
$$

Remember $x_3$ is a combination of $x_1$ and $x_2$, so this is a quadratic expression.

### Polynomial Regression

Sometimes even a quadratic expression can not suit our need, and we have to add more parameters into the expression. Usually, we observe the data and determine which model should we try. The model will like this

$$
f_{\vec{w}, b}(x) = w_1x + w_2\sqrt{x} + w_3x^2 + \cdots + b
$$

**Notice**

In polynomial regression, feature scaling is very important. Consider a model $f_{\vec{w}, b}(x) = w_1x + w_2x^2 + w_3x^3$ . If we have $x \in [0, 100]$ , then $x^2 \in [0, 10000], x^3 \in [0, 1000000]$ . That means $w_3$ will almost determine the value of $f_{\vec{w}, b}(x)$ and that is not what we want to. So feature scaling is really important. 

## Week 3

### Logistic Regression

#### Classification

In a classification problem, the variable you want to predict $y$ is a discrete value, we will use an algorithm called logistic regression to solve the problem. In a classification problem, we are trying to predict whether an outcome belongs to a certain class (e.g., correct or incorrect). 

To begin with, let's consider a **binary classification** first. We divide the dependent variable into two possible classes: negative class and positive class. Then we have $y \in \{0, 1\}$, where $0$ implies negative class and $1$ implies positive class. Obviously, the predicted value of logistic regression is between $0$ and $1$. 

#### Hypothesis Representation

As we mentioned above, we need a model which can output a value between $0$ and $1$. So the hypothesis of logistic regression model is

$$
f_{\vec{w}, b}(\vec{x}) = g(\vec{w}\cdot\vec{x}+b) = \frac{1}{1+e^{-(\vec{w}\cdot\vec{x}+b)}}
$$

$X$ implies feature vector and $g$ implies logistic function. Function $g(z) = \frac{1}{1+e^{-z}}$ , also called **Sigmoid function** is a commonly used logical function. 

The purpose of  $h_{\theta}(x)$ is, for a given input variable, calculate the possibility of the predicted variable = $ 1 $ according to the selected parameters. 

#### Decision Boundary

Consider we have a straight line $y = -x + 1$, the part of the model above this line has a predicted value of $1$ and the part of the model below this line has a predicted value of $0$. So we called this line **Decision Boundary** because it divides the predicted value into two parts. 

For a simple model, it always has a simple decision boundary. And for a complicated model, it may have a complicated decision boundary. 

#### Cost Function

For linear regression models, the cost function we define is the sum of squares of all model errors. Theoretically, we can also use this definition for the logistic regression model, but the problem is that when we change $f_{\vec{w}, b}(\vec{x}) = \frac{1}{1+e^{-(\vec{w}\cdot\vec{x}+b)}}$ is brought into the cost function defined in this way, the cost function we get will be a **non-convex function**.

This means that our cost function has many local minima, which will affect the gradient descent algorithm to find the global minimum.

Therefore, we define **Logistic Loss Function**

$$
L(f_{\vec{w}, b}(\vec{x}^{(i)}), y^{(i)}) = 
\begin{cases}
\begin{array}{ll}
-\log(f_{\vec{w}, b}(\vec{x}^{(i)}))  & \mathrm{if}\space y^{(i)} = 1 \\
-\log(1-f_{\vec{w}, b}(\vec{x}^{(i)}))  & \mathrm{if}\space y^{(i)} = 0 \\
\end{array}
\end{cases}
$$

So the Cost function of logistic regression is

$$
    J(\vec{w}, b) = \frac{1}{m}\sum_{i=1}^m{L\left(f_{\vec{w}, b}(\vec{x}^{(i)}) - y^{(i)}\right)}
$$


#### Simplified Cost Function

From logistic loss function, we can get a simplified version of this expression

$$
L(f_{\vec{w}, b}(\vec{x}^{(i)}), y^{(i)}) = -y^{(i)}\log{\left(f_{\vec{w}, b}(\vec{x}^{(i)})\right)} - (1-y^{(i)})\log{\left(1-f_{\vec{w}, b}(\vec{x}^{(i)})\right)}
$$

Therefore the simplified cost function is 

$$
\begin{aligned}
J(\vec{w}, b) &= \frac{1}{m}\sum_{i=1}^m{L\left(f_{\vec{w}, b}(\vec{x}^{(i)}) - y^{(i)}\right)} \\
&= \frac{1}{m}\sum_{i=1}^m{\left(-y^{(i)}\log{\left(f_{\vec{w}, b}(\vec{x}^{(i)})\right)} - (1-y^{(i)})\log(1-f_{\vec{w}, b}(\vec{x}^{(i)})\right)} 
\end{aligned}
$$

#### Gradient Descent Implementation

Let's apply the gradient descent to the logistic regression. 

$$
\begin{aligned}
\frac{\partial}{\partial{w_j}}J(\vec{w}, b) &= \frac{1}{m}\sum_{i=1}^{m}{f_{\vec{w}, b}(\vec{x}^{(i)} - y^{(i)})}x_j^{(i)} \\
\frac{\partial}{\partial{b}}J(\vec{w}, b) &= \frac{1}{m}\sum_{i=1}^{m}{f_{\vec{w}, b}(\vec{x}^{(i)} - y^{(i)})} \\
\end{aligned}
$$

Then we updates the parameter simultaneously

$$
\begin{aligned}
w_j &= w_j - \alpha\left[\frac{1}{m}\sum_{i=1}^{m}{f_{\vec{w}, b}(\vec{x}^{(i)} - y^{(i)})}x_j^{(i)}\right] \\
b &= b - \alpha\left[\frac{1}{m}\sum_{i=1}^{m}{f_{\vec{w}, b}(\vec{x}^{(i)} - y^{(i)})}\right] \\
\end{aligned}
$$

Although this expression is similar with the gradient descent of linear regression, they are very different because we use $f_{\vec{w},b}(x) = \vec{w}\cdot\vec{x} + b$ for linear regression and use $\displaystyle f_{\vec{w},b}(x) = \frac{1}{1+e^{(-\vec{w}\cdot\vec{x}+b)}}$ for logistic regression. 

### Overfitting

#### The Problem of overfitting

If a model fits the training set extremely well, we call it **overfitting**. A overfitting model always have a small bias but a high variance. It emphasis on fitting the original data too much and losing the origin of the algorithm. For a bigger dataset, a overfitting model will not perform very well. 

For example, a polynomial model is consists of terms containing powers of x. The higher the power of x, the better the fit, but the corresponding prediction ability may become worse.

#### Addressing Overfitting

We have already know that what is overfitting. So now let's talk about what we can do to address it. 

One way to address the problem is to **collect more training data**. If we are able to get more data, that is more training examples, then with the larger training set, the learning algorithm will learn to fit a function that is less wiggly. So we can keep this model with lots of features. 

But there isn't more data to be add usually. Hence, the second method to solve overfitting is to see if we can **use fewer features**. We can also discard some features that do not help us predict correctly, and manually select which feature to retain, or use some automatic algorithms to help us. 

The third option for reducing overfitting is **regularization**. We may find that the parameters are often relatively large in a overfitting model. Regularization is a way to gently reduce the impacts of some of the features without eliminating them. What regularization does is **encouraging the learning algorithm to shrink the values of the parameters** without necessarily demanding that the parameters is set to exactly 0. By using regularization, we can end up with a curve that ends up fitting the training data much better. So what regularization does is it lets you keep all of your features, but they just prevents the features from having a overly large effect which is what sometimes can cause overfitting. By convention, we normally just reduce the size of the $w_j$ parameters, but not really encourage $b$ to become smaller. 

### Regularization

#### Cost Function with Regularization

To reduce overfitting, we need to shrink the values of the parameters, especially the high power features. So the way regularization is typically implemented is to penalize all of the features of more precisely and preventing them from being too large. It's possible to show that this will usually result in fitting a smoother simpler, less weekly function that's less prone to overfitting. We introduce a new parameter $\lambda$ called **regularization parameter**, which is similar to the learning rate $\alpha$. And then we can get the cost function with regularization

$$
J(\vec{w}, b) = \frac{1}{2m}\sum_{i=1}^m\left(f_{\vec{w}, b}(x^{(i)}) - y^{(i)}\right)^2 + \frac{\lambda}{2m}\sum_{j=1}^n{w_j^2}
$$

If the regularization parameter we choose is enormous, all parameters will be minimized, leading to **under fitting**. And if the regularization parameter we choose is very small, then the penalty is also small, may still lead to **overfitting**. So for the regularization, we need to choose a reasonable $\lambda$ value, so that regularization can be better applied. 

#### Regularized Linear Regression

For the cost function with regularization, the term $\frac{\partial}{\partial{w_j}}J(\vec{w},b)$ is 

$$
\frac{\partial}{\partial{w_j}}J(\vec{w},b) = \frac{1}{m}\sum_{i=1}^m\left[{(f_{\vec{w}, b}(\vec{x}^{(i)}) - y^{(i)})}x_j^{(i)}\right]
$$

Then we can implement gradient descent for linear regression with following formula

$$
\begin{aligned}
w_j &= w_j - \alpha\left[\frac{1}{m}\sum_{i=1}^m\left[{(f_{\vec{w}, b}(\vec{x}^{(i)}) - y^{(i)})}x_j^{(i)}\right]+\frac{\lambda}{m}w_j\right] \\
b &= b - \alpha\frac{1}{m}\sum_{i=1}^m{(f_{\vec{w}, b}(\vec{x}^{(i)}) - y^{(i)}})
\end{aligned}
$$

#### Regularized Logistic Regression

Similar to the regularization of linear regression, the renewed cost function of logistic regression is

$$
J(\vec{w}, b) = -\frac{1}{m}\sum_{i=1}^m{\left(y^{(i)}\log{\left(f_{\vec{w}, b}(\vec{x}^{(i)})\right)} + (1-y^{(i)})\log(1-f_{\vec{w}, b}(\vec{x}^{(i)})\right)} + \frac{\lambda}{2m}\sum_{j=1}^n{w_j^2}
$$

 Then we can implement gradient descent for logistic regression with following formula

$$
\begin{aligned}
w_j &= w_j - \alpha\left[\frac{1}{m}\sum_{i=1}^{m}{f_{\vec{w}, b}(\vec{x}^{(i)} - y^{(i)})}x_j^{(i)} + \frac{\lambda}{m}w_j\right] \\
b &= b - \alpha\left[\frac{1}{m}\sum_{i=1}^{m}{f_{\vec{w}, b}(\vec{x}^{(i)} - y^{(i)})}\right] \\
\end{aligned}
$$

