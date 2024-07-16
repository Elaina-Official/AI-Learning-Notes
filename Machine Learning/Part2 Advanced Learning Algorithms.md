# Advanced Learning Algorithms

## Week 1

### Neurons and the brain

#### Neural Networks

The human brain, or maybe more generally, the biological brain demonstrates a higher level or more capable level of intelligence and anything else would be on the bill so far. So neural networks has started with the motivation of trying to build software to mimic the brain. The first application area that modern neural networks or deep learning, had a huge impact on was probably speech recognition, where we started to see much better speech recognition systems due to modern deep learning. And then it started to make inroads into computer vision. Then the next few years, it made us inroads into texts or into natural language processing, and so on and so forth. Now, neural networks are used in everything from climate change to medical imaging to online advertising to products recommendations and really lots of application areas of machine learning now use neural networks. 

The artificial neural networks uses a very simplified mathematical model of what a biological neuron does. A **neuron** take some inputs and then does some computation and outputs some other number, which then could be an input to a second neuron. When you are building an artificial neural network or deep learning algorithm, rather than building one neuron at a time, you often want to simulate many such neurons at the same time. So there will be many neurons and they will collectively input a few numbers, carry out some computation, and output some other numbers. 

#### Why Now

In many application areas, the amount of digital data has exploded. And those traditional learning algorithm were not able to take effective advantage of more data we had for different applications. But if you were to train a small neural network on a dataset, then the performance is much better than traditional learning algorithms. And if you were to train a very large neural network, meaning one with a lot of these artificial neurons, then for some applications, the performance will keep on going up. If you are able to train a very large neural network to take advantage of that huge amount of data you have, then you could attain performance on anything, ranging from speech recognition to image recognition, to natural language processing applications and many more, they just were not possible with earlier generations of learning algorithms. This caused deep learning algorithms to taken off, and this too is why faster computer processes, including the rise of GPUs or graphics processer units. This is hardware originally designed to generate nice-looking computer graphics, but turned out to be really powerful for deep learning as well. That was also a major force in allowing deep learning algorithms to become what it is today. 

### Neural Network Model

#### Simple Neural Network

A simple three-layer neural network consists of an input layer, a hidden layer, and an output layer. The **input layer** receives the initial data, with each neuron representing a feature of this data. It then passes the data to the hidden layer, which performs core computations. Neurons in the **hidden layer** apply weights and biases to the inputs, process them through an activation function, and learn complex patterns. The **output layer** receives this processed information and generates the final output. The number of neurons in the output layer corresponds to the number of output classes or the type of prediction required. 

#### More Complex Neural Networks

A complex neural network always contains more than one hidden layer. To denote the multiple hidden layer, we use superscript $[i]$ to represent the $i\text{-th}$ layer of the network. And by convention, layer $0$ is the input layer and layer $n$ is the output layer. When we say that a neural network has $n$ layers, that includes all the hidden layers and the output layer, but we don't count the input layer. 

#### Forward Propagation

In the network, we propagating the activations of the neurons from the left to right. Starting from the input layer, each layer of neurons processes the input by applying weights, biases, and an activation function, and then passes the result to the next layer. This continues through the hidden layers until the output layer produces the final prediction or classification. Thus this is called forward propagation. 

With the neural networks inference using the forward propagation algorithm, we are able to download the parameters of a neural network that someone had trained to carry out inference on our new data using their neural network. 

### Other Activation Functions

#### Three Basic Activation functions

Besides Sigmoid function, we have 2 other basic activation functions as well. 

- Linear Activation Function (No activation function)

  The expression is

$$
g(z) = z
$$

- **ReLU (Rectified Linear Unit)**

  The expression is

$$
g(z) = max(0, z)
$$

#### Choosing Activation functions

Now we know that there are many activation functions are used in deep learning. But the **ReLU** activation function is by far the most common choice in how neural networks are trained in the hidden layer. 

Here are the reasons why people prefer ReLU activation function

- ReLU is a bit faster to compare because it just requires computing max of 0 and z, where sigmoid requires taking an exponentiation and then a inverse and so on. 
- ReLU function goes flat only in one part of the graph where the sigmoid activation function it goes flat in both left and right of the graph. So ReLU activation function will performance better in gradient descent and make out neural network to learn a bit faster as well.  

Therefore, for the output layer, we recommend **sigmoid activation function** for **binary classification**, **linear activation function** for **linear regression** and $y$ can take on positive or negative values, and **ReLU 0999778activation function** for **regression** and $y$ can take only non-negative values. And for the hidden layer, we recommend just using **ReLU** as a default activation function. 

**Notice: Do not use linear activations in hidden layers.**

### Multiclass Classification

A multiclass classification problem is still a classification problem in that $y$ can take on more than two possible values. But multiclass classification problem is still a classification problem in that $y$ can take on only a small number of discrete categories is not any number. 

#### Softmax

We have already know the expression of logistic regression. And **Softmax regression** is similar with logistic regression. For a softmax regression, we have
$$
z_j = \vec{w_j}\cdot\vec{x} + b_j,\ j = 1, 2, \cdots, n \\
a_j = \frac{e^{z_j}}{\sum_{k=1}^{n}e^{z_k}} = P(y = j|\vec{x}) \\
$$
Apparently, 
$$
\sum_{j=1}^{n}a_j = 1, \ j = 1, 2, \cdots, n \\
$$
The **loss function** is 
$$
loss(a_1, a_2, \cdots, a_n, y) =  
\begin{cases}
\begin{matrix}
-\log{a_1}\ \text{if}\ y = 1 \\
-\log{a_2}\ \text{if}\ y = 2 \\
\vdots \\
-\log{a_n}\ \text{if}\ y = n \\
\end{matrix}
\end{cases}
$$

#### Multi-label Classification

Sometimes we may encounter some problems that require us to classify more than one thing. We can choose multiclass classification or multi-label classification. For multiclass classification, there will be three neural networks and each of them has a output. For multi-label classification, it trains one neural network with three outputs, or we can say, a vector. 

## Week 2

## Week 3

## Week 4

