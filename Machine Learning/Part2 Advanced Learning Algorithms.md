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

## Week 2

## Week 3

## Week 4

