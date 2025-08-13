# Overview

ScalarFlow is neural network library for scalar valued inputs, named as a spinoff of TensorFlow, (Although the API is closer to PyTorch). It is heavily based off of [micrograd](https://github.com/karpathy/micrograd/blob/master/micrograd/) and I made it as a resource for myself to get familiar with some simpler ML concepts like backpropagation, loss, gradient descent, and evaluating the performance of models.

## Features of the library

-The 'Scalar' class in kernel.py: Create objects that store data and gradients of each neuron. Scalars have numerous methods that help with computation and automatically handles differentiation during backpropagation:
  - Addition, multiplication, subtraction, division, exponentiation (currently can only raise to integers and floats), logarithms (currently only natural log):
    ```
    x = Scalar(4)
    y = Scalar(5)
    x + y #Outputs Scalar(9.0)
    x * y #Outputs Scalar(20.0)
    x - y #Outputs Scalar(-1.0)
    x / y #Outputs Scalar(0.8)
    x**2 #Outputs Scalar(16.0)
    log(2 * y) #Outputs Scalar(1.0)
    ```
  - That last example had an integer before the scalar, which will also work for each operation; order between scalar-integer or scalar-float computation does not matter.
  
  - Each scalar stores its 'parents', the nodes that it was born from via an operation.
    - Example:
      ```
      x = Scalar(4)
      y = Scalar(5)
      z = x + y #z is a child of x and y

      z.prev #Output: {Scalar(data = 4.0, grad = 0.0), Scalar(data = 5.0, grad = 0.0)} 
      ```
- The 'backward()' method: Call this to perform backpropagation. THis can actually be called from anywhere, whether you choose a singular neuron or an entire model. if you call it on an MLP object, the root node will be the output neuron, otherwise it will be   the specified neuron. This will adjust the gradients of each neuron (Which is stored in a scalar as previously mentioned). See the example for some insight into its functionality. 
      
- Initializing multi layer perceptrons (MLPs) with multidimensional inputs with ease:
  - Example: `model = MLP(13, [20, 20, 1])` initializes a model that accepts 13 inputs, and has 2 hidden layers of 20 neurons each.

- You can access the parameters of any singular neuron, an entire layer, or even the entire model by simply using the `parameters()` method:
```
x = Neuron(2) #A neuron accepting a 2D input
x.parameters() #Output: [Scalar(data=-0.04005001083775157, gradient=0.0), Scalar(data=-0.6467923074411892, gradient=0.0), Scalar(data=0.05844456602120592, gradient=0.0)] -> These are random numbers between 1 and negative 1, the first 2 are weights, and the last element is the bias.
```
- Notice how the parameters are also scalars. This may be important if you decide to implement your own squashing function!

- Currently this library supports two squashing functions, tanh, sigmoid, and ReLU

- Engine.py contains a train_mlp function (As well as the Neuron, Layer, and MLP classes), which accepts the following inputs:
  1. model: The model you wish to train (of type MLP)
  2. Inputs: Must be a native python array, unfortunately it doesn't support numpy arrays (yet)
  3. Outputs: Same condition as inputs
  4. Batching Size (__NEW__): You can batch your training data together, and weights will only update per batch as opposed to per sample.
  5. Epochs: The number of times you want to adjust the weights.
  6. Learning rate: The factor by which the weights are changed per epoch.
  - train_mlp currently only supports binary cross entropy for loss functions, I'll add more as soon as I fix example.py, which currently sucks at learning unfortunately (any help would be appreciated!)

Thanks for taking a look at my project! 
