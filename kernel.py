import math
import random

class Scalar :

    def __init__(self, data, children=()):
        #Every Scalar has a value and a corresponding gradient
        self.data = data
        self.grad = 0.0

        #Store the parents of each scalar in an unordered set
        self.prev = set(children)

        #A function that computes the chain rule during backpropagation
        self.backward = lambda : None

    def __add__(self, other):
        out = Scalar(self.data + other.data, (self, other))

        def backward():
            self.grad += out.grad
            other.grad += out.grad
        out.backward = backward
        return out;
    
    def __mul__(self, other):
        out = Scalar(self.data * other.data, (self, other))

        def backward():
            self.grad += other.grad * out.grad
            other.grad += self.grad * out.grad
        out.backward = backward

        return out
    
    def tanh(self):
        out = Scalar(math.tanh(self.data), (self))

        def backward():
            self.grad = (1-math.tanh(self.data)**2) * out.grad
        out.backward = backward
        return out
    
    def backward(self):
        #Order the nodes from right to left via a topological sort done DFS style
        #Recursively start at the root node and go backwards, appending each node's children before itself, to ensure topological order
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        #Set the output node's gradient to 1, since for any function x, dx/dx is 1
        self.grad = 1.0

        #You need to start backpropagation at the output node, so reverse the topological sort then go backwards
        for node in reversed(topo):
            node.backward()


    def __repr__(self):
        return f"Scalar(data={self.data}, gradient={self.grad})"
    
class Neuron:

    def __init__(self, dimNeurons):
        self.weights = [Scalar(random.uniform(-1,1) for input in dimNeurons)]
        self.bias = Scalar(random.uniform(-1,1))

    def __call__(self, x):
        #Dot the inputs to the neuron with one another, and add a bias
        activation = sum((wi*xi for wi, xi in zip(self.weights, x)), self.bias)
        
        #Squash the output so that it is between 0 and 1.
        return activation.tanh()
    
class Layer:

    def __init__(self, dimNeurons, numNeurons):
        #Initialize a list of n-dimensional neurons where n = dimNeurons
        self.neurons = [Scalar(dimNeurons) for neuron in range(numNeurons)]

    def __call__(self, x):
        Layer = [n(x) for n in self.neurons]

        return Layer
    
class MLP:

    def __init__(self, dimNeurons, numNeurons):
        layerSize = [dimNeurons] + numNeurons
        self.layers = [Layer(layerSize[i], layerSize[i+1]) for i in range(len(numNeurons))] 

    def __call__(self, x):
        for layer in self.layers:
            return layer(x)