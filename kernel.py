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
        self._backward = lambda : None

    def __add__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data + other.data, (self, other))

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out    


    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Scalar(self.data**other, (self,))

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out
    
    def log(self):
        out = Scalar(math.log10(self.data + 1e-7), (self,))

        def _backward():
            self.grad = (1/((self.data + 1e-7)*math.log(10))) * out.grad
        out._backward = _backward
    
        return out
    
    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def tanh(self):
        out = Scalar(math.tanh(self.data), (self,))

        def _backward():
            self.grad += (1-math.tanh(self.data)**2) * out.grad
        out._backward = _backward
        return out
    
    def sigmoid(self):
        result = 1/(1 + math.exp(self.data))
        out = Scalar(result, (self,))
        
        def _backward():
            self.grad += (result*(1-result)) * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        #Order the nodes from right to left via a topological sort done DFS style
        #Recursively start at the root node and go backwards, appending each node's children before itself, to ensure topological order
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        #Set the output node's gradient to 1, since for any function x, dx/dx is 1
        self.grad = 1.0

        #You need to start backpropagation at the output node, so reverse the topological sort then go backwards
        for node in reversed(topo):
            node._backward()


    def __repr__(self):
        return f"Scalar(data={self.data}, gradient={self.grad})"
    
class Neuron:

    def __init__(self, dimNeurons):
        self.weights = [Scalar(random.uniform(-1,1)) for _ in range(dimNeurons)]
        self.bias = Scalar(random.uniform(-1,1))

    def __call__(self, x):
        #Dot the inputs to the neuron with one another, and add a bias
        activation = sum((wi*xi for wi, xi in zip(self.weights, x)), self.bias)
        
        #Squash the output so that it is between 0 and 1.
        return activation.sigmoid()
    
    def parameters(self):
        return self.weights + [self.bias]
    
class Layer:

    def __init__(self, dimNeurons, numNeurons):
        #Initialize a list of n-dimensional neurons where n = dimNeurons
        self.neurons = [Neuron(dimNeurons) for _ in range(numNeurons)]

    def __call__(self, x):
        Layer = [n(x) for n in self.neurons]
        return Layer[0] if len(Layer) == 1 else Layer
    
    def parameters(self):
        params = []
        for neuron in self.neurons:
            ps = neuron.parameters()
            params.extend(ps)
        return params

    
class MLP:

    def __init__(self, dimNeurons, numNeurons):
        layerSize = [dimNeurons] + numNeurons
        self.layers = [Layer(layerSize[i], layerSize[i+1]) for i in range(len(numNeurons))] 

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
    def parameters(self):
        params = []
        for layer in self.layers:
            ps = layer.parameters()
            params.extend(ps)
        return params