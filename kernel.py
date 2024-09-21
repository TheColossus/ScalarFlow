import math

class Scalar :

    def __init__(self, data, parents=()):
        #Every Scalar has a value and a corresponding gradient
        self.data = data
        self.grad = 0.0

        #Store the parents of each scalar in an unordered set
        self.prev = set(parents)

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
        out = Scalar(math.log(self.data), (self,))

        def _backward():
            self.grad += (1 / self.data) * out.grad
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
    
    
    def relu(self):
        out = Scalar(0 if self.data < 0 else self.data, (self,))

        def _backward():
            self.grad += (out.data > 0) * out.grad
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