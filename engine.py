import random
from kernel import Scalar

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
    
def binary_cross_entropy(y_true, y_pred):
    return -y_true * y_pred.log() - (1 - y_true) * (1 - y_pred).log()

#Inputs: MLP, inputs, outputs, batch size, epochs, learning rate
def train_mlp(model, X, y, batch_size, epochs, learning_rate):
    for epoch in range(epochs):
        epoch_loss = Scalar(0)
        
        # Create mini-batches
        indices = list(range(len(X)))
        random.shuffle(indices)
        
        for start_idx in range(0, len(X), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            batch_X = [X[i] for i in batch_indices]
            batch_y = [y[i] for i in batch_indices]
            
            batch_loss = Scalar(0)
            
            # Forward pass
            for x, y_true in zip(batch_X, batch_y):
                y_pred = model(x)
                loss = binary_cross_entropy(y_true, y_pred)
                batch_loss = batch_loss + loss
            
            # Compute average batch loss
            batch_loss = batch_loss * (1.0 / len(batch_indices))
            
            # Backward pass
            for p in model.parameters():
                p.grad = 0
            batch_loss.backward()
            
            # Update parameters
            for p in model.parameters():
                p.data -= learning_rate * p.grad
            
            epoch_loss = epoch_loss + batch_loss
        
        # Compute average epoch loss
        epoch_loss = epoch_loss * (1.0 / len(X))
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss.data}")