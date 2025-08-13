import random
from kernel import Scalar

class Neuron:
    def __init__(self, dimNeurons, activation='relu'):
        self.weights = [Scalar(random.uniform(-1,1)) for _ in range(dimNeurons)]
        self.bias = Scalar(random.uniform(-1,1))
        self.activation = activation

    def __call__(self, x):
        # Dot the inputs to the neuron with weights, and add bias
        activation = sum((wi*xi for wi, xi in zip(self.weights, x)), self.bias)
        
        # Apply activation function
        if self.activation == 'relu':
            return activation.relu()
        elif self.activation == 'sigmoid':
            return activation.sigmoid()
        elif self.activation == 'tanh':
            return activation.tanh()
        else:
            return activation  # linear activation
    
    def parameters(self):
        return self.weights + [self.bias]
    
class Layer:
    def __init__(self, dimNeurons, numNeurons, activation='relu'):
        # Initialize a list of n-dimensional neurons where n = dimNeurons
        self.neurons = [Neuron(dimNeurons, activation) for _ in range(numNeurons)]

    def __call__(self, x):
        outputs = [n(x) for n in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs
    
    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params

class MLP:
    def __init__(self, dimNeurons, numNeurons, hidden_activation='relu', output_activation='sigmoid'):
        layerSize = [dimNeurons] + numNeurons
        self.layers = []
        
        # Create hidden layers with specified activation
        for i in range(len(numNeurons) - 1):
            self.layers.append(Layer(layerSize[i], layerSize[i+1], hidden_activation))
        
        # Create output layer with output activation (sigmoid for binary classification)
        if len(numNeurons) > 0:
            self.layers.append(Layer(layerSize[-2], layerSize[-1], output_activation))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

def mean_squared_error(y_true, y_pred):
    diff = y_pred - y_true
    return diff * diff

def train_mlp(model, X, y, batch_size, epochs, learning_rate, loss_fn=mean_squared_error):
    for epoch in range(epochs):
        total_loss = Scalar(0)
        num_batches = 0
        
        # Create mini-batches
        indices = list(range(len(X)))
        random.shuffle(indices)
        
        for start_idx in range(0, len(X), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            batch_X = [X[i] for i in batch_indices]
            batch_y = [y[i] for i in batch_indices]
            
            # Clear gradients before forward pass
            for p in model.parameters():
                p.grad = 0.0
            
            batch_loss = Scalar(0)
            
            # Forward pass
            for x, y_true in zip(batch_X, batch_y):
                y_pred = model(x)
                loss = loss_fn(y_true, y_pred)
                batch_loss = batch_loss + loss
            
            # Compute average batch loss
            batch_loss = batch_loss * (Scalar(1.0) / len(batch_indices))
            
            # Backward pass
            batch_loss.backward()
            
            # Update parameters
            for p in model.parameters():
                if p.grad is not None:
                    p.data -= learning_rate * p.grad
            
            total_loss = total_loss + batch_loss
            num_batches += 1
        
        # Compute average epoch loss
        avg_epoch_loss = total_loss * (Scalar(1.0) / num_batches)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss.data:.6f}")

# Example usage for XOR problem
if __name__ == "__main__":
    # XOR dataset
    X = [
        [Scalar(0), Scalar(0)],
        [Scalar(0), Scalar(1)],
        [Scalar(1), Scalar(0)],
        [Scalar(1), Scalar(1)]
    ]
    y = [Scalar(0), Scalar(1), Scalar(1), Scalar(0)]
    
    # Create model: 2 inputs -> 4 hidden (ReLU) -> 1 output (Sigmoid)
    model = MLP(2, [4, 1], hidden_activation='relu', output_activation='sigmoid')
    
    print("Training with Mean Squared Error:")
    train_mlp(model, X, y, batch_size=4, epochs=1000, learning_rate=0.1, loss_fn=mean_squared_error)
    
    # Test the trained model
    print("\nTesting trained model:")
    for i, x in enumerate(X):
        pred = model(x)
        print(f"Input: {[xi.data for xi in x]}, Target: {y[i].data}, Prediction: {pred.data:.4f}")