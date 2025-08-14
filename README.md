# ScalarFlow

ScalarFlow is a minimalist neural network library for scalar-valued inputs, inspired by TensorFlow and PyTorch. It is heavily based on [micrograd](https://github.com/karpathy/micrograd/) and was created as a learning tool to explore backpropagation, gradient descent, and model evaluation.  

---

## Features

### Scalar class (`kernel.py`)
Represents a scalar value with automatic gradient tracking for backpropagation. Supports:  

- Arithmetic: `+`, `-`, `*`, `/`, exponentiation (ints/floats)  
- Natural logarithm: `log()`  
- Keeps track of parent nodes for automatic differentiation  
- `.backward()` to compute gradients  

**Example:**  
```
x = Scalar(4)
y = Scalar(5)
z = x + y         # Scalar(9.0)
z.prev            # {Scalar(4.0), Scalar(5.0)}
z.backward()      # computes gradients
```

### Multi-Layer Perceptrons (MLPs)
Easily initialize MLPs with arbitrary input dimensions and hidden layers:
```
model = MLP(13, [20, 20, 1], hidden_activation='relu', output_activation='sigmoid')  # 13 inputs, 2 hidden layers of 20 neurons, 1 output. ReLU activation for hidden layers,                                                                                      # and sigmoid activation for the output layer
```
Access parameters of neurons, layers, or the entire model:
```
neuron = Neuron(2)
neuron.parameters()  # returns [w1, w2, bias] as Scalars
```

#### Activation Functions
Currently supports `ReLU`, `Sigmoid`, `Tanh`

#### Training (`engine.py`)
- `train_mlp()` supports batch training with Python-native array inputs
- Adjustable batch size, epochs, learning rate, and learning rate decay
- Learning rate decay can be exponential, step-wise, cosine annealing, or linear
- Currently supports mean squared error loss

Example:
```
train_mlp(
    model, x_train, y_train,
    batch_size=32,
    epochs=100,
    learning_rate=0.01,
    loss_fn=mean_squared_error,
    lr_decay_type='exponential',
    lr_decay_rate=0.90,
    lr_decay_step=10,
    lr_min=1e-6
)
```
### Notes
- Inputs and outputs must be Python arrays; NumPy arrays are not yet supported.
- Only scalar operations are supported; no vectorized operations.

### Testing
- View the jupyter notebook for an example using the classic heart disease dataset. Scalarflow achieved an accuracy of 79%. Using pretty much the same implementation with Pytorch also yielded an 79% accuracy (Although binary cross entropy was used instead of mean squared error).


Thanks for checking out ScalarFlow! Contributions and suggestions are welcome.
