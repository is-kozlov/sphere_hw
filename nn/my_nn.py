import numpy as np
import pandas as pd


class Linear:
    def __init__(self, input_size, output_size):
        '''
        Creates weights and biases for linear layer.
        Dimention of inputs is *input_size*, of output: *output_size*.
        '''
        #### Create weights, initialize them with samples from N(0, 0.1).
        self.input_size = input_size
        self.output_size = output_size
        self.w = np.random.normal(0, 0.1, (self.input_size, self.output_size))
    
    def forward(self, X):
        '''
        Passes objects through this layer.
        X is np.array of size (N, d)
        '''
        #### Apply layer to input
        self.X = X.copy()
        self.X_intercept = np.c_[X, np.ones(X.shape[0])]
        return np.dot(self.X, self.w)
    
    def backward(self, dLdy):
        '''
        1. Compute dLdw and dLdx.
        2. Store dLdw for step() call
        3. Return dLdx
        '''
        self.dLdw = np.dot(self.X.T, dLdy)
        return np.dot(dLdy, self.w.T)
    
    def step(self, learning_rate):
        '''
        1. Apply gradient dLdw to network:
        w <- w - l*dLdw
        '''
        self.w -= learning_rate * self.dLdw




class Sigmoid:
    def __init__(self):
        pass
    
    def _sigmoid(self, x):
        return 1.0/(1 + np.exp(-x))
    
    def _derivative_sigmoid(self, x):
        return np.exp(-x) * (self._sigmoid(x)**2)
    
    def forward(self, X):
        '''
        Passes objects through this layer.
        X is np.array of size (N, d)
        '''
        #### Apply layer to input
        self.X = X.copy()
        return self._sigmoid(X)
    
    def backward(self, dLdy):
        '''
        1. Compute dLdx.
        2. Return dLdx
        '''
        return dLdy * self._derivative_sigmoid(self.X)
    
    def step(self, learning_rate):
        pass



class ReLU():
    def __init__(self):
        pass
    
    def _derivative_relu(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x
    
    def forward(self, X):
        '''
        Passes objects through this layer.
        X is np.array of size (N, d)
        '''
        self.X = X.copy()
        return np.maximum(0, X)
    
    def backward(self, dLdy):
        '''
        1. Compute dLdx.
        2. Return dLdx
        '''
        return dLdy * self._derivative_relu(self.X)
    
    def step(self, learning_rate):
        pass



class ELU():
    def __init__(self, alpha):
        self.alpha = alpha
        pass
    
    def _elu(self, x, alpha):
        x[x <= 0] = alpha * (np.exp(x[x <= 0]) - 1)
        x[x > 0] = x[x > 0]
        return x
    
    def _derivative_elu(self, x, alpha):
        x[x <= 0] = alpha * np.exp(x[x <= 0])
        x[x > 0] = 1
        return x
    
    def forward(self, X):
        '''
        Passes objects through this layer.
        X is np.array of size (N, d)
        '''
        self.X = X.copy()
        return self._elu(X, self.alpha)
    
    def backward(self, dLdy):
        '''
        1. Compute dLdx.
        2. Return dLdx
        '''
        return dLdy * self._derivative_elu(self.X, self.alpha)
    
    def step(self, learning_rate):
        pass




class NLLLoss:
    def __init__(self):
        '''
        Applies Softmax operation to inputs and computes NLL loss
        '''
        #### YOUR CODE HERE
        #### (Hint: No code is expected here, just joking)
    
    def forward(self, X, y):
        '''
        Passes objects through this layer.
        X is np.array of size (N, d)
        y is np.array of size (N), contains correct labels
        '''
        #### Apply layer to input
        self.y = y.copy()
        self.p = np.divide(np.exp(X).T, np.exp(X).sum(axis=1)).T
        return self.p, -np.log(self.p[y == 1]).sum()
    
    def backward(self):
        '''
        Note that here dLdy = 1 since L = y
        1. Compute dLdx
        2. Return dLdx
        '''
        dLdx = self.p.copy()
        dLdx[self.y == 1] -= 1
        return dLdx
    
    def step(self, learning_rate):
        pass





class NeuralNetwork:
    def __init__(self, modules, loss_layer, epochs, learning_rate=0.01):
        '''
        Constructs network with *modules* as its layers
        '''
        self.modules = modules
        self.loss_layer = loss_layer
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss_arr = []
    
    def forward(self, X):
        #### Apply layers to input
        self.X = X.copy()
        for i, layer in enumerate(self.modules):
            self.X = self.modules[i].forward(self.X)
        return self.X
        
    
    def backward(self, dLdy):
        '''
        dLdy here is a gradient from loss function
        '''
        self.dLdy = dLdy
        for layer in self.modules[::-1]:
            self.dLdy = layer.backward(self.dLdy)
            layer.step(self.learning_rate)
    
    def step(self, learning_rate):
        pass
    
    def fit(self, X, y):
        self.y = pd.get_dummies(pd.Series(y)).values
        for epoch in xrange(self.epochs):
            self.output = self.forward(X)
            self.output, loss = self.loss_layer.forward(self.output, self.y)
            self.loss_arr.append(loss.sum())
            dLdy = self.loss_layer.backward()
            self.backward(dLdy)
        return self
    
    def predict(self, X):
        return self.loss_layer.forward(self.forward(X), X)[0]

