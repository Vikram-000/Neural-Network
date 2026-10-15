import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=4, hidden_size=16, output_size=2, weight_scale=0.5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.W1 = np.random.randn(input_size, hidden_size) * weight_scale
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * weight_scale
        self.b2 = np.zeros((1, output_size))

    def relu(self, Z):
        return np.maximum(0, Z)
    
    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, X):
        X = np.atleast_2d(X)
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        
        return self.A2

    def act(self, observation):
        probs = self.forward(observation)
        return int(np.argmax(probs))

    def clone(self):
        child = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        child.W1 = self.W1.copy()
        child.b1 = self.b1.copy()
        child.W2 = self.W2.copy()
        child.b2 = self.b2.copy()
        return child

    def mutate(self, mutation_rate=0.2, mutation_strength=0.3):
        for param in [self.W1, self.b1, self.W2, self.b2]:
            mask = np.random.rand(*param.shape) < mutation_rate
            param += mask * np.random.randn(*param.shape) * mutation_strength

    def relu_derivative(self, Z):
        return Z > 0

    def backward(self, X, y, learning_rate=0.01):
        X = np.atleast_2d(X)
        y = np.atleast_2d(y)
        m = X.shape[0]
        
        dZ2 = self.A2 - y
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def save(self, filepath):
        np.savez(filepath, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load(self, filepath):
        data = np.load(filepath)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']