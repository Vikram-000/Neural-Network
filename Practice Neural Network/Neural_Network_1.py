import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        W1 = np.random.rann(input_size, hidden_size) *0.01
        b1 = np.zeros((1, hidden_size))

        W2 = np.random.rann(hidden_size, output_size) *0.01
        b2 = np.zeros((1, output_size))

    def relu(self):
        return np.maximum(0,Z)
    
    def softmax(self, Z):
            exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
            return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        
        return self.A2