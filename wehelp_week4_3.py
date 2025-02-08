import numpy as np

class NeuralNetwork:
    def __init__(self, weight, bias):
        self.weight = [np.array(w) for w in weight]
        self.bias = [np.array(b) for b in bias]
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def binary_cross_entropy(self, predicted, expected):

        loss = -np.mean(expected * np.log(predicted) + (1 - expected) * np.log(1 - predicted))
        return loss

    def forward(self, inputs):

        activation = np.array(inputs)
        for i, (w, b) in enumerate(zip(self.weight, self.bias)):
            activation = np.dot(w, activation) + b
            if i < len(self.weight) - 1:
                activation = self.relu(activation)
            else:
                activation = self.sigmoid(activation)
        return activation.tolist()
    

weight = [
    [[0.5, 0.2],[0.6, -0.6]],
    [[0.8, -0.4],[0.5, 0.4],[0.3, 0.75]]
]

bias = [
    [0.3, 0.25],
    [0.6, 0.5, -0.5]
]

nn = NeuralNetwork(weight, bias)

inputs_list = [
    [1.5, 0.5],
    [0, 1]
]

expects_list = [
    [1, 0, 1],
    [1, 1, 0]
]

print("Model 3:")
for inputs, expects in zip(inputs_list, expects_list):
    outputs = nn.forward(inputs)
    loss = nn.binary_cross_entropy(np.array(outputs), np.array(expects))
    print(f"{inputs} -> Total Loss: {loss}")
