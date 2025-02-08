import numpy as np

class NeuralNetwork:
    def __init__(self, weight, bias):
        self.weight = [np.array(w) for w in weight]                                               # 將 weight & bias 轉為 numpy 
        self.bias = [np.array(b) for b in bias]

    def relu(self, x):                                                                            # relu公式
        return np.maximum(0, x)
    
    def sigmoid(self, x):                                                                         # sigmoid公式
        return 1 / (1 + np.exp(-x))
    
    def binary_cross_entropy(self, predicted, expected):                                          # binary cross entropy 公式

        loss = -np.mean(expected * np.log(predicted) + (1 - expected) * np.log(1 - predicted))
        return loss
    
    def forward(self, inputs):
        activation = np.array(inputs)
        for i, (w, b) in enumerate(zip(self.weight, self.bias)):
            activation = np.dot(w, activation) + b
            if i < len(self.weight) - 1:                                                          # 隱藏層使用Relu
                activation = self.relu(activation)
            else:                                                                                 # 最後輸出層使用sigmoid 
                activation = self.sigmoid(activation)
        return activation.tolist()

weight = [
    [[0.5, 0.2], [0.6, -0.6]],  
    [[0.8, 0.4]] 
]

bias = [
    [0.3, 0.25],
    [-0.5]
]

nn = NeuralNetwork(weight, bias)

inputs_list = [
    [0.75, 1.25],
    [-1, 0.5]
]

expects_list = [
    1,
    0
]

print("Model 2:")
for i, (inputs, expects) in enumerate(zip(inputs_list, expects_list)):
    outputs = nn.forward(inputs)
    loss = nn.binary_cross_entropy(np.array(outputs[0]),np.array(expects))
    print(f"{inputs} -> Total Loss: {loss}")