import numpy as np

class NeuralNetwork:
    def __init__(self, weight, bias):
        self.weight = [np.array(w) for w in weight]                                               # 將 weight & bias 轉為 numpy 
        self.bias = [np.array(b) for b in bias] 
    
    def relu(self, x):                                                                            # relu公式
        return np.maximum(0, x)

    def mse_loss(self, predicted, expected):                                                      # MSE公式
        return np.mean((predicted - expected) ** 2)
    
    def forward(self, inputs):
        activation = np.array(inputs)
        for i, (w, b) in enumerate(zip(self.weight, self.bias)):
            activation = np.dot(w, activation) + b                                  
            if i < len(self.weight) - 1:                                                          # 隱藏層使用Relu
                activation = self.relu(activation)
        return activation.tolist()

weight = [                                                                                        # 輸入weight
    [[0.5, 0.2], [0.6, -0.6]],
    [[0.8, -0.5], [0.4, 0.5]]
]

bias = [                                                                                          # 輸入bias
    [0.3, 0.25],
    [0.6, -0.25]
]

nn = NeuralNetwork(weight, bias)

inputs_list = [ 
    [1.5, 0.5],
    [0, 1]
]

expects_list = [
    [0.8, 1],
    [0.5, 0.5]
]

print("Model 1:")

for inputs, expects in zip(inputs_list, expects_list):                                            #帶入input值及expect值
    outputs = nn.forward(inputs) 
    loss = nn.mse_loss(np.array(outputs), np.array(expects))
    print(f"{inputs} -> Total Loss: {loss}")
