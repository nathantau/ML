import numpy as np
import math

def sigmoid(x):
    return 1/(1 + math.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# let's try to approximate z = x^2 + 5y + 4

input1 = np.array(2)
input2 = np.array(3)

# expected output = 23

# 3 layer neural network (1 input 1 hidden 1 output)
# 2 nodes in each layer

i1 = np.random.rand(1)
i2 = np.random.rand(1)
h1 = np.random.rand(1)
h2 = np.random.rand(1)
o1 = np.random.rand(1)
o2 = np.random.rand(1)

# for now let's not use any biases

for epoch in range(100):

    # forward pass

    output = (input1 * i1)






