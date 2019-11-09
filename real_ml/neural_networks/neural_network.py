import numpy as np
import math

# let's try to approximate y = x^2 + 5


# 3 layer neural network (1 input 1 hidden 1 output)
# 2 nodes in each layer

i1 = np.random.rand(1)
i2 = np.random.rand(1)
h1 = np.random.rand(1)
h2 = np.random.rand(1)
o1 = np.random.rand(1)
o2 = np.random.rand(1)

def sigmoid(x):
    return 1/(1 + math.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))






