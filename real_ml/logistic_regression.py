import numpy as np
import math

class LogisticRegressionModel():

    def __init__(self, dimensions):
        self.W = np.random.rand(dimensions)

        print(self.W)

    def sigmoid(self, x):
        return 1/(1 + math.exp(-x))

    def hypothesis(self, x):
        return self.sigmoid(self.W.dot(x))

    def cost_function(self, hypothesis, y):
        if y == 0:
            return -1 * math.log(hypothesis)
        elif y == 1:
            return -1 * math.log(1 - hypothesis)
    



l = LogisticRegressionModel(3)