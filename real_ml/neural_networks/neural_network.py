import numpy as np
import math

def relu(x):
    if x < 0: return 0
    else: return x

def relu_derivative(x):
    if x < 0: return 0
    else: return 1

def sigmoid(x):
    return 1/(1 + math.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# let's try to approximate z = x^2 + y^2

np.random.seed(42)

input_vector = np.array([[2],[3]])

expected_output = 13

learning_rate = 0.001

# 3 layer neural network (1 input 1 hidden 1 output)
# 2 nodes in each layer except for last

hidden1_matrix = np.random.rand(2, 2)
output1_matrix = np.random.rand(1, 2)

bias1_matrix = np.random.rand(2, 1)
bias2_matrix = np.random.rand(1)

print(hidden1_matrix)
print(bias2_matrix)
# print(input_vector)
print('output1_matrix', output1_matrix)

for epoch in range(100):

    # forward pass
    
    # print(hidden1_matrix[0], 'multiplied by', input_vector)

    hidden1_output1 = hidden1_matrix[0].dot(input_vector) + bias1_matrix[0]
    hidden1_output1 = sigmoid(hidden1_output1)

    hidden1_output2 = hidden1_matrix[1].dot(input_vector) + bias1_matrix[1]
    hidden1_output2 = sigmoid(hidden1_output2)

    # print('hidden o1', hidden1_output1)
    # print('hidden o2', hidden1_output2)

    new_input_vector = np.array([[hidden1_output1],[hidden1_output2]])

    print(output1_matrix[0])
    # print('multiplied by', new_input_vector)

    output = output1_matrix[0].dot(new_input_vector) + bias2_matrix[0]
    actual_output = relu(output)
    print('hidden layer', hidden1_matrix)
    print('output layer', output1_matrix)

    print('your output is' , output)
    print('your actual output is' , actual_output)

    # we have now finished completing forward pass, time to compute loss and grad. descent

    loss = 0.5 * (expected_output - actual_output)**2
    print('your loss is', loss)

    # now, time to do back prop

    output1_matrix_grad = np.random.rand(1,2)
    bias2_matrix_grad = np.random.rand(1)
    bias1_matrix_grad = np.random.rand(2,1)

    output1_matrix_grad[0][0] = (expected_output - actual_output) * relu_derivative(output) * output1_matrix[0][0]
    output1_matrix_grad[0][1] = (expected_output - actual_output) * relu_derivative(output) * output1_matrix[0][1]
    bias2_matrix_grad = (expected_output - actual_output) * relu_derivative(output)


    # print('gradient for output1_matrix', output1_matrix_grad)
    # print('gradient for bias2_matrix', bias2_matrix_grad)

    # grad for output layer computed, time for hidden layer!

    hidden1_matrix_grad = np.random.rand(2, 2)
    
    hidden1_matrix_grad[0][0] = (expected_output - actual_output) * relu_derivative(output) * output1_matrix[0][0] * sigmoid(hidden1_output1) * (1 - sigmoid(hidden1_output1)) * input_vector[0][0]
    hidden1_matrix_grad[0][1] = (expected_output - actual_output) * relu_derivative(output) * output1_matrix[0][0] * sigmoid(hidden1_output1) * (1 - sigmoid(hidden1_output1)) * input_vector[1][0]
    hidden1_matrix_grad[1][0] = (expected_output - actual_output) * relu_derivative(output) * output1_matrix[0][1] * sigmoid(hidden1_output2) * (1 - sigmoid(hidden1_output2)) * input_vector[0][0]
    hidden1_matrix_grad[1][1] = (expected_output - actual_output) * relu_derivative(output) * output1_matrix[0][1] * sigmoid(hidden1_output2) * (1 - sigmoid(hidden1_output2)) * input_vector[1][0]
    bias1_matrix_grad[0][0] = (expected_output - actual_output) * relu_derivative(output) * output1_matrix[0][1] * sigmoid(hidden1_output1) * (1 - sigmoid(hidden1_output1))
    bias1_matrix_grad[1][0] = (expected_output - actual_output) * relu_derivative(output) * output1_matrix[0][1] * sigmoid(hidden1_output2) * (1 - sigmoid(hidden1_output2))

    # grad for hidden layer computed

    # now time to subtract grads from matrices!

    bias1_matrix -= bias1_matrix_grad * learning_rate
    bias2_matrix -= bias2_matrix_grad * learning_rate
    hidden1_matrix -= hidden1_matrix_grad * learning_rate
    output1_matrix -= output1_matrix_grad * learning_rate



    