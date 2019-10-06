import numpy as np

X = [
    [1,1,3],
    [1,2,4],
    [1,3,5],
    [1,4,6]]
y  = [1,2,3,4]

Xt = np.transpose(X)

# the normal equation - this assumes that the inverse exists.
theta = np.matmul(np.linalg.inv(np.matmul(Xt, X)), np.matmul(Xt, y))

print(theta)