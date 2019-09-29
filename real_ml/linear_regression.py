data = [
    [1,1],
    [2,2],
    [3,3],
    [4,4]]

# hypothesis function: theta_0 + theta_1 * x

theta_0 = 1
theta_1 = 0

learning_rate = 0.1

subtract_0 = 1
subtract_1 = 1

while abs(subtract_0) > 0.000000000001 or abs(subtract_1) > 0.000000000001:
    subtract_0 = 0
    subtract_1 = 0
    for x,y in data:
        subtract_0 += (1/(len(data))) * (theta_0 + theta_1 * x - y)
        subtract_1 += (1/(len(data))) * (theta_0 + theta_1 * x - y) * x

    print('subtract_0: ' + str(subtract_0))
    print('subtract_1: ' + str(subtract_1))

    theta_0 -= learning_rate * subtract_0
    theta_1 -= learning_rate * subtract_1

print('theta_0: ' + str(theta_0))
print('theta_1: ' + str(theta_1))