import numpy as np
import matplotlib.pyplot as plt

# Using train/test to prevent overfitting with polynomial regression

np.random.seed(2)

# Sets of data
page_speeds = np.random.normal(3,1,100)
purchase_amounts = np.random.normal(50,30,100) / page_speeds

plt.scatter(page_speeds,purchase_amounts)
plt.show()

# Let us split the data into training and testing data
# 80 % for training 20 % for testing
trainX = page_speeds[:80]
testX = page_speeds[80:]

trainY = purchase_amounts[:80]
testY = purchase_amounts[80:]

# Training data
plt.scatter(trainX,trainY)
plt.show()

poly = np.polyfit(trainX,trainY,8) # Degree of 8 to simulate overfitting
poly = np.poly1d(poly)

# Let's use the r2_score function to measure the accuracy of polynomial
from sklearn.metrics import r2_score

r2 = r2_score(testY,poly(testX))
print(f'R2 Score: {r2}') # R2 Score: 0.3001816861129213 // VEry low

# Let's compare this with the R2 Score from training data
r2 = r2_score(trainY,poly(trainX))
print(f'R2 Score: {r2}') # R2 Score: 0.6427069514692472 // Matches training data better

# PRACTICE
page_speeds = np.random.normal(3,1,100)
purchase_amounts = np.random.normal(101,34,100) * page_speeds / 4

trainX = page_speeds[:60] # First 60 %
testX = page_speeds[60:] # Last 40 %
trainY = purchase_amounts[:60] 
testY = purchase_amounts[60:]

polynomial = np.polyfit(trainX,trainY,5) # Let's give a degree of 5 for example
polynomial = np.poly1d(polynomial)

from sklearn.metrics import r2_score

r2 = r2_score(testY,polynomial(testX))
print(f'R2 Score: {r2}') # R2 Score: 0.5288523764023845

# More Practice

books_read = np.random.normal(100,30,2000)
salary = np.random.normal(10,20,2000)

trainX = books_read[:1600] # EVerything before 1600
trainY = salary[:1600]

testX = books_read[1600:]
testY = books_read[1600:]

# Polynomial Regression
poly = np.polyfit(trainX,trainY,10)
poly = np.poly1d(poly)

print(f'Poly: {poly}')

from sklearn.metrics import r2_score

r2 = r2_score(trainY,poly(trainX))
print(f'R2: {r2}') # 0.003400625390954315

