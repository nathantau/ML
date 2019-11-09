import numpy as np 
import matplotlib.pyplot as plt

#                              ^ Standard Deviation
incomes = np.random.normal(100,1000,10000)

# Finding standard deviation of data set
standard_deviation = incomes.std()
print(standard_deviation) # 19.857 

# Finding variance of data set
variance = incomes.var()
print(variance) # 395.26795703482384 (std squared)

plt.hist(incomes,50)
plt.show()

