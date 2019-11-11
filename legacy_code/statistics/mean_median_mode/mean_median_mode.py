import numpy as np 
import matplotlib.pyplot as plt

# Create a numpy list centered around 27000 w standard dev of 15000 with 10000 data points
incomes = np.random.normal(27000,15000,10000)
print(incomes)

# Getting the mean of a numpy array
mean = np.mean(incomes)
print(f'The mean is {mean}')
median = np.median(incomes)
print(f'The median is {median}')

# Let's add an outlier and see how the mean and median are affected
incomes = np.append(incomes, [1000000000000])
mean = np.mean(incomes)
print(f'The new mean is {mean}') # Greatly affected
median = np.median(incomes)
print(f'The new median is {median}')

# Mode
ages = np.random.randint(18,high=90,size=500)

# Use SciPy
from scipy import stats
mode = stats.mode(ages)
print(f'The mode is {mode[0][0]}') #The mode is ModeResult(mode=array([32]), count=array([14]))
















# We can then plot this:
# plt.hist(incomes, 50) # histogram
# plt.show()





#plt.hist(incomes,50)
#plt.show()
