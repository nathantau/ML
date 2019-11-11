import numpy as np 

# Create normal distribution
incomes = np.random.normal(100,20,10000)

# Find mean/median
mean = np.mean(incomes)
median = np.median(incomes) 
print(f'The mean is {mean}')
print(f'The median is {median}')

# Add more values to skew data/trends
incomes = np.append(incomes,[100000,100000,10000])

# Find mean/median
mean = np.mean(incomes)
median = np.median(incomes) 
print(f'The new mean is {mean}')
print(f'The new median is {median}')

# Finding mode of data
import scipy.stats
stats = scipy.stats
# from scipy import stats
mode = stats.mode(incomes)
print(f'The mode is {mode}')