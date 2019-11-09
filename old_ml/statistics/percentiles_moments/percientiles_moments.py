import numpy as np 
import matplotlib.pyplot as plt

# Center: 0 , Standard Dev. : 0.5, Points: 10000
vals = np.random.normal(0,0.5,10000)

# Plotting on histogram
# plt.hist(vals,50)
# plt.show()

# Getting percentile 
fiftiethPercentile = np.percentile(vals, 50) # 50th percentile value
print(fiftiethPercentile) # Very close to 0

ninetiethPercentile = np.percentile(vals, 90)
print(ninetiethPercentile)

# Moments
vals = np.random.normal(0,0.5,1000)

# Plotting on histogram
plt.hist(vals, 50)
plt.show()

# Mean (moment 1)
mean = np.mean(vals)
print(mean)

# Variance (moment 2)
variance = np.var(vals)
print(variance)

# Standard deviation
std = np.std(vals)
print(std)

# Skew (moment 3) - need SciPy package
import scipy.stats as sp 
skew = sp.skew(vals)
print(skew)

# Kurtosis (moment 4)
kurtosis = sp.kurtosis(vals)
print(kurtosis)
