import numpy as np
import matplotlib.pyplot as plt

# Uniform distribution - Each value has equal chance of occurring
values = np.random.uniform(-10,10,100000) # Low,High,Population
plt.hist(values,50)
plt.show()

# Normal/Gaussian distribution
values = np.random.normal(100,10,1000) # Center,Std.Dev,Population
plt.hist(values,50)
plt.show()

# Exponential Probability Density Funciton
from scipy.stats import expon
x = np.arange(0,10,0.001)
plt.plot(x,expon.pdf(x))

# Poisson Probability Mass Function
from scipy.stats import poisson
mu = 500
x = np.arange(400,600,0.5)
plt.plot(x,poisson.pmf(x,mu))