import numpy as np 
import matplotlib.pyplot as plt

x_values = np.random.normal(10,40,1000)
y_values = np.random.normal(0,10,1000)

plt.scatter(x_values,y_values)
plt.show()

import scipy.stats as stats

line_of_best_fit = stats.linregress(x_values,y_values)
slope,intercept,r_val,p_val,err_val = line_of_best_fit

print(f'The line of best fit is {slope}x + {intercept}')

# Curve of best fit
# Will use same data as above

polynomial = np.polyfit(x_values,y_values,10)
polynomial = np.poly1d(polynomial)
print(polynomial)

from sklearn.metrics import r2_score

r2 = r2_score(y_values,polynomial(x_values))
print(r2)
