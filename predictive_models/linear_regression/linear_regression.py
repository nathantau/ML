import numpy as np
import matplotlib.pyplot as plt

page_speeds = np.random.normal(50,20,1000)
purchase_amount = 3*(page_speeds + np.random.normal(30, 20, 1000))*0.3

plt.scatter(page_speeds, purchase_amount)
plt.show()

import scipy.stats as scistats

data = scistats.linregress(page_speeds,purchase_amount)
slope,intercept,rvalue,pvalue,err = data
print(data)

print(rvalue**2) # 0.9899 (Very close to 1, so very good line of best fit)

# We also have the slope and intercept now
