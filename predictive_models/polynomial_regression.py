import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
page_speeds = np.random.normal(3,1,1000)
purchase_amount = np.random.normal(50,10,1000)/page_speeds

plt.scatter(page_speeds,purchase_amount)
plt.show()

# LEt's say we want to get a polynomial of best fit with degree of 4
poly4 = np.polyfit(page_speeds,purchase_amount,10) # Now using degree of 10
print(poly4) # [  -2.99642872   36.89414652 -153.72793857  239.52918098  -77.24537323]
                        #         4         3         2
print(np.poly1d(poly4)) # -2.996 x + 36.89 x - 153.7 x + 239.5 x - 77.25

# Time to measure r^2
from sklearn.metrics import r2_score
r2 = r2_score(purchase_amount,np.poly1d(poly4)(page_speeds))
print(r2) # 0.08363928880074978 when degree is 3, 0.8999013605079299 when degree is 10

# Confirming theory
a = np.array([1,2,3])
b = np.array([1,2,3])
c = a/b
print(c)