import numpy as np

x = np.random.uniform(100,30,1000)
y = np.random.uniform(400,10,1000)

polynomial = np.polyfit(x,y,20)
print(np.poly1d(polynomial))

from sklearn.metrics import r2_score

r2 = r2_score(y,np.poly1d(polynomial)(x))
print(r2)