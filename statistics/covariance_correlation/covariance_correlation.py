import numpy as np
import matplotlib.pyplot as plt

def diff_from_mean(dataset):
  mean = np.mean(dataset)
  return [mean - data_point for data_point in dataset]

def covariance(set1, set2):
  length = len(set1)
  return np.dot(diff_from_mean(set1),diff_from_mean(set2)) / (length-1)

page_speeds = np.random.normal(3,1,1000)
purchase_amount = np.random.normal(50,10,1000)

plt.scatter(page_speeds,purchase_amount)
# plt.show()

print(covariance(page_speeds,purchase_amount)) # 0.1579 -> Very close to 0, so we see that there's no real relationshp between values

# Now, let's make purchase amounts a function of pagespeeds
purchase_amount = np.random.normal(50,10,1000) / page_speeds

plt.scatter(page_speeds,purchase_amount)
# plt.show()

print(covariance(page_speeds,purchase_amount)) # 8.40 -> High, so we see that there's no real relationshp between values

def correlation(set1, set2):
  std1 = np.std(set1)
  std2 = np.std(set2)
  return covariance(set1,set2)/std1/std2

print(correlation(purchase_amount,page_speeds)) # -0.399

# This tells us that there is some correlation (between 0 and -1)

# Numpy correlation
numpy_correlation = np.corrcoef(page_speeds,purchase_amount)
print(numpy_correlation) # Returns a matrix of all possible combinations 

# Now, let's force a good correlation
purchase_amount = 100 - 3*page_speeds # Linear

numpy_correlation = np.corrcoef(page_speeds,purchase_amount)
print(numpy_correlation)
# 1,-1,-1,1 (Perfect inverse correlation)

# Activity
# numpy covariance
numpy_covariance = np.cov(page_speeds,purchase_amount)
print(numpy_covariance)