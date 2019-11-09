# Unsupervised learning
# Groups data into clusters that are similar
import numpy as np
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale


def create_clustered_data(n,k):
  random.seed(10)
  points_per_cluster = float(n/k)
  X = []
  for i in range(k):
    income_centroid = random.uniform(20000,200000)
    age_centroid = random.uniform(20,70)
    for j in range(int(points_per_cluster)): 
      X.append([np.random.normal(income_centroid,10000),np.random.normal(age_centroid,2)])
  X = np.array(X)
  return X

#data = create_clustered_data(100,5)
#model = KMeans(n_clusters=5)
#model = model.fit(scale(data))
#print(model.labels_)

print(create_clustered_data(100,4))