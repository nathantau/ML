import numpy as np 
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import os
import io

def readFiles(path):
  for root, dirnames, filenames in os.walk(path):
      for filename in filenames:
        path = os.path.join(root, filename)

        inBody = False
        lines = []
        f = io.open(path, 'r', encoding='latin1')
        for line in f:
          if inBody:
            lines.append(line)
          elif line == '\n':
            inBody = True
        f.close()
        message = '\n'.join(lines)
        yield path, message

def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
      rows.append({'message': message, 'class': classification})
      index.append(filename)

df = DataFrame({'message':[],'class':[]})
df = df.append(dataFrameFromDirectory('c:/Users/Nathan/PythonProjects/machine_learning/predictive_models/spam','spam'),sort=True)
#df = df.append(dataFrameFromDirectory('C:\Users\Nathan\PythonProjects\machine_learning\predictive_models\ham','ham'),sort=True)

print(df)