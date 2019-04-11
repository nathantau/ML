import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler as scale 
scale = scale()

# Creates dataframe from excel file
df = pd.read_excel('predictive_models/cars.xls')

print(df.head())

#          Price  Mileage   Make    Model  ... Doors Cruise  Sound  Leather
#0  17314.103129     8221  Buick  Century  ...     4      1      1        1
#1  17542.036083     9135  Buick  Century  ...     4      1      1        0
#2  16218.847862    13196  Buick  Century  ...     4      1      1        0
#3  16336.913140    16342  Buick  Century  ...     4      1      0        0
#4  16339.170324    19832  Buick  Century  ...     4      1      0        1

# We will aim to predict price with 3 of these features
# There is some categorical data, we cannot use regression for it

X = df[['Mileage','Cylinder','Doors']] # Features that might affect price
y = df['Price'] # TO predict

# WE do this assignment instead of X = ... to preserve the dataframe structure

# X = ...
#[[-1.41748516  0.52741047  0.55627894]
# [-1.30590228  0.52741047  0.55627894]
# [-0.81012759  0.52741047  0.55627894]
# ...
# [ 0.07960546  0.52741047  0.55627894]
# [ 0.75044563  0.52741047  0.55627894]
# [ 1.93256489  0.52741047  0.55627894]]

# X[['Mileage','Cylinder','Doors']] = ...
#      Mileage  Cylinder     Doors
#0   -1.417485  0.527410  0.556279
#1   -1.305902  0.527410  0.556279
#2   -0.810128  0.527410  0.556279
#3   -0.426058  0.527410  0.556279
#4    0.000008  0.527410  0.556279
#5    0.293493  0.527410  0.556279
#..        ...       ...       ...
#774 -0.161262 -0.914896  0.556279
#775 -0.089234 -0.914896  0.556279

# Normalizes terms from -1 to 1
X[['Mileage','Cylinder','Doors']] = scale.fit_transform(X)
print(X)

# Ordinarily Squares Model
est = sm.OLS(y,X)
print(est.summary())

#                 coef    std err          t      P>|t|      [0.025      0.975]
#------------------------------------------------------------------------------
#Mileage    -1272.3412    804.623     -1.581      0.114   -2851.759     307.077
#Cylinder    5587.4472    804.509      6.945      0.000    4008.252    7166.642
#Doors      -1404.5513    804.275     -1.746      0.081   -2983.288     174.185

# CYlinders have the highest coefficient (abs value), so most impact on price

# Practice

# Reads data from excel spreadsheet into dataframe
#df = pd.read_excel('predictive_models/cars.xls')

# Get features
#X = df[['Cruise','Sound','Leather']]

# Get value to predict
#y = df['Price']

# Scaling and fitting data
#X[['Cruise','Sound','Leather']] = scale.fit_transform(X[['Cruise','Sound','Leather']].as_matrix())
#print(X)

#est = sm.OLS(y,X).fit() # Fit method gives access to summary function
#print(est.summary())

#                 coef    std err          t      P>|t|      [0.025      0.975]
#------------------------------------------------------------------------------
#Cruise      4293.2667    818.349      5.246      0.000    2686.904    5899.629
#Sound      -1173.7460    827.715     -1.418      0.157   -2798.493     451.001
#Leather     2050.0791    826.286      2.481      0.013     428.138    3672.020

# Cruise has the greatest impact on price

# A bit more practice!
#df = pd.read_excel('predictive_models/cars.xls')

# Features, this time they are categorical for the sake of experimentation
#X = df[['Make','Model']]

# The value (price) that we want to predict
#y = df['Price']

# Fitting the data and scaling it
#X[['Make','Model']] = scale.fit_transform(X[['Make','Model']].as_matrix()) # Not possible

# X['Cruise','Sound','Leather'] = scale.fit_transform(X['Cruise','Sound','Leather'])
# print('hello')
# print(X)

# Creating model (value to predict, training data)
# est = sm.OLS(y,X).fit()
# print(est.summary())

