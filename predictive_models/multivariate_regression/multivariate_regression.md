# Multivariate Regression

## Preface

Multivariate Regression is a method of predicting values of outputs given multiple input variables with unknown weights of importance. This form of regression assumes that the input variables have no dependence on each other.

## Libraries Needed

* Pandas
* Statsmodel
* Sklearn

## Scenario

Given data on multiple variable aspects of cars including its mileage, make, model, number of doors, etc., we will be creating a model to determine the prices of cars based on changes to such variables.

## Code

We must first import the necessary libraries:

```Python
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
```

Let's read in data from an excel file containing the data we need into a Pandas *dataframe* object.

```Python
df = pandas.read_excel('path/cars.xls')
print(df.head())
#          Price  Mileage   Make    Model  ... Doors Cruise  Sound  Leather
#0  17314.103129     8221  Buick  Century  ...     4      1      1        1
#1  17542.036083     9135  Buick  Century  ...     4      1      1        0
#2  16218.847862    13196  Buick  Century  ...     4      1      1        0
#3  16336.913140    16342  Buick  Century  ...     4      1      0        0
#4  16339.170324    19832  Buick  Century  ...     4      1      0        1
```

Let's try predicting the price of cars using 3 of these features, for this example, let's use *mileage*, *cylinder*, and *doors*. Try to remember that for regression, we can only use numerical data, not categorical data. We will first extract our data, with the *y-value*, or the output, being the price of the car. Let's call this variable *y*. We would also like to extract the features we are using for predictions:

```Python
X = df[['Mileage','Cylinder','Doors']] # What we will use to predict
y = df['Price'] # What we want to predict
```

In order to get accurate information from our features, we need to scale our data points so that they fall into the range of -1 to 1. Doing so ensures that some features do not have a greater effect on the outcome than other features. We will do this by using the StandardScaler function in the Sklearn.preprocessing library. By calling this function, a scaler is returned to help us scale our data.

```Python
scaler = StandardScaler()
```

We will now use its fit_transform() method, which essentially fits our data to the scale between -1 and 1, as well as transforms it into a useable dataframe for manipulation.

```Python
X[['Mileage','Cylinders','Doors']] = scaler.fit_transform(X)
```

We keep the three feature names to allow for the dataframe returned to have those headings. We will now like to create a model from this data, specifically, the Ordinarily Least Squares model.

```Python
model = sm.OLS(y,X) # Takes in y-values and x-values to perform regression
model = model.fit() # This returns a form where we can get specific important information
summary = model.summary() # This lets us see the information
```
