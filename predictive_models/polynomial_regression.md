# Polynomial Regression

## Preface

We will be using polynomial regression to create polynomial functions given certain data sets, which will allow us to predict new values.

## Libraries Needed

* Numpy
* Sklearn

## Scenario

We will continue to use the scenario of page speeds and purchase amounts, as covered in the experiment with linear regression.

## Code

Let's import our libraries:

```Python
import numpy as np
from sklearn.metrics import r2_score
```

Now, let's create our datasets:

```Python
page_speeds = np.random.normal(3,5,1000)
purchase_amounts = np.random.normal(3,1,1000)/page_speeds
```

In order to perform polynomial regression, we must use the *polyfit* function in the Numpy library. This function takes in the parameters (x_values,y_values,polynomial_degree), and requires you to give a good estimate of what the degree of the polynomial you want returned. It returns a Numpy array of the coefficients of the polynomial that Numpy has determined.

```Python
polynomial = np.polyfit(page_speeds,purchase_amounts,10) # Estimate that polynomial is of degree 10
# [  -2.99642872   36.89414652 -153.72793857  239.52918098  -77.24537323]
```

We can then convert this polynomial into a form which allows for it to take in data points as input (x-values) and produce an output (y-values).

```Python
polynomial = np.poly1d(polynomial)
#         4         3         2
# -2.996 x + 36.89 x - 153.7 x + 239.5 x - 77.25
```

We have now created a model of the polynomial, and want to test its accuracy by using Sklearn's r2_score function to determine its r-value squared. This function takes in the y-values that are in the data set, followed by the y-values that your function has predicted, and determines how accurate the model is.

```Python
r2 = r2_score(purchase_amounts,polynomial(page_speeds))
```