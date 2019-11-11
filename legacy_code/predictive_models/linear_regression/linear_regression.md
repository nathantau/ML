# Linear Regression

## Preface

We will be using linear regression to create a model to determine, using a linear function, other data points given a certain set of data.

## Libraries Needed

* Numpy
* SciPy

## Scenario

We would like to simulate a relationship between the speed of pages loading on shopping websites and the purchase amount the customer is willing to pay.

## Code

Let's import our libraries:

```Python
import numpy as np
import scipy.stats as stats
```

We will first create 1000 data points (page_speeds, purchase_amount), which will be done using the Numpy library. We will follow a normal distribution in this example. As well, we will try to simulate a relationship between the purchase amounts and page speeds.

```Python
page_speeds = np.random.normal(50,20,1000) # Centre, Standard Deviation, Number of data points
purchase_amounts = 3*(page_speeds + np.random.normal(30,20,1000))
```

This will end up returning two Numpy arrays for us to manipulate, with a relationship occurring between them. We would now like obtain a linear model of our data using the *scipy.stats* library, specifically by using the *linregress* method, which returns a tuple of important data.

```Python
data = stats.linregress(page_speeds,purchase_amounts)
slope,y_intercept,r_value,p_value,err = data
# slope=0.9002615971263267, intercept=27.18052286989289, rvalue=0.7071943433780686, pvalue=1.9259480682867387e-152, stderr=0.02849022562121119)
```

Now, since we have a slope and y-intercept, we will be able to predict new values using the linear model we have constructed. Furthermore, the r-value we have gotten can tell us how accurate our model is, with values of r_value*r_value closer to 0 being inaccurate and those being closer to 1 as very accurate.