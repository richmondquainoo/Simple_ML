# Simple Linear Regression In Machine Learning.

## Introduction
This is a simple machine learning algorithm performed on a dataset. It is used to understand the relationship between input and output variables.
It is a linear model and it assumes a linear relationship between input variables (X) and the output variable (y).
Hence, we try to find out a linear function that predicts the response value (y) as accurately as possible as a function of the feature (or independent variable) (x).

### How to find the best fit line?
In this regression model, we are trying to find the "line of best fit" - the regression line which would lead to minimal errors. 
We are actually trying to minimize the distance between the actual value (y_actual) and the predicted value from our model (y_predicted).


## Dependencies
* numpy
* scikit-learn
* matplotlib


### Importing the required libraries
```python
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import rcParams 
rcParams['figure.figsize'] = (14, 7) 
rcParams['axes.spines.top'] = False 
rcParams['axes.spines.right'] = False
```
### Create the python class

```python
class SimpleLinearRegression:
    def __init__(self):
        self.b0 = None 
        self.b1 = None
    
    def fit(self, X, y):
        """
        Used to calculate slope and intercept coefficients.
        :param X: array, single feature 
        :param y: array, true values 
        :return: None
        """
        numerator = np.sum((X - np.mean(X)) * (y - np.mean(y))) 
        denominator = np.sum((X - np.mean(X)) ** 2)
        self.b1 = numerator / denominator
        self.b0 = np.mean(y) - self.b1 * np.mean(X)
        
    def predict(self, X):
        """
        Makes predictions using the simple line equation.
        :param X: array, single feature 
        :return: None  
        """   
        if not self.b0 or not self.b1:
            raise Exception('Please call `SimpleLinearRegression.fit(X, y)` before making predictions.')
        return self.b0 + self.b1 * X
    
```

### Testing
```python
X = np.arange(start=1, stop=301)
y = np.random.normal(loc=X, scale=20)

plt.scatter(X, y, s=200, c='#087E8B', alpha=0.65) 
plt.title('Source dataset', size=20)
plt.xlabel('X', size=14)
plt.ylabel('Y', size=14)
plt.show()
```

