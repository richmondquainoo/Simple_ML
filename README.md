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
Click on the image to have a better view
![bestfit](https://user-images.githubusercontent.com/84714883/122829187-36411000-d2d6-11eb-85f4-375aa096f62b.png)

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = SimpleLinearRegression() 
model.fit(X_train, y_train)
preds = model.predict(X_test)
model.b0, model.b1
```
(-3.5170837783608135, 1.0219248804797427)



```python
preds
y_test
from sklearn.metrics import mean_squared_error
rmse = lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred))
rmse(y_test, preds)
```
14.941281910012036

```python
model_all = SimpleLinearRegression() 
model_all.fit(X, y)
preds_all = model_all.predict(X)

plt.scatter(X, y, s=200, c='#087E8B', alpha=0.65, label='Source data')
plt.plot(X, preds_all, color='#000000', lw=3, label=f'Best fit line > B0 = {model_all.b0:.2f}, B1 = {model_all.b1:.2f}')
plt.title('Best fit line', size=20)
plt.xlabel('X', size=14)
plt.ylabel('Y', size=14)
plt.legend()
plt.show()
```
Click the image to have a better view of it
![fitline](https://user-images.githubusercontent.com/84714883/122830252-9c7a6280-d2d7-11eb-96f0-1984d4c294f2.png)

#### Comparison with Scikit-Learn
``` python
from sklearn.linear_model import LinearRegression
sk_model = LinearRegression() 
sk_model.fit(np.array(X_train).reshape(-1, 1), y_train) 
sk_preds = sk_model.predict(np.array(X_test).reshape(-1, 1))
sk_model.intercept_, sk_model.coef_
```
(-3.517083778360842, array([1.02192488]))
```python
rmse(y_test, sk_preds)
```
14.941281910012039
