import numpy as np
import quandl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn import datasets
from linear_regression import LinearRegression


x, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1234)

# fig = plt.figure(figsize=(8,6))
# plt.scatter(x[:, 0], y, color="b", marker="o", s=30)
# plt.show()

regressor = LinearRegression(lr=1)

regressor.fit(x_train, y_train)

predicted = regressor.predict(x_test)

def mse(y_true, y_predicted):
    return np.mean((y_true -  y_predicted)**2)


mse_value = mse(y_test, predicted)

print(mse_value)


y_predline = regressor.predict(x)

map = plt.get_cmap('viridis')
fig = plt.figure(figsize=[8,6])

m1 = plt.scatter(x_train, y_train, color=map(0,9), s=10)
m2 = plt.scatter(x_test, y_test, color='red', s=10)

plt.plot(x, y_predline, color='black', linewidth=2, label='Prediction')
plt.show()