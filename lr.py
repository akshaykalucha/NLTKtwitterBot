# # import the pandas library and aliasing as pd
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

dataframe = pd.read_csv('50_Startups.csv')

# y = dataframe.head()

x = dataframe.iloc[:, :-1].values #Selects all columns except last one and binds their values to an array
y = dataframe.iloc[:, -1].values #selects only last column which is dependednt variable and selects all row for that

#Categorizing String values into dummy variables
# we want to encode the 4th column which is state and is on 3rd index in the array x
ct = ColumnTransformer(transformers=[[ 'encoder', OneHotEncoder(), [3] ]], remainder='passthrough')


#fitting encoded value of state in x
x = np.array(ct.fit_transform(x))

# print(x)


# Splitting the x and y values in array into 80% trainign daa nd 20% testing data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=0)



# Training our model of x_train and y_train in LinearRegression(multiple)

regressor = LinearRegression()

# fitting and training our data in LR
regressor.fit(x_train, y_train)

# Predicting our values of x_test on this trained model
y_predicted = regressor.predict(x_test)
np.set_printoptions(precision=2)

#predicted vs actual value
print(np.concatenate((y_predicted.reshape(len(y_predicted), 1), y_test.reshape(len(y_test),1)), 1))



# df = pd.DataFrame(np.random.randn(8, 4),
# index = ['a','b','c','d','e','f','g','h'], columns = ['A', 'B', 'C', 'D'])

# # Select range of rows for all columns
# print(df.loc[:, :-1].values)