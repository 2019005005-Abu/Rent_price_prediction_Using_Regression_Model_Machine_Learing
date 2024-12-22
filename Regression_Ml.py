##Regression Model
'''
Regression is a  technique used to to predict the numberical value on input features.
It models  the relationship between a dependent variant  and independent variables.

'''
##Classification Model

'''
Classification is a supervised learning task that involves predicting the class of a given input data point.
It models  the relationship between a dependent categorical variable and one or more independent variables.
'''
x=10;
y=(2*x)+3;
print(y);
x=30;
y=(2*x)+3;
print(y);
x=50;
y=(2*x)+3;
print(y);
###
'''
y=mx+c
F(x)=max+c;
y=prediction
##
m=Sumassion(x-x bar)(y-y bar)/Sumassion(x-x)sqrt
##Underfit Model
##Overfit Model
##bestfit model
'''
##Implementation  Model
import pandas as pd;
data = {'X': [10, 20, 30, 40], 'Y': [5, 15, 25, 35]}
df = pd.DataFrame(data)
mean_x=df['X'].mean();
mean_y=df['Y'].mean();
print('x-axis',mean_x);
print('y-axis',mean_y);
'''
Linear Regression 
L1 Loss & L2 Loss
1.Mean Absolute Error(MAE)
MAE=1/nth(|actual_values-predictions|+|actual_values-predictions|)
2.Mean Squared Error(MSE)
MSC=(1/nth(sqrt(actual_values-predictions))
3.Root mean squared error(RMSC)
RMSC=sqrt(MSC)
'''

import numpy as np
import matplotlib.pyplot as plt
import warnings as w
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

# Ignore warnings
w.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('./data__set/Dhaka_Rent.csv', sep=';')
print(df.head(3))
print("DataFrame shape:", df.shape)

# Define variables
x = df['area']
y = df['rent']

# Calculate mean values
x__mean = np.mean(x)
y__mean = np.mean(y)
print("Mean of area (x):", x__mean)
print("Mean of rent (y):", y__mean)

# Calculate slope (m) and intercept (c) for the regression line
dev_x = x - x__mean
dev_y = y - y__mean
m = np.sum(dev_x * dev_y) / np.sum(dev_x ** 2)
c = y__mean - m * x__mean

# Prediction
prediction_input = input('Enter prediction area: ')
prediction_input = int(prediction_input)
prediction_value = m * prediction_input + c
print("Enter value for prediction:", round(prediction_value))

# Add predictions using manual regression to the dataframe
df['manual_predictions'] = m * df['area'] + c

# Implementation of regression model using scikit-learn
regression_model = LinearRegression()
regression_model.fit(df[['area']], df[['rent']])
coefcient_value = regression_model.coef_[0][0]
intercept_value = regression_model.intercept_[0]
print("Coefficient value (from model):", coefcient_value)
print("Intercept value (from model):", intercept_value)

# Predict rent for a specific area using the model
predicted_value_model = regression_model.predict([[2000]])[0][0]
print("Predicted rent value (from model):", predicted_value_model)

# Add predictions using the regression model to the dataframe
df['predictions'] = regression_model.predict(df[['area']])

# Evaluate model performance
MSE = mean_squared_error(df['rent'], df['predictions'])
MAE = mean_absolute_error(df['rent'], df['predictions'])
print('Mean Squared Error (MSE):', MSE)
print('Mean Absolute Error (MAE):', MAE)

# Plot the data and regression line
plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(x, m * x + c, color='red', label='Manual Regression Line')
plt.plot(df['area'], df['predictions'], color='red', linestyle='--', label='Model Predictions')
plt.xlabel('Area')
plt.ylabel('Rent')
plt.title('Area vs Rent')
plt.legend()
plt.show()



