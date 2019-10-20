# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 14:35:02 2019

@author: Brandon
"""

import pandas_datareader
print(pandas_datareader.__version__)
import pandas as pd
import numpy as np
from pandas_datareader import data as pdata
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from yahoo_fin import stock_info as si

# Set the start and end date
start_date = '1990-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')
# Set the ticker
ticker = ['AMZN', 'AAPL', 'IBM', 'MSFT']
ticker = ['AAPL', 'IBM', 'MSFT']
# Get the data
data = pdata.get_data_yahoo(ticker, start_date, end_date)
head = data.head()
print(head)

#data['Adj Close'].plot()
# Define the label for the title of the figure
#plt.title("Adjusted Close Price of %s" % ticker, fontsize=16)
# Define the labels for x-axis and y-axis
#plt.ylabel("Price", fontsize=14)
#plt.xlabel("Year", fontsize=14)
# Plot the grid lines
#plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
#plt.show()


def runRegression(tick, model):
    # Run regression for just Apple
    ticker = [tick]
    data = pdata.get_data_yahoo(ticker, start_date, end_date)
    head = data.head()
    print(head)
    
    data['Adj Close'].plot()
    # Define the label for the title of the figure
    plt.title("Adjusted Close Price of %s" % ticker, fontsize=16)
    # Define the labels for x-axis and y-axis
    plt.ylabel("Price", fontsize=14)
    plt.xlabel("Year", fontsize=14)
    # Plot the grid lines
    plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
    plt.show()
    
    '''
    Lets try to predict closing price based on opening price
    '''
    # Plot the open and close values
    data = data[['Open', 'Adj Close']]
    data.plot(x=0, y=1, style='o')
    plt.title('Open vs Adj Close')
    plt.xlabel('Open')
    plt.ylabel('Adj Close')
    plt.show()
    
    # Plot the average close value
    plt.figure(figsize=(10,5))
    #plt.tight_layout()
    seabornInstance.distplot(data['Adj Close'])
    plt.show()
    
    # reshape adds another dimension to our array we need this to run LinearRegression
    X = data['Open'].values.reshape(-1,1)
    Y = data['Adj Close'].values.reshape(-1,1)
    # Create a 80% 20% split of hte data to be sued for training and testing
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 0)
    
    # Create a regression model

    # Train the model
    model.fit(X_train, Y_train)
    
    # retrieve the intercept
    print("Intercept: ", model.intercept_)
    # retrieve the slope
    print("Slope: ", model.coef_)
    
    # Test the trained model
    predicted = model.predict(X_test)
    
    # Compare the actual oputput values with the predicted values
    df = pd.DataFrame({'Actual': Y_test.flatten(), 'Predicted': predicted.flatten()})
    print(df)
    
    # Visualize the comparison as a bar graph
    df1 = df.head(25)
    df1.plot(kind='bar', figsize=(10,5))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()
    
    # Plot the straight line with the test data
    plt.scatter(X_test, Y_test, color='gray')
    plt.plot(X_test, predicted, color='red', linewidth=2)
    plt.show()
    
    # The mean absolute error is the difference between the actual value and the predicted value
    print('Mean Absolute Error: ', metrics.mean_absolute_error(Y_test, predicted))
    # The mean squared error tells you how close the regression line is to a set of points.
    # the distance of the points are squared to remove negative values and add more weight to larger differences. 
    print('Mean Squared Error: ', metrics.mean_squared_error(Y_test, predicted))
    # Root Mean Square Error (RMSE) is the standard deviation of the residuals (prediction errors). 
    # Residuals are a measure of how far from the regression line data points are; 
    # RMSE is a measure of how spread out these residuals are. In other words, 
    # it tells you how concentrated the data is around the line of best fit.
    print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(Y_test, predicted)))
    return X_test, Y_test


def predictDay(tick):
    # Get today's opening price
    aaplQuoteTable = si.get_quote_table(tick, dict_result = False)
    today_price_orig = aaplQuoteTable.iloc[12]['value']
    today_price = pd.Series([today_price_orig], index=[0])
    today_price = today_price.values.reshape(-1,1)
    # Predict the closing price based on the opening price
    predicted = aaplRegressor.predict(today_price)
    print("Today's opening price ", tick, ": ", today_price_orig)
    print("Today's predicted adj closing price: ", round(predicted.flatten()[0], 2))
    # find the difference in the prediction
    difference = round(predicted.flatten()[0],2) - today_price[0][0]
    print("Difference: ", difference)
def predictDayVal(tick, val):
    # predict today's opening price
    today_price_orig = val
    today_price = val.reshape(1,-1)
    predicted = aaplRegressor.predict(today_price)
    print("Today's opening price ", tick, ": ", today_price_orig)
    print("Today's predicted adj closing price: ", round(predicted.flatten()[0], 2))
    difference = round(predicted.flatten()[0],2) - today_price
    print("Predicted difference: ", difference)
    if (difference > 0):
        print("Buy stock")
        return True
    else:
        print("Don't buy stock")
        return False


aaplRegressor = LinearRegression()
runRegression("AAPL", aaplRegressor)
predictDay('AAPL')

ibmRegressor = LinearRegression()
runRegression("IBM", ibmRegressor)
predictDay('IBM')

msftRegressor = LinearRegression()
X_test, Y_test = runRegression("MSFT", msftRegressor)
predictDay("MSFT")

input("Press enter to continue")

# Treat each entry in our test set as a new day
# For every day predict the adj closing based on the opening
i = 0
money = 100
for x in X_test:
    print(i)
    buyStock = predictDayVal('MSFT', x)
    if (buyStock == True):
        if (money > x):
            # If our predicted difference is positive spend money to buy the stock
            money = money - x
            print("Money after buying opening price: ", money)
            # Sell the stock back at the end of the day
            money = money + Y_test[i]
            print("Money after selling closing price: ", money)
            input("Press enter to continue")
    print("Today's actual adj closing price: ", Y_test[i])
    actualDifference = Y_test[i] - x
    print("Actual difference: ", actualDifference)
    print("\n")
    i = i+1
    
