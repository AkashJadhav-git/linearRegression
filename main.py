import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def main():
    # Read the csv file.
    customers = pd.read_csv("Ecommerce Customers")

    # Print the range of data, total columns, types of data.
    print(customers.info())
    print("___________________________________________________________________________________________")

    # split the data into training and testing sets
    # A variable y equal to the "Yearly Amount Spent" column.
    y = customers['Yearly Amount Spent']

    # a variable X equal to the numerical features of the customers.
    X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]

    # By using model_selection.train_test_split from sklearn to split the data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    # Create instance of a LinearRegression() model.
    lm = LinearRegression()

    # Fit lm on the training data.
    lm.fit(X_train, y_train)

    # Create coefficient of the lm data.
    'Coefficients: \n', lm.coef_

    # Now use lm.predict() to predict off the X_test set of the data.
    predictions = lm.predict(X_test)

    # Create a scatterplot of the real test values versus the predicted values. Which is save in figure_1.png.
    plt.scatter(y_test, predictions)
    plt.xlabel('Y Test')
    plt.ylabel('Predicted Y')
    plt.show()

    # Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error.
    print('MAE:', metrics.mean_absolute_error(y_test, predictions))
    print('MSE:', metrics.mean_squared_error(y_test, predictions))
    print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, predictions)))
    print("___________________________________________________________________________________________")

    coeffecients = pd.DataFrame(lm.coef_, X.columns)
    coeffecients.columns = ['Coeffecient']
    print(coeffecients)


if __name__ == '__main__':
    main()


