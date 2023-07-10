# Linear Regression for Sales Prediction
This project aims to build a simple linear regression model to predict sales based on TV marketing expenses. Three different approaches are investigated: using NumPy and Scikit-Learn linear regression models, as well as constructing and optimizing the sum of squares cost function with gradient descent from scratch.

# 1. Open the Dataset and State the Problem
The dataset used in this project is stored in the file data/tvmarketing.csv. It contains two fields: TV marketing expenses (TV) and sales amount (Sales). The goal is to build a linear regression model that can predict sales based on TV marketing expenses.

# 2. Linear Regression in Python with NumPy and Scikit-Learn
This section explores two different implementations of linear regression using popular Python libraries.
The NumPy library is used to perform linear regression. The dataset is loaded, and the linear regression model is fitted to the data. The model's coefficients and performance metrics are evaluated and displayed.
Scikit-Learn is a powerful machine learning library that provides various regression models. Here, the Scikit-Learn linear regression model is used to fit the data, and the model's coefficients and performance metrics are computed.

# 3. Linear Regression using Gradient Descent
This section introduces a manual implementation of linear regression using gradient descent. The sum of squares cost function is constructed, and gradient descent is used to optimize the model's coefficients. The process of finding the optimal coefficients and the cost function's convergence are explained.

# 4. Results
The results obtained from the different approaches are as follows:

Linear Regression with NumPy:

Predictions of sales: [ 9.40942557 12.7369904 20.34285287]
Linear Regression with Scikit-Learn:

Predictions of sales: [ 9.40942557 12.7369904 20.34285287]

Linear Regression using Gradient Descent:

Predictions of sales (Scikit-Learn): [ 9.40942557 12.7369904 20.34285287]
Predictions of sales (Gradient Descent): [ 9.40942557 12.7369904 20.34285287]
5. Analysis
The linear regression models built using NumPy, Scikit-Learn, and gradient descent all provide similar predictions for sales based on TV marketing expenses. The predicted sales amounts increase as the TV marketing expenses increase, indicating a positive linear relationship between these variables.
The models' coefficients, performance metrics, and convergence can be further analyzed to assess the accuracy and reliability of the predictions. It is recommended to evaluate additional metrics such as mean squared error or R-squared to quantify the models' performance.
Furthermore, visualizing the regression line and the predicted points can provide insights into the quality of the model's fit to the data.
