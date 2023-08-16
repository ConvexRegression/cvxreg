===================
Predictive accuracy
===================

We compare the predictive accuracy of convex regression with several different machine learning models for regression. The models are:

- Linear regression
- Ridge regression
- Lasso regression
- Support vector regression

We use the same data as in the previous example.

The following examples are run in python 3.9.15 with Intel(R) Core(TM) i5-1135G7 CPU @ 2.40GHz.

Example: Estimating convex function and calculate mean squared errors
---------------------------------------------------------------------

.. code:: python

    # import packages
    import numpy as np
    from cvxreg.models import CR, PCR
    from sklearn import linear_model, svm

    # generate data
    np.random.seed(0)
    n, d, SNR = 100, 3, 3
    x = np.random.uniform(low=-1, high=1, size=(n*2, d))
    y_true = np.linalg.norm(x, axis=1)**2 + 3
    sigma = np.sqrt(np.var(y_true, ddof=1, axis=0)/SNR)
    nse = np.random.normal(0, sigma, n*2)
    y = y_true + nse

    # split data into training and testing
    x_tr, y_tr = x[:n,:], y[:n]
    x_te, y_te = x[-n:,:], y_true[-n:]

    # Fit the penalized convex regression model
    pcr = PCR(c=0.01)
    pcr.fit(x_tr, y_tr)
    # calculate mean squared errors
    y_hat = pcr.predict(x_te)
    mse = np.mean((y_te - y_hat)**2)
    print('Mean squared error of PCR: %.3f' % mse) # MSE = 0.024

    # Fit the linear regression model
    lr = linear_model.LinearRegression()
    lr.fit(x_tr, y_tr)
    # calculate mean squared errors
    y_hat = lr.predict(x_te)
    mse = np.mean((y_te - y_hat)**2)
    print('Mean squared error of linear regression: %.3f' % mse) # MSE = 0.310

    # Fit the ridge regression model
    rr = linear_model.Ridge(alpha=0.01)
    rr.fit(x_tr, y_tr)
    # calculate mean squared errors
    y_hat = rr.predict(x_te)
    mse = np.mean((y_te - y_hat)**2)
    print('Mean squared error of ridge regression: %.3f' % mse) # MSE = 0.310

    # Fit the lasso regression model
    lasso = linear_model.Lasso(alpha=0.01)
    lasso.fit(x_tr, y_tr)
    # calculate mean squared errors
    y_hat = lasso.predict(x_te)
    mse = np.mean((y_te - y_hat)**2)
    print('Mean squared error of lasso regression: %.3f' % mse) # MSE = 0.308

    # Fit decision tree regression model
    regtr = DecisionTreeRegressor(max_depth=5)
    regtr.fit(x_tr, y_tr)
    # calculate mean squared errors
    y_hat = regtr.predict(x_te)
    mse = np.mean((y_te - y_hat)**2)
    print('Mean squared error of decision trees regression: %.3f' % mse) # MSE = 0.222

    # Fit the support vector regression model
    svr = svm.SVR(kernel='rbf')
    svr.fit(x_tr, y_tr)
    # calculate mean squared errors
    y_hat = svr.predict(x_te)
    mse = np.mean((y_te - y_hat)**2)
    print('Mean squared error of support vector regression: %.3f' % mse) # MSE = 0.042

We concude that the penalized convex regression model has the best predictive accuracy.
