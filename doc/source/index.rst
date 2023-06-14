.. cvxreg documentation master file, created by
   sphinx-quickstart on Tue Jun  6 14:06:22 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to cvxreg's documentation!
============================================

**Convex Regression for machine learning.**

cvxreg is an open source Python package implementing convex regression models for machine learning. It is built on top of the `pyStoNED <https://pypi.org/project/pystoned/>`_ package for Convex Nonparametric Least Square estimator. 
It lets you implement the convex regression models in a few lines of code. It is well documented and tested. It is compatible with Python 3.6+ and runs on Linux, MacOS X and Windows.

For example, the following code estimates a convex function with CR model:

.. code:: python

   import numpy as np
   from cvxreg.models import CR

   # Generate data
   np.random.seed(0)
   n, d, SNR = 50, 3, 3
   x = np.random.uniform(low=-1, high=1, size=(n, d))
   y_true = np.linalg.norm(x, axis=1)**2 + 3

   sigma = np.sqrt(np.var(y_true, ddof=1, axis=0)/SNR)
   nse = np.random.normal(0, sigma, n)
   y = y_true + nse

   # Fit CR model
   cr = CR()
   cr.fit(x, y)

   # print the coefficients
   print(cr.coef_)
   # print the intercept
   print(cr.intercept_)
   # predict the response
   y_pred = cr.predict([[0.1, 0.2, 0.3]])

.. toctree::
   :hidden:

   install/index

.. toctree::
   :maxdepth: 3
   :hidden:

   examples/index
