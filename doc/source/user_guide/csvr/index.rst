====================
cvxreg.models.CSVR
====================

.. code:: python

    cvxreg.models.CSVR(*, c=1.0, epsilon=0.1, shape='convex', monotonic=None, fit_intercept=True, solver='ecos')

Convex Support Vector Regression (CSVR) model.
------------------------------------------------

CSVR fit a convex function with coefficients :math:`\boldsymbol{\xi}_1,\ldots,\boldsymbol{\xi}_n` from the data. :math:`\boldsymbol{\xi}_i` is d-dimensional vector and n is the number of observations.
The optimization problem is:

.. math::

    \min_{\boldsymbol{\xi}_1,\ldots,\boldsymbol{\xi}_n; \boldsymbol{\theta}, \pi, \pi^*} & \frac{1}{2}\sum_{i=1}^n \|\boldsymbol{\xi}_i\|_2^2 + c * \sum_{i=1}^n (\pi_i + \pi_i^*) \\\\
    s.t. & y_i - \theta_i \leq \epsilon + \pi_i,  i=1,\ldots,n \\\\
         & \theta_i - y_i \leq \epsilon + \pi_i^*,  i=1,\ldots,n \\\\
         & \theta_i + \boldsymbol{\xi}_i^T (\boldsymbol{x}_j - \boldsymbol{x}_i) \geq \theta_j,  i,j=1,\ldots,n

where :math:`\boldsymbol{x}_i` is the i-th observation, :math:`y_i` is the i-th target value, :math:`\theta_i` is the value of :math:`f(\boldsymbol{x}_i)`, 
:math:`\boldsymbol{\xi}_i` is the coefficient at the i-th observation, and :math:`c` is the regularization parameter.

Parameters
----------

======================    =======
Parameters                Options
======================    =======
:code:`c`                 Float, default: 1.0. c must be non-negative Float, i.e. in :math:`[0, inf)`.

                          The regularization parameter.
:code:`epsilon`           Float, default: 0.1. epsilon must be non-negative Float, i.e. in :math:`[0, inf)`.

                            The epsilon parameter in the epsilon-insensitive loss function.
:code:`shape`             Selection: {:code:`convex`, :code:`concave`}, default: :code:`convex`

                          The shape of the function to be fitted.
:code:`monotonic`         Selection: {:code:`increasing`, :code:`decreasing`}, default: None

                          Whether to constrain the function monotonic.
:code:`fit_intercept`     Boolean, default: True

                          Whether to fit the intercept.
:code:`solver`            Selection: {:code:`ecos`, :code:`osqp`, :code:`scs`, :code:`cvxopt`, :code:`mosek`, :code:`gurobi`, :code:`cplex`, :code:`copt`}, default: :code:`ecos`

                          The solver to use. There three open-source solvers: :code:`ecos`, :code:`osqp`, :code:`scs`, and five commercial solvers: :code:`cvxopt`, :code:`mosek`, :code:`gurobi`, :code:`cplex`, :code:`copt`.

                          To use commercial solvers, you need to install them first, see :ref:`install`.
======================    =======

Attributes
----------

====================  =======
Attributes            Type
====================  =======
:code:`coef_`         numpy.ndarray 

                      The coefficients of the fitted function.
:code:`intercept_`    numpy.ndarray 

                      The intercept of the fitted function.
====================  =======

Examples
--------
.. code:: python

    import numpy as np
    from cvxreg.models import CSVR
    X = np.array([[1, 1], [1, 2], [2, 2]])
    y = [1, 2, 3]
    csvr = CSVR()
    csvr.fit(X, y)
    print(csvr.coef_)
    # [[-2.72282908e-09 -9.16695789e-09]
    #  [-4.73065053e-09  4.99999982e-01]
    #  [ 9.99992687e-01  4.99999982e-01]]
    print(csvr.intercept_)
    # [ 1.40000369  0.9000037  -0.09998899]

Methods
-------

====================  =======
Methods               Type
====================  =======
:code:`fit(X, y)`     Fit model with solver. 

                      X of shape (n_samples, n_features) 
                      
                      y of shape (n_samples,)

:code:`predict(X)`    Predict using the convex regression model. 

                      X of shape (n_samples, n_features)
====================  =======

See examples: :ref:`examples`.