====================
cvxreg.models.CR
====================

.. code:: python

    cvxreg.models.CR(*, shape='convex', positive=False, fit_intercept=True, solver='ecos')

Convex regression (CR) model.
-----------------------------

CR fit a convex function with coefficients beta = (beta_1, ..., beta_n, where beta_i is d-dimensional vector and n is the number of samples) to the data. 
The optimization problem is:

.. math::

    \min_{\beta, \alpha, \epsilon} \sum_{i=1}^n \epsilon_i^2

    s.t. y_i = \alpha_i + \beta_i * x_i + \epsilon_i

               \alpha_i + \beta_i * x_i \geq \alpha_j + \beta_j * x_i, \forall j != i

where :math:`x_i` is the i-th sample, :math:`y_i` is the i-th target value, :math:`\alpha_i` is the intercept of the i-th sample, 
:math:`\beta_i` is the coefficient of the i-th sample, :math:`\epsilon_i` is the error of the i-th sample.

Parameters
----------

====================  =======
Parameters            Options
====================  =======
:code:`shape`         Selection: {'convex', 'concave'}, default: 'convex'

                      The shape of the function to be fitted.
:code:`positive`      Boolean, default: False

                      Whether to constrain the coefficients to be positive.
:code:`fit_intercept` Boolean, default: True

                      Whether to fit the intercept.
:code:`solver`        Selection: {'ecos', 'osqp', 'scs', 'cvxopt','mosek', 'gurobi', 'cplex', 'copt'}, default: 'ecos'

                      The solver to use. There three open-source solvers: 'ecos', 'osqp', 'scs', and five commercial solvers: 'cvxopt', 'mosek', 'gurobi', 'cplex', 'copt'.

                      To use commercial solvers, you need to install them first, see :ref:`install`.
====================  =======

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
    from cvxreg.models import CR
    X = np.array([[1, 1], [1, 2], [2, 2]])
    y = [1, 2, 3]
    cr = CR()
    cr.fit(X, y)
    print(cr.coef_)
    # [[0.99999721 0.27092186]
    #   [0.27092381 1.72907812]
    #   [1.72907904 1.00000423]]
    print(cr.intercept_)
    # [-0.27091906 -1.72908007 -2.45816658]

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

See examples: :ref:`Examples <comparison>`.

