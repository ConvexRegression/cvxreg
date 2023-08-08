====================
cvxreg.models.PCR
====================

.. code:: python

    cvxreg.models.PCR(*, c=1.0, shape='convex', positive=False, fit_intercept=True, solver='ecos')

Penalized Convex Regression (PCR) model.
----------------------------------------

PCR fit a convex function with coefficients beta = (beta_1, ..., beta_n, where beta_i is d-dimensional vector and n is the number of samples) to the data. 
The optimization problem is:

.. math::

    \minimize \sum_{i=1}^n \epsilon_i^2 + c * \sum_{i=1}^n \|\beta_i\|_2^2

    s.t. y_i = \alpha_i + \beta_i * x_i + \epsilon_i
               \alpha_i + \beta_i * x_i \geq \alpha_j + \beta_j * x_i \forall j != i

where :math:`x_i` is the i-th sample, :math:`y_i` is the i-th target value, :math:`\alpha_i` is the intercept of the i-th sample, 
:math:`\beta_i` is the coefficient of the i-th sample, :math:`\epsilon_i` is the error of the i-th sample, and :math:`c` is the regularization parameter.

Parameters
----------

====================              =======
Parameters                        Options
====================              =======
:code:`c`                         Float, default: 1.0. c must be non-negative Float, i.e. in :math:`[0, inf)`.
                                  The regularization parameter.
:code:`shape`                     Selection: {'convex', 'concave'}, default: 'convex'
                                  The shape of the function to be fitted.
:code:`positive`                  Boolean, default: False
                                  Whether to constrain the coefficients to be positive.
:code:`fit_intercept`             Boolean, default: True
                                  Whether to fit the intercept.
:code:`solver`                    Selection: {'ecos', 'osqp', 'scs', 'cvxopt','mosek', 'gurobi', 'cplex', 'copt'}, default: 'ecos'
                                  The solver to use. There three open-source solvers: 'ecos', 'osqp', 'scs', and five commercial solvers: 'cvxopt', 'mosek', 'gurobi', 'cplex', 'copt'.
                                  To use commercial solvers, you need to install them first, see :ref:`install`.
====================              =======

Attributes
----------

====================  =======
Attributes            Type
====================  =======
:code:`coef_`         numpy.ndarray <br />
                      The coefficients of the fitted function.
:code:`intercept_`    numpy.ndarray <br />
                      The intercept of the fitted function.
====================  =======

Examples
--------
.. code:: python

    import numpy as np
    from cvxreg.models import PCR
    X = np.array([[1, 1], [1, 2], [2, 2]])
    y = [1, 2, 3]
    pcr = PCR()
    pcr.fit(X, y)
    print(pcr.coef_)
    # [[-2.80851372e-06 -4.48428251e-06]
    #   [ 2.02631940e-07  1.89185893e-01]
    #   [ 3.51352926e-01  1.89185893e-01]]
    print(pcr.intercept_)
    # [1.75676572 1.56757233 1.21621961]

Methods
-------

====================  =======
Methods               Type
====================  =======
:code:`fit(X, y)`     Fit model with solver. <br />
                      X of shape (n_samples, n_features) <br />
                      y of shape (n_samples,)

:code:`predict(X)`    Predict using the convex regression model. <br />
                      X of shape (n_samples, n_features)
====================  =======

See examples: :ref:`Examples <examples>`.