====================
cvxreg.models.CR
====================

.. code:: python

    cvxreg.models.CR(*, shape='convex', monotonic=None, fit_intercept=True, solver='ecos')

Convex regression (CR) model.
-----------------------------

CR fit a convex function with coefficients :math:`\boldsymbol{\xi}_1,\ldots,\boldsymbol{\xi}_n` from the data. :math:`\boldsymbol{\xi}_i` is d-dimensional vector and n is the number of samples. 
The optimization problem is:

.. math::

    \min_{\boldsymbol{\xi}_1,\ldots,\boldsymbol{\xi}_n; \boldsymbol{\theta}} & \frac{1}{2}\sum_{i=1}^n (y_i-\theta_i)^2 \\\\
    s.t. & \theta_i + \boldsymbol{\xi}_i^T (\boldsymbol{x}_j-\boldsymbol{x}_i) \geq \theta_j,  i,j=1,\ldots,n

where :math:`\boldsymbol{x}_i` is the i-th observation, :math:`y_i` is the i-th target value, :math:`\theta_i` is the value of :math:`f(\boldsymbol{x}_i)`, 
:math:`\boldsymbol{\xi}_i` is the coefficient at the i-th observation.

Parameters
----------

======================  =======
Parameters              Options
======================  =======
:code:`shape`           Selection: {:code:`convex`, :code:`concave`}, default: :code:`convex`

                        The shape of the function to be fitted.
:code:`monotonic`       Selection: {:code:`increasing`, :code:`decreasing`}, default: None

                        Whether to constrain the function to be monotonic.
:code:`fit_intercept`   Boolean, default: True

                        Whether to fit the intercept.
:code:`solver`          Selection: {:code:`ecos`, :code:`osqp`, :code:`scs`, :code:`cvxopt`, :code:`mosek`, :code:`gurobi`, :code:`cplex`, :code:`copt`}, default: :code:`ecos`

                        The solver to use. There three open-source solvers: :code:`ecos`, :code:`osqp`, :code:`scs`, and five commercial solvers: :code:`cvxopt`, :code:`mosek`, :code:`gurobi`, :code:`cplex`, :code:`copt`.

                        To use commercial solvers, you need to install them first, see :ref:`install`.
======================  =======

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

See examples: :ref:`examples`.