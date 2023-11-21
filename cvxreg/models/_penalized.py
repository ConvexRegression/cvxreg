"""
Penalized Convex Regression (PCR) model.
"""

# Author: Zhiqiang Liao @ Aalto University <zhiqiang.liao@aalto.fi>
# License: MIT

from numbers import Real
import numpy as np
from cvxpy import Variable, sum_squares

from ._base import CRModel, _calculate_matrix_A, _calculate_matrix_B, _shape_constraint
from ._cvxpy_opt import solve_model
from ..constant import convex, concave
from ..utils._param_check import Interval, StrOptions


class PCR(CRModel):
    """
    Penalized Convex Regression (PCR) model.

    parameters
    ----------
    c : float, optional (default=1.0)
        The penalty parameter.
    shape : string, optional (default=Convex)
        The shape of the estimated function. It can be either Convex or Concave.
    positive : boolean, optional (default=False)
        Whether the estimated function is monotonic increasing or not.
    fit_intercept : boolean, optional (default=True)
        Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations.
    solver : string, optional (default='mosek')
        The solver chosen for optimization. It will optimize with mosek solver if None is given.
    """

    _parameter_constraints: dict = {
        "c": [Interval(Real, 0, None)],
        "shape": [StrOptions({convex, concave})],
        'fit_intercept': ['boolean'],
        'positive': ['boolean'],
        'solver': [str]
    }

    def __init__(
        self, 
        c=1.0, 
        shape=convex, 
        positive=False, 
        fit_intercept=True,
        solver='ecos'
    ):
        self.c = c
        self.shape = shape
        self.fit_intercept = fit_intercept
        self.positive = positive
        self.solver = solver

    def fit(self, x, y):
        """Optimize the function by requested method

        parameters
        ----------
        x : ndarray of shape (n, d) data.
        y : ndarray of shape (n,) target values.

        Returns
        -------
        self : returns an instance of self
        """
        self._validate_params()
        x, y = self._validate_data(x, y)

        # calculate the matrix A and B
        n, d = x.shape
        A = _calculate_matrix_A(n)
        B = _calculate_matrix_B(x, n, d)

        # interface with cvxpy
        Xi = Variable(n*d)
        theta = Variable(n)
        objective = 0.5*sum_squares(y - theta) + self.c*sum_squares(Xi)

        # add shape constraint
        constraint = _shape_constraint(A, B, Xi, theta, shape=self.shape, positive=self.positive)

        # optimize the model with solver
        self.solution = solve_model(objective, constraint, self.solver)
        
        Xi_val = Xi.value.reshape(n,d)
        theta_val = theta.value

        alpha = list([theta_val[i] - Xi_val[i,:]@x[i,:] for i in range(n)])
        beta = Xi_val
    
        if self.fit_intercept:
            self.intercept_ = np.array(alpha)
            self.coef_ = beta
        else:
            self.intercept_ = 0.0
            self.coef_ = beta

        return self