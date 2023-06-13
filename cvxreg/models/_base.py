"""
Base convex regression models
"""

# Author: Zhiqiang Liao @ Aalto University <zhiqiang.liao@aalto.fi>
# License: MIT

from abc import ABCMeta, abstractmethod
import numpy as np
from cvxpy import Variable, sum_squares

from ..base import BaseEstimator
from ..opt.cvxpy_opt import solve_model
from ..constant import convex, concave
from ..utils.extmath import yhat
from ..utils._param_check import StrOptions
from ..utils.check import check_ndarray

def _calculate_matrix_A(n):
    res = np.zeros((n*(n-1), n))
    k = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                res[k, i] = -1
                res[k, j] = 1
                k += 1
    return res

def _calculate_matrix_B(x, n, d):
    res = np.zeros((n*(n-1), n*d))
    k = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                res[k, i*d:(i+1)*d] = x[j,:] - x[i,:]
                k += 1
    return res

def _shape_constraint(A, B, Xi, theta, n, d, shape=convex, positive=False):
    check_ndarray(A, n*(n-1), n)
    check_ndarray(B, n*(n-1), n*d)

    if shape == convex:
        cons_shape = A @ theta >= B @ Xi
    elif shape == concave:
        cons_shape = A @ theta <= B @ Xi

    if positive:
        cons_positive = Xi >= 0.0
    else:
        return cons_shape

    return cons_shape, cons_positive

class CRModel(BaseEstimator, metaclass=ABCMeta):
    """
    Base class for CR models
    """
    @abstractmethod
    def fit(self):
        """Fit model."""

    def _decision_function(self, x):
        
        x = self._validate_data(x)
        y_hat = yhat(self.intercept_, self.coef_, x, fun=self.shape)
        return y_hat
    
    def predict(self, x):
        """
        Predict the output variable

        Args:
            x (float): input variables.

        Returns:
            float: predicted output variable
        """
        return self._decision_function(x)
    

class CR(CRModel):
    """
    Convex Regression (CR) model.

    parameters
    ----------
    shape : string, optional (default=Convex)
        The shape of the estimated function. It can be either Convex or Concave.
    positive : boolean, optional (default=False)
        Whether the estimated function is monotonic increasing or not.
    fit_intercept : boolean, optional (default=True)
        Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations.
    email : string, optional (default=None)
        The email address for remote optimization. It will optimize locally if None is given.
    solver : string, optional (default='mosek')
        The solver chosen for optimization. It will optimize with mosek solver if None is given.
    """


    _parameter_constraints: dict = {
        "shape": [StrOptions({convex, concave})],
        'fit_intercept': ['boolean'],
        'positive': ['boolean'],
        'email': [None, str],
        'solver': [str]
    }
    
    def __init__(
        self, 
        shape=convex, 
        positive=False, 
        fit_intercept=True, 
        email=None, 
        solver='mosek'
    ):
        self.shape = shape
        self.fit_intercept = fit_intercept
        self.positive = positive
        self.email = email
        self.solver = solver

    def fit(self, x, y):
        """fit the model with solver

        parameters
        ----------
        x : ndarray of shape (n, d) data.
        y : ndarray of shape (n,) target values.

        Returns
        -------
        self : returns an instance of self
                Fitted model.
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
        objective = 0.5*sum_squares(y - theta)

        # add shape constraint
        constraint = [_shape_constraint(A, B, Xi, theta, n, d, shape=self.shape, positive=self.positive)]

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
