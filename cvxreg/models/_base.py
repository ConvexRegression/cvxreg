# import dependencies
from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
from pystoned import CNLS

from ..base import BaseEstimator
from ..constant import convex, concave
from ..utils.extmath import yhat
from ..utils._pyomo_opt import optimize_model, check_optimization_status
from ..utils._param_check import StrOptions

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
    

class CR(CRModel, CNLS.CNLS):
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
        x : ndarray of shape (n_samples, n_features) data.
        y : ndarray of shape (n_samples,) target values.

        Returns
        -------
        self : returns an instance of self
                Fitted model.
        """
        self._validate_params()
        x, y = self._validate_data(x, y)

        # interface with CNLS
        if self.shape == convex:
            fun_var = CNLS.FUN_COST
        elif self.shape == concave:
            fun_var = CNLS.FUN_PROD
        if self.fit_intercept:
            intercept = CNLS.RTS_VRS
        else:
            intercept = CNLS.RTS_CRS
        CNLS.CNLS.__init__(self, y, x, z=None, cet=CNLS.CET_ADDI, fun=fun_var, rts=intercept)
        if self.positive:
            self.__model__.beta.setlb(0.0)
        else:
            self.__model__.beta.setlb(None)

        # optimize the model with solver
        self.problem_status, self.optimization_status = optimize_model(
            self.__model__, self.email, self.solver)
        check_optimization_status(self.optimization_status)
        
        alpha = list(self.__model__.alpha[:].value)

        beta = np.asarray([i + tuple([j]) for i, j in zip(list(self.__model__.beta),
                                                          list(self.__model__.beta[:, :].value))])
        beta = pd.DataFrame(beta, columns=['Name', 'Key', 'Value'])
        beta = beta.pivot(index='Name', columns='Key', values='Value')
    
        if self.fit_intercept:
            self.intercept_ = np.array(alpha)
            self.coef_ = beta.to_numpy()
        else:
            self.intercept_ = 0.0
            self.coef_ = beta.to_numpy()

        return self
