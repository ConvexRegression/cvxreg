# import dependencies
from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
from pystoned import CNLS

from ..base import BaseEstimator
from ..constant import Convex, Concave, OPT_DEFAULT, OPT_LOCAL
from ..utils import tools
from ..utils.extmath import yhat
from ..utils._param_check import StrOptions

class CRModel(BaseEstimator, metaclass=ABCMeta):
    """
    Base class for CR models
    """
    @abstractmethod
    def fit(self):
        """Fit model."""

    def _decision_function(self, x):
        tools.assert_optimized(self.optimization_status)
        
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
    Convex Regression (CR) model
    """
    _parameter_constraints: dict = {
        "shape": [StrOptions({Convex, Concave})],
        'fit_intercept': ['boolean'],
        'positive': ['boolean']
    }
    
    def __init__(self, shape=Convex, positive=False, fit_intercept=True):
        """CNLS model

        Args:
            y (float): output variable. 
            x (float): input variables.
            fun (String, optional): FUN_CVX (convex funtion) or FUN_CCV (concave funtion). Defaults to FUN_CVX.
            positive (bool, optional): True if the coefficients are positive. Defaults to False.
            fit_intercept (bool, optional): True if the model should include an intercept. Defaults to True.
        """
        
        self.shape = shape
        self.fit_intercept = fit_intercept
        self.positive = positive


    def fit(self, x, y, email=OPT_LOCAL, solver=OPT_DEFAULT):
        """Optimize the function by requested method

        Args:
            email (string): The email address for remote optimization. It will optimize locally if OPT_LOCAL is given.
            solver (string): The solver chosen for optimization. It will optimize with default solver if OPT_DEFAULT is given.
        """
        self._validate_params()
        x, y = self._validate_data(x, y)

        # interface with CNLS
        if self.shape == Convex:
            self.fun_var = CNLS.FUN_COST
        elif self.shape == Concave:
            self.fun_var = CNLS.FUN_PROD
        if self.fit_intercept:
            self.intercept = CNLS.RTS_VRS
        else:
            self.intercept = CNLS.RTS_CRS
        CNLS.CNLS.__init__(self, y, x, z=None, cet=CNLS.CET_ADDI, fun=self.fun_var, rts=self.intercept)
        if self.positive:
            self.__model__.beta.setlb(0.0)
        else:
            self.__model__.beta.setlb(None)

        # optimize the model with solver
        self.problem_status, self.optimization_status = tools.optimize_model(
            self.__model__, email, solver)
        
        tools.assert_optimized(self.optimization_status)

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
