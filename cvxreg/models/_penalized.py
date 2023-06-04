# import dependencies
import numpy as np
import pandas as pd
from pystoned import CNLS
from pyomo.environ import Objective, minimize

from ._base import CRModel
from ..constant import Convex, Concave, OPT_DEFAULT, OPT_LOCAL
from ..utils import tools


class PCR(CRModel, CNLS.CNLS):
    """
    Penalized Convex Regression (PCR) model
    """

    def __init__(self, x, y, c=1.0, shape=Convex, positive=False, fit_intercept=True):
        """PCR model

        Args:
            y (float): output variable. 
            x (float): input variables.
            fun (String, optional): FUN_CVX (convex funtion) or FUN_CCV (concave funtion). Defaults to FUN_CVX.
            positive (bool, optional): True if the coefficients are positive. Defaults to False.
            fit_intercept (bool, optional): True if the model should include an intercept. Defaults to True.
        """
        
        self.c = c
        self.shape = shape
        self.fit_intercept = fit_intercept
        self.positive = positive

        if self.shape == Convex:
            fun_var = CNLS.FUN_COST
        elif self.shape == Concave:
            fun_var = CNLS.FUN_PROD
        if self.fit_intercept:
            intercept = CNLS.RTS_VRS
        else:
            intercept = CNLS.RTS_CRS
        CNLS.CNLS.__init__(self, y, x, z=None, cet=CNLS.CET_ADDI, fun=fun_var, rts=intercept)

        # new objective function
        self.__model__.objective.deactivate()
        self.__model__.new_objective = Objective(rule=self.__new_objective_rule(),
                                                     sense=minimize,
                                                     doc='objective function')

        if self.positive:
            self.__model__.beta.setlb(0.0)
        else:
            self.__model__.beta.setlb(None)

    def __new_objective_rule(self):
        """return new objective function"""
        def objective_rule(model):
            return sum(model.epsilon[i] ** 2 for i in model.I) \
                + self.c * sum(model.beta[i, j] ** 2 for i in model.I for j in model.J)
        return objective_rule

    def fit(self, email=OPT_LOCAL, solver=OPT_DEFAULT):
        """Optimize the function by requested method

        Args:
            email (string): The email address for remote optimization. It will optimize locally if OPT_LOCAL is given.
            solver (string): The solver chosen for optimization. It will optimize with default solver if OPT_DEFAULT is given.
        """
        # TODO(error/warning handling): Check problem status after optimization
        self.problem_status, self.optimization_status = tools.optimize_model(
            self.__model__, email, solver)
        
        tools.assert_optimized(self.optimization_status)

        alpha = list(self.__model__.alpha[:].value)

        beta = np.asarray([i + tuple([j]) for i, j in zip(list(self.__model__.beta),
                                                          list(self.__model__.beta[:, :].value))])
        beta = pd.DataFrame(beta, columns=['Name', 'Key', 'Value'])
        beta = beta.pivot(index='Name', columns='Key', values='Value')
    
        if self.fit_intercept:
            self.intercept_ = alpha
            self.coef_ = beta.to_numpy()
        else:
            self.intercept_ = 0.0
            self.coef_ = beta.to_numpy()
        return self