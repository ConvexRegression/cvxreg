from numbers import Real

import numpy as np
import pandas as pd
from pystoned import CNLS
from pyomo.environ import Objective, minimize

from ._base import CRModel
from ..constant import convex, concave, OPT_DEFAULT, OPT_LOCAL
from ..utils._pyomo_opt import check_optimization_status, optimize_model
from ..utils._param_check import Interval, StrOptions


class PCR(CRModel, CNLS.CNLS):
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
    email : string, optional (default=None)
        The email address for remote optimization. It will optimize locally if None is given.
    solver : string, optional (default='mosek')
        The solver chosen for optimization. It will optimize with mosek solver if None is given.
    """

    _parameter_constraints: dict = {
        "c": [Interval(Real, 0, None)],
        "shape": [StrOptions({convex, concave})],
        'fit_intercept': ['boolean'],
        'positive': ['boolean'],
        'email': [None, str],
        'solver': [str]
    }

    def __init__(
        self, 
        c=1.0, 
        shape=convex, 
        positive=False, 
        fit_intercept=True,
        email=None, 
        solver='mosek'
    ):
        self.c = c
        self.shape = shape
        self.fit_intercept = fit_intercept
        self.positive = positive
        self.email = email
        self.solver = solver

    def fit(self, x, y):
        """Optimize the function by requested method

        Args:
            email (string): The email address for remote optimization. It will optimize locally if OPT_LOCAL is given.
            solver (string): The solver chosen for optimization. It will optimize with default solver if OPT_DEFAULT is given.
        """
        self._validate_params()
        x, y = self._validate_data(x, y)

        if self.shape == convex:
            fun_var = CNLS.FUN_COST
        elif self.shape == concave:
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
            self.intercept_ = alpha
            self.coef_ = beta.to_numpy()
        else:
            self.intercept_ = 0.0
            self.coef_ = beta.to_numpy()
        return self
    
    def __new_objective_rule(self):
        """return new objective function"""
        def objective_rule(model):
            return sum(model.epsilon[i] ** 2 for i in model.I) \
                + self.c * sum(model.beta[i, j] ** 2 for i in model.I for j in model.J)
        return objective_rule