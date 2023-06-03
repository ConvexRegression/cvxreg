# import dependencies
from pyomo.environ import ConcreteModel, Set, Var, Objective, minimize, Constraint, log
from pyomo.core.expr.numvalue import NumericValue
import numpy as np
import pandas as pd

from ..constant import Convex, Concave, OPT_DEFAULT, OPT_LOCAL
from ..utils import tools


class PCR():
    """
    Penalized Convex Regression (PCR) model
    """

    def __init__(self, x, y, c=1.0, fun=Convex, positive=False, fit_intercept=True):
        """CNLS model

        Args:
            y (float): output variable. 
            x (float): input variables.
            c (float, optional): penalty parameter. Defaults to 1.0.
            fun (String, optional): Convex (convex funtion) or FConcave (concave funtion). Defaults to Convex.
            positive (bool, optional): True if the coefficients are positive. Defaults to False.
            fit_intercept (bool, optional): True if the model should include an intercept. Defaults to True.
        """
        # TODO(error/warning handling): Check the configuration of the model exist
        self.y, self.x = tools.assert_valid_basic_data(y, x)

        self.c = c
        self.fun = fun
        self.fit_intercept = fit_intercept
        self.positive = positive

        # Initialize the CR model
        self.__model__ = ConcreteModel()

        # Initialize the sets
        self.__model__.I = Set(initialize=range(len(self.y)))
        self.__model__.J = Set(initialize=range(len(self.x[0])))

        # Initialize the variables
        self.__model__.alpha = Var(self.__model__.I, doc='alpha')
        self.__model__.beta = Var(self.__model__.I,
                                  self.__model__.J,
                                  bounds=(0.0, None) if self.positive else None,
                                  doc='beta')
        self.__model__.epsilon = Var(self.__model__.I, doc='residual')

        # Setup the objective function and constraints
        self.__model__.objective = Objective(rule=self.__objective_rule(),
                                             sense=minimize,
                                             doc='objective function')
        self.__model__.regression_rule = Constraint(self.__model__.I,
                                                    rule=self.__regression_rule(),
                                                    doc='regression equation')
        self.__model__.afriat_rule = Constraint(self.__model__.I,
                                                self.__model__.I,
                                                rule=self.__afriat_rule(),
                                                doc='afriat inequality')

        # Optimize model
        self.optimization_status = 0
        self.problem_status = 0

    
    def __objective_rule(self):
        """Return the proper objective function"""

        def objective_rule(model):
            return sum(model.epsilon[i] ** 2 for i in model.I)

        return objective_rule

    def __regression_rule(self):
        """Return the proper regression constraint"""
        if self.fit_intercept:

            def regression_rule(model, i):
                return self.y[i] == model.alpha[i] \
                    + sum(model.beta[i, j] * self.x[i][j] for j in model.J) \
                    + model.epsilon[i]

            return regression_rule
        
        elif not self.fit_intercept:

            def regression_rule(model, i):
                return self.y[i] == sum(model.beta[i, j] * self.x[i][j] for j in model.J) \
                    + model.epsilon[i]

            return regression_rule


        raise ValueError("Undefined model parameters.")

    def __afriat_rule(self):
        """Return the proper afriat inequality constraint"""
        if self.fun == Concave:
            __operator = NumericValue.__le__
        elif self.fun == Convex:
            __operator = NumericValue.__ge__

        if self.fit_intercept:

            def afriat_rule(model, i, h):
                if i == h:
                    return Constraint.Skip
                return __operator(
                    model.alpha[i] + sum(model.beta[i, j] * self.x[i][j]
                                            for j in model.J),
                    model.alpha[h] + sum(model.beta[h, j] * self.x[i][j]
                                            for j in model.J))

            return afriat_rule
        
        elif not self.fit_intercept:

            def afriat_rule(model, i, h):
                if i == h:
                    return Constraint.Skip
                return __operator(
                    sum(model.beta[i, j] * self.x[i][j] for j in model.J),
                    sum(model.beta[h, j] * self.x[i][j] for j in model.J))

            return afriat_rule
            

        raise ValueError("Undefined model parameters.")

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
        tools.assert_various_return_to_scale(self.fit_intercept)

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

    def predict(self, x):
        """Return prediction value by array"""
        y_hat = tools.yhat(self.intercept_, self.coef_, x, fun=self.fun)
        return np.asarray(y_hat)
    
    def display_status(self):
        """Display the status of problem"""
        tools.assert_optimized(self.optimization_status)
        print(self.display_status)
