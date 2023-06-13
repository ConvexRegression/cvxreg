"""
Solve qp problem using cvxpy package.
"""

# Author: Zhiqiang Liao @ Aalto University <zhiqiang.liao@aalto.fi>
# License: MIT

from cvxpy import Minimize, Problem, installed_solvers
from cvxpy import MOSEK, OSQP, ECOS, SCS, CVXOPT, GUROBI, CPLEX, COPT


def solve_model(objective, constraints, solver):
    """Solve model using cvxpy package.
    """
    solver = check_installed_solver(solver)
    
    prob = Problem(Minimize(objective), constraints)

    return prob.solve(solver=solver)


def check_installed_solver(solver):
    if solver.upper() not in installed_solvers():
        raise ValueError("The solver %s is not installed." % solver)
    
    if solver == 'ecos':
        return ECOS
    elif solver == 'mosek':
        return MOSEK
    elif solver == 'osqp':
        return OSQP
    elif solver == 'scs':
        return SCS
    elif solver == 'cvxopt':
        return CVXOPT
    elif solver == 'gurobi':
        return GUROBI
    elif solver == 'cplex':
        return CPLEX
    elif solver == 'copt':
        return COPT
    else:
        return None
    