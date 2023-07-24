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
    """Check if the solver is installed.
    """
    SOLVER = solver.upper()
    if SOLVER not in installed_solvers():
        raise ValueError("The solver %s is not installed." % solver)
    
    if SOLVER == 'ECOS':
        return ECOS
    elif SOLVER == 'MOSEK':
        return MOSEK
    elif SOLVER == 'OSQP':
        return OSQP
    elif SOLVER == 'SCS':
        return SCS
    elif SOLVER == 'CVXOPT':
        return CVXOPT
    elif SOLVER == 'GUROBI':
        return GUROBI
    elif SOLVER == 'CPLEX':
        return CPLEX
    elif SOLVER == 'COPT':
        return COPT
    else:
        raise ValueError("The solver %s is not installed." % SOLVER)
    