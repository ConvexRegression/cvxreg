"""
Solve qp problem using cvxpy package.
"""

# Author: Zhiqiang Liao @ Aalto University <zhiqiang.liao@aalto.fi>
# License: MIT

from cvxpy import Minimize, Problem, installed_solvers


def solve_model(objective, constraints, solver):
    """Solve model using cvxpy package.
    """

    solver = solver.upper()
    assert_installed_solver(solver)

    # optimization problem
    if solver is None:
        solver = 'ECOS'
    
    prob = Problem(Minimize(objective), constraints)

    return prob.solve(solver=solver)


def assert_installed_solver(solver):
    if solver.upper() not in installed_solvers():
        raise ValueError("The solver %s is not installed." % solver)