"""
Functions for optimization with Pyomo.
"""

# Author: Zhiqiang Liao @ Aalto University <zhiqiang.liao@aalto.fi>
# License: MIT

from re import compile
from os import environ
__email_re = compile(r'([^@]+@[^@]+\.[a-zA-Z0-9]+)$')

from pyomo.environ import SolverFactory, SolverManagerFactory

def set_neos_email(address):
    """pass email address to NEOS server 

    Args:
        address (String): your own vaild email address.
    """
    if address == None:
        return False
    if not __email_re.match(address):
        raise ValueError("Invalid email address.")
    environ['NEOS_EMAIL'] = address
    return True

def optimize_model(model, email, solver):
    if not set_neos_email(email):
        if solver is not None:
            check_local_solver(solver)
        else:
            solver = "mosek"
        solver_instance = SolverFactory(solver)
        return solver_instance.solve(model, tee=False), 1
    else:
        if solver is None:
            solver = "mosek"
        solver_instance = SolverManagerFactory('neos')
        return solver_instance.solve(model, tee=False, opt=solver), 1

def check_optimization_status(optimization_status):
    if optimization_status == 0:
        raise Exception(
            "Model isn't optimized. Use optimize() method to estimate the model.")

def check_local_solver(solver):
    if not SolverFactory(solver).available():
        raise ValueError("Solver {} is not available locally.".format(solver))
