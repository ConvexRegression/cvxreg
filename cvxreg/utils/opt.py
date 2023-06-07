from re import compile
from os import environ
__email_re = compile(r'([^@]+@[^@]+\.[a-zA-Z0-9]+)$')

from pyomo.environ import SolverFactory, SolverManagerFactory
from .check import check_local_solver

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


