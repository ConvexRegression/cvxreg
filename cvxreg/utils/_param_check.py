from abc import ABC
from abc import abstractmethod
from numbers import Real,Integral

import numpy as np

class InvalidParameterError(ValueError):
    """Exception class to raise if a parameter value is invalid."""

def validate_parameter_constraints(parameter_constraints, params, caller_name):
    """Validate types and values of constructor parameters.

    Parameters
    ----------
    parameter_constraints : dict
        A dictionary `param_name: list of constraints`, where `param_name` is a
        string and the list of constraints is a list of `Constraint` objects.
        See the docstring of `Constraint` for a description of the accepted
        constraints.
        Constraints can be:
        -an Interval, in which case the parameter value must be in the range of numbers
        -a StrOptions, in which case the parameter value must be one of the strings
        -the string 'boolean', in which case the parameter value must be a boolean
        -the string 'array-like', in which case the parameter value must be an array-like object
    params : dict
        A dictionary `param_name: param_value`, where `param_name` is a string
        and `param_value` is the value passed to the constructor.
    caller_name : str
        The name of the calling function, used to format error messages.
    """
    for para_names, para_values in params.items():
        if para_names not in parameter_constraints:
            raise InvalidParameterError(
                "Invalid parameter %s for estimator %s. "
                "Check the list of available parameters "
                "with `estimator.get_params().keys()`." %
                (para_names, caller_name))
        
        constraints = parameter_constraints[para_names]

        if constraints == "novalidation":
            continue

        constraints = [make_constraint(constraint) for constraint in constraints]

        for constraint in constraints:
            if constraint.check(para_values):
                break
        else:
            raise InvalidParameterError(
                f"The parameter {para_names} must be {constraint}."
                f" Got {para_values} instead."
                )
            
def make_constraint(constraint):
    """Make a constraint object from a user-provided value."""

    if isinstance(constraint, type):
        return _Instancesof(constraint)
    if constraint is None:
        return _NoneConstraint()
    if isinstance(constraint, (Interval, StrOptions)):
        return constraint
    if isinstance(constraint, str) and constraint == 'boolean':
        return _booleans()
    raise ValueError("Invalid constraint: %s" % constraint)


class _Constraint(ABC):
    """Base class for parameter constraints."""
    
    def __init__(self):
        self.hidden = False

    @abstractmethod
    def check(self, value):
        """Check that the parameter value satisfies the constraint.

        Parameters
        ----------
        value : object
            The parameter value to check.
        """

    @abstractmethod
    def __repr__(self):
        """Return a string representation of the constraint."""

class RealNotInt(Real):
    """Class representing a real number that is not an integer."""

RealNotInt.register(float)
    

class _Instancesof(_Constraint):
    """Constraint representing a list of allowed types."""

    def __init__(self, types):
        super().__init__()
        self.types = types

    def check(self, value):
        return isinstance(value, self.types)
    
    def __repr__(self):
        return "Instances of %s" % (self.types,)

class _NoneConstraint(_Constraint):
    """Constraint representing the None singleton."""

    def check(self, value):
        return value is None
    
    def __repr__(self):
        return "None"
    
class Interval(_Constraint):
    """Constraint representing a closed interval of numbers.

    parameters
    ----------
    type : {numeric, int, float}
        The type of the parameter values.
    lower : numeric or None
        The lower bound of the interval. If None, there is no lower bound.
    upper : numeric or None
        The upper bound of the interval. If None, there is no upper bound.
    """

    def __init__(self, type, lower, upper):
        super().__init__()
        self.type = type
        self.lower = lower
        self.upper = upper

        self._check_params()
    
    def _check_params(self):
        if self.type not in (Integral, Real, RealNotInt):
            raise ValueError(
                "type must be Integral, Real or RealNotInt"
                "Got %s instead." % self.type)
        if self.lower is not None and not isinstance(self.lower, Real):
            raise ValueError("lower bound must be a real number")
        if self.upper is not None and not isinstance(self.upper, Real):
            raise ValueError("upper bound must be a real number")
        if self.lower is not None and self.upper is not None and self.lower >= self.upper:
            raise ValueError(
                "lower bound must be less than or equal to upper bound."
                " Got %s and %s instead." % (self.lower, self.upper)
                )

    def __contains__(self, value):
        if np.isnan(value):
            return False
        if self.lower is not None and value < self.lower:
            return False
        if self.upper is not None and value > self.upper:
            return False
        return True
    
    def check(self, value):
        if not isinstance(value, self.type):
            return False
        return value in self
    
    def __repr__(self):
        return "Interval(%s, %s)" % (self.lower, self.upper)
    

class Options(_Constraint):
    """Constraint representing a list of allowed options."""

    def __init__(self, type, options):
        super().__init__()
        self.options = options
        self.type = type

    def check(self, value):
        return isinstance(value, self.type) and value in self.options
    
    def __repr__(self):
        return "Options: %s" % (self.options,)

class StrOptions(Options):
    """Constraint representing a list of allowed strings."""

    def __init__(self, options):
        super().__init__(str, options=options)
    

class _booleans(_Constraint):
    """Base class for boolean constraints."""

    def __init__(self):
        super().__init__()
        self._constraint = [
            _Instancesof(bool),
            _Instancesof(np.bool_)
        ]

    def check(self, value):
        return any(constraint.check(value) for constraint in self._constraint)
    
    def __repr__(self):
        return ("Boolean")