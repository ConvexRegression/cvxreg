import numpy as np
import inspect

from .utils.check import check_x_y
from .utils.check import check_array
from .utils.check import _check_y
from .utils._param_check import validate_parameter_constraints

class BaseEstimator:
    """Base class for all estimators in cvxreg."""

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        init_signature = inspect.signature(cls.__init__)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "cvxreg estimators should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])
    
    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def _validate_data(self, 
                       x="novalidattion", 
                       y="novalidation"):
        """Validate input data.
        parameters
        ----------
        x : input object to be checked
        if novalidation, x will not be validated.
        y : input object to be checked shape of y should be (n, )
        """
        no_val_x = isinstance(x, str) and x == "novalidation"
        no_val_y = y is None or isinstance(y, str) and y == "novalidation"

        if no_val_x and no_val_y:
            raise ValueError("Please input x or y.")
        elif not no_val_x and no_val_y:
            x = check_array(x)
            out = x
        elif no_val_x and not no_val_y:
            y = _check_y(y)
            out = y
        else:
            x, y = check_x_y(x, y)
            out = x, y
        return out
    
    def _validate_params(self):
        """Validate types and values of constructor parameters

        The expected type and values must be defined in the `_parameter_constraints`
        class attribute, which is a dictionary `param_name: list of constraints`. See
        the docstring of `validate_parameter_constraints` for a description of the
        accepted constraints.
        """
        validate_parameter_constraints(
            self._parameter_constraints,
            self.get_params(deep=False),
            caller_name=self.__class__.__name__,
        )