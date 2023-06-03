import numpy as np

from .utils.check import check_x_y
from .utils.check import check_array
from .utils.check import _check_y

class BaseEstimator:

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