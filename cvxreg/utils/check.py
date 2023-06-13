"""
Data input check.
"""

# Author: Zhiqiang Liao @ Aalto University <zhiqiang.liao@aalto.fi>
# License: MIT

import numpy as np

def _num_samples(x):
    """Return number of samples in array-like x."""
    message = "Expected sequence or array-like, got %s" % type(x)
    if hasattr(x, "fit") and callable(x.fit):
        raise TypeError(message)
    if not hasattr(x, "__len__") and not hasattr(x, "shape"):
        if hasattr(x, "__array__"):
            x = np.asarray(x)
        else:
            raise TypeError(message)
    if hasattr(x, "shape"):
        if len(x.shape) == 0:
            raise TypeError(
                "Singleton array %r cannot be considered a valid"
                " collection." % x)
        return x.shape[0]
    else:
        return len(x)
    
def check_array(array, ensure_2d=True):
    """Input check on array and list.
    parameters
    ----------
    array : input object to be checked
    """
    if isinstance(array, np.matrix):
        raise ValueError("Please input a numpy array instead of a numpy matrix.")
    
    
    array = np.array(array)

    if ensure_2d:
        if array.ndim == 0:
            raise ValueError("Please input a 2D array.")
        elif array.ndim == 1:
            raise ValueError("Expected 2D array, got 1D array instead:\narray={}.\n"
                            "Reshape your data either using array.reshape(-1, 1) if "
                            "your data has a single feature or array.reshape(1, -1) ")
    return array
    
def _check_y(y):
    """Input check for y."""
    if isinstance(y, np.matrix):
        raise ValueError("Please input a numpy array instead of a numpy matrix.")
    y = check_array(y, ensure_2d=False)

    shape = y.shape
    if len(shape) == 1:
        return y
    if len(shape) == 2 and shape[1] == 1:
        return y.ravel()
    raise ValueError("y should be a 1d array, got an array of shape {} instead.".format(shape))

def check_consistent_length(*arrays):
    """Check that all arrays have consistent first dimensions.
    Checks whether all objects in arrays have the same shape or length.

    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """
    lengths = [_num_samples(x) for x in arrays if x is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError("Found input variables with inconsistent numbers of samples: %r" % [int(l) for l in lengths])
    
def check_x_y(x, y):
    """Input check for x and y.
    Check x and y for consistent length, enforece x to be 2d and y 1d.

    Parameters
    ----------
    x : {ndarray, list}
        Input data.
    y : {ndarray, list}
        Output data.
    """
    if y is None:
        raise ValueError("y must be specified.")
    x = check_array(x)
    y = _check_y(y)

    check_consistent_length(x, y)

    return x, y

def check_ndarray(x, n, d):
    """Input check for x.
    Check x for consistent length and shape.

    Parameters
    ----------
    x : {ndarray, list}
        Input data.
    n : int
        Number of samples.
    d : int
        Number of features.
    """
    x = check_array(x)
    if x.shape != (n, d):
        raise ValueError("x should be a {}x{} array, got an array of shape {} instead.".format(n, d, x.shape))
    return x