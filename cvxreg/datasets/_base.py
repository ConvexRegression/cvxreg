"""
Base functions for all datasets.
"""

# Author: Zhiqiang Liao @ Aalto University
# License: MIT

import sys
from importlib import resources

import os
import numpy as np
import pandas as pd
import csv

cvxreg_data_path = os.path.dirname(os.path.abspath(__file__))

def _open_text(data_file_name):
    return open(os.path.join(cvxreg_data_path, data_file_name), 'r')

def load_csv_data(data_file_name):
    """
    Load data from a csv file.

    Parameters
    ----------
    data_file_name : string
        The name of the csv file.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features)
        The data matrix.
    target : ndarray of shape (n_samples,)
        The target vector.
    """

    with _open_text(data_file_name) as f:
        data_file = csv.reader(f)
        header = next(data_file)
        n_samples = int(header[0])
        n_features = int(header[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,))
        for i, sample in enumerate(data_file):
            data[i] = np.asarray(sample[:-1], dtype=np.float64)
            target[i] = np.asarray(sample[-1], dtype=np.float64)

    return data, target

def load_electricity_firm(return_X_y, as_frame):
    """
    Load the electricity firm dataset.

    This dataset has multiple inputs and multiple outputs.

    ==============   ==============
    Samples total    89
    Dimensionality   3(both for input and output)
    Features         real
    Targets          real
    ==============   ==============

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is a pandas DataFrame or
        Series depending on the number of target columns. If `return_X_y`
        is True, then (`data`, `target`) will be pandas DataFrames or Series
        as described above.
    """

    file_name = 'electricity_firm.csv'

    data, target = load_csv_data(file_name)

    feature_names = [
        'Energy', 
        'Length', 
        'Customers', 
        'OPEX', 
        'CAPEX', 
        'TOTEX',
        ]
    
    frame = None
    target_columns = None