"""
Load datasets from the cvxreg data folder.
"""

# Author: Zhiqiang Liao @ Aalto University
# License: MIT

import os
import numpy as np
import pandas as pd

from sklearn.utils import Bunch

cvxreg_data_path = os.path.dirname(os.path.abspath(__file__))

def _open_text(data_file_name):
    return open(os.path.join(cvxreg_data_path, data_file_name), 'r')

def _convert_to_dataframe(data, target, feature_names, target_names):

    data_df = pd.DataFrame(data, columns=feature_names)
    target_df = pd.DataFrame(target, columns=target_names)
    combined_df = pd.concat([data_df, target_df], axis=1)
    X = combined_df[feature_names]
    y = combined_df[target_names]

    return combined_df, X, y

def elect_firms(return_X_y=False, as_frame=False):
    """
    Load the electricity dataset.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features)
        The data matrix.
    target : ndarray of shape (n_samples,)
        The target vector.
    target_names : ndarray of shape (n_targets,)
        The names of the targets.
    """
    data_file_name = 'data/electricityFirms.csv'

    with _open_text(data_file_name) as f:
        data_file = np.loadtxt(f, delimiter=',', skiprows=1)
    
    feature_names = ['Energy', 'Length', 'Customers']

    frame = None
    target_names = ['OPEX', 'CAPEX', 'TOTEX']

    data = data_file[:, -4:-1]
    target = data_file[:, :-4]

    if as_frame:
        frame, data, target = _convert_to_dataframe(
            data, target, feature_names, target_names
        )
   
    if return_X_y:
        return data, target
    
    return Bunch(
        data=data,
        target=target,
        frame=frame,
        feature_names=feature_names,
        target_names=target_names,

    )

def front_firms(return_X_y=False, as_frame=False):
    """
    Load the 41 firms dataset.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features)
        The data matrix.
    target : ndarray of shape (n_samples,)
        The target vector.
    target_names : ndarray of shape (n_targets,)
        The names of the targets.
    """
    data_file_name = 'data/41Firm.csv'

    with _open_text(data_file_name) as f:
        data_file = np.loadtxt(f, delimiter=',', skiprows=1)
    
    feature_names = ['capital', 'labour']

    frame = None
    target_names = ['output']

    data = data_file[:, -2:]
    target = data_file[:, 1]

    if as_frame:
        frame, data, target = _convert_to_dataframe(
            data, target, feature_names, target_names
        )
   
    if return_X_y:
        return data, target
    
    return Bunch(
        data=data,
        target=target,
        frame=frame,
        feature_names=feature_names,
        target_names=target_names,

    )

def Boston_housing(return_X_y=False, as_frame=False):
    """
    Load the Boston dataset.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features)
        The data matrix.
    target : ndarray of shape (n_samples,)
        The target vector.
    target_names : ndarray of shape (n_targets,)
        The names of the targets.
    """
    data_file_name = 'data/Boston.csv'

    with _open_text(data_file_name) as f:
        data_file = np.loadtxt(f, delimiter=',', skiprows=1)
    
    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]

    frame = None
    target_names = ['MEDV']

    data = data_file[:, :-1]
    target = data_file[:, -1]

    if as_frame:
        frame, data, target = _convert_to_dataframe(
            data, target, feature_names, target_names
        )
   
    if return_X_y:
        return data, target
    
    return Bunch(
        data=data,
        target=target,
        frame=frame,
        feature_names=feature_names,
        target_names=target_names,

    )