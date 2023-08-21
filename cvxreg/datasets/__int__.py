"""
'cvxreg.datasets' is a module that provides functions to load datasets.
"""

from ._base import (
    load_elect_firms,
    load_41_firms,
)

__all__ = [
    'load_elect_firms',
    'load_41_firms',
]