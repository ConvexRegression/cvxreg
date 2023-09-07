"""
'cvxreg.datasets' is a module that provides functions to load datasets.
"""

from .load import (
    elect_firms,
    front_firms,
    Boston_housing,
)


__all__ = [
    'elect_firms',
    'front_firms',
    'Boston_housing',
]