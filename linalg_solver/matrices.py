"""Module containing multiple types of common matrices."""

import numpy as np
import sympy as sy


def jordan_block(x, n):
    """Return a x-Jordan block of size n."""
    return x * np.eye(n) + np.eye(n, k=-1)


def companion(poly):
    """Return the companion matrix of a polynomial."""
    n = sy.degree(poly)

    return np.eye(n, k=-1) - np.hstack((np.zeros(
        (n, n - 1)), np.transpose([poly.all_coeffs()[:0:-1]
                                   ]))).astype('float64')
