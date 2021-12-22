"""Module containing helper functions that operate on polynomials."""

import numpy as np


def field_roots(poly, field='R'):
    """Return the roots of a polynomial over an arbitrary field."""
    import sympy as sy

    if field == 'R':
        return sy.real_roots(poly)

    elif field == 'C':
        deg = sy.degree(poly)
        return [poly.root(i) for i in range(deg)]

    else:
        fact = sy.factor(poly, modulus=field)
        real_roots = sy.real_roots(fact)

        return np.array([root % field for root in real_roots])


def _zerofunc(field):
    """Return the iszerofunc for a given field."""
    from sympy import isprime

    if field == 'R' or field == 'C':
        # Vectors over the real or complex fields
        iszerofunc = lambda x: x.is_zero  # noqa:E731
    elif isprime(field):
        # Vectors over finite fields
        iszerofunc = lambda x: (x % field) == 0  # noqa:E731
    else:
        raise ValueError(
            "Unexpected value for field. field should be one of 'R' or 'C', "
            "or a prime."
        )

    return iszerofunc
