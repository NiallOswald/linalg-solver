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


def as_factors(poly, field='R'):
    """Split a polynomial into irreducibles."""
    import sympy as sy

    x = sy.Symbol('x')

    # Cases on fields
    if field == 'R':
        factors_nonmon = sy.factor_list(poly, domain='RR')[1]

        factors = np.array([(sy.monic(factor), deg)
                            for factor, deg in factors_nonmon])

    elif field == 'C':
        roots = field_roots(poly, field='C')
        factors_rep = np.array([x - root for root in roots])

        factors = np.array([(factor, np.sum(factors_rep == factor))
                            for factor in np.unique(factors_rep)])

    else:
        factors_split = np.array(sy.factor_list(poly, modulus=field))
        factor_list, degree_list = factors_split[:, 0], factors_split[:, 1]
        degree_unique = np.array([
            np.sum(degree_list[np.where(factor_list == factor)])
            for factor in np.unique(factor_list)
        ])

        factors = np.column_stack(np.unique(factor_list), degree_unique)

    return factors


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
            "or a prime.")

    return iszerofunc
