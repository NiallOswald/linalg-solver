"""Module containing helper functions that operate on sets of vectors."""

import numpy as np
import sympy as sy


def extend_basis(vecs, field='R'):
    """Return a basis containing the given row vectors."""

    return np.concatenate(vecs, kernel_basis(vecs, field))


def kernel_basis(mat, field='R'):
    """Return a basis of the kernel of a matrix."""
    # Setup iszerofunc for the given field
    iszerofunc = _zerofunc(field)

    # Find the nullspace
    null = mat.nullspace(iszerofunc=iszerofunc)

    return np.transpose(np.hstack(null))


def linearise(vecs, field='R'):
    """Return a set of linearly independent vectors."""
    # Setup iszerofunc for the given field
    iszerofunc = _zerofunc(field)

    # Find linearly independent rows
    mat = sy.Matrix(vecs)
    ind = np.array(mat.T.rref(iszerofunc=iszerofunc)[1])

    return np.array(mat)[ind]


def _zerofunc(field):
    """Return the iszerofunc for a given field."""
    if field == 'R' or field == 'C':
        # Vectors over the real or complex fields
        iszerofunc = lambda x: x.is_zero
    elif sy.isprime(field):
        # Vectors over finite fields
        iszerofunc = lambda x: (x % field) == 0
    else:
        raise ValueError(
            "Unexpected value for field. field should be one of 'R' or 'C', or a prime."
        )

    return iszerofunc
