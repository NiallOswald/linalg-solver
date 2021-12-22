"""Module containing helper functions that operate on sets of vectors."""

import numpy as np
from .polynomials import _zerofunc


def extend_basis(vecs, field='R'):
    """Return a basis containing the given row vectors."""
    return np.concatenate([vecs, kernel_basis(vecs, field)])


def kernel_basis(mat, field='R'):
    """Return a basis of the kernel of a matrix."""
    from sympy import Matrix

    # Find the nullspace
    null = Matrix(mat).nullspace(iszerofunc=_zerofunc(field))

    return np.transpose(np.hstack(np.asarray(null, dtype='float64')))


def linearise(vecs, field='R'):
    """Return a set of linearly independent vectors."""
    from sympy import Matrix

    # Find linearly independent rows
    mat = Matrix(vecs)
    ind = np.array(mat.T.rref(iszerofunc=_zerofunc(field))[1])

    return np.asarray(mat, dtype='float64')[ind]
