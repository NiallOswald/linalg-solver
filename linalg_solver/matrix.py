"""Module for finding favorable forms of matrices."""

from .polynomials import field_roots
from .basis import kernel_basis, extend_basis
from .spaces import FieldSpace, Span, QuotientSpace
import numpy as np


def triangularise(mat, field='R'):
    """Triangularise a matrix over the given field."""
    n = mat.shape[0]
    space = FieldSpace(field, n)

    w = field_eigvecs(mat, field)[0][0]
    subspace = Span([w])
    basis = extend_basis(subspace.basis, field)

    for i in range(1, n - 1):
        q_space = QuotientSpace(space, subspace)
        q_basis = q_space.basis

        map_basis = basis_change(mat, space.basis, basis)
        qmap_basis = map_basis[np.ix_(np.arange(i, n), np.arange(i, n))]

        w_field = field_eigvecs(qmap_basis, field)[0][0]
        w_coset = np.dot(q_basis, w_field)
        w = w_coset.vector

        subspace = subspace + Span([w])
        basis = extend_basis(subspace.basis, field)

    return basis


def basis_change(mat, basis, new_basis):
    """Find the matrix with respect to the out_basis."""
    b1 = basis_change_matrix(basis)
    b2 = basis_change_matrix(new_basis)

    return np.linalg.inv(b2) @ b1 @ mat @ np.linalg.inv(b1) @ b2


def basis_change_matrix(basis):
    """Return the change of basis matrix from given to canonical basis."""
    return np.transpose(basis)


def field_eigvals(mat, field='R'):
    """Return the eigenvalues of a matrix over an arbitrary field."""
    from sympy import Matrix

    char = Matrix(mat).charpoly()
    return np.unique(field_roots(char, field))


def field_eigvecs(mat, field='R'):
    """Return the eigenvectors of a matrix over an arbitrary field."""
    eigvals = field_eigvals(mat, field)
    eigvecs = np.array([kernel_basis(mat - x*np.identity(mat.shape[0]), field)
                        for x in eigvals])

    return eigvecs
