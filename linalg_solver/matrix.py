"""Module for finding favorable forms of matrices."""

from .polynomials import field_roots
from .basis import kernel_basis, extend_basis
from .spaces import FieldSpace, Span, QuotientSpace
import numpy as np
from sympy import zeros


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
    from sympy import Matrix, Symbol

    x = Symbol('x')
    char = Matrix(mat).charpoly(x)

    return np.unique(field_roots(char, field))


def field_eigvecs(mat, field='R'):
    """Return the eigenvectors of a matrix over an arbitrary field."""
    eigvals = field_eigvals(mat, field)
    eigvecs = np.array([kernel_basis(mat - x*np.identity(mat.shape[0]), field)
                        for x in eigvals])

    return eigvecs


def poly_matrix(poly, mat):
    """Substitute a matrix into a polynomial."""
    coeffs = poly.all_coeffs()[::-1]

    res = zeros(mat.shape[0])
    for i in range(len(coeffs)):
        res += coeffs[i]*(mat**i)

    return res


def min_poly(mat, field='R'):
    """Return the minimal polynomial of a matrix."""
    import sympy as sy

    # Find the characteristic polynomial
    x = sy.Symbol('x')
    char = sy.Matrix(mat).charpoly(x)

    # Split the characteristic polynomial into irreducibles
    if field == 'R':
        factors_nonmon = sy.factor_list(char, domain='RR')[1]

        factors = np.array([(sy.monic(factor), deg)
                            for factor, deg in factors_nonmon])

    elif field == 'C':
        roots = field_roots(char, field='C')
        factors_rep = np.array([x - root for root in roots])

        factors = np.array([(factor, np.sum(factors_rep == factor))
                            for factor in np.unique(factors_rep)])

    else:
        factors_split = np.array(sy.factor_list(char, modulus=field))
        factor_list, degree_list = factors_split[:, 0], factors_split[:, 1]
        degree_unique = np.array([np.sum(
            degree_list[np.where(factor_list == factor)]
        ) for factor in np.unique(factor_list)])

        factors = np.column_stack(np.unique(factor_list), degree_unique)

    # Find the minimal polynomial
    trial_deg = factors[:, 1]

    # Iterate over the factors
    for i in range(factors.shape[0]):
        trial_poly = np.prod(factors[:, 0] ** trial_deg)
        val = poly_matrix(trial_poly, mat)

        # Reduce the degree until the evaluation is non-zero
        while not np.asarray(val, dtype='float64').any():
            trial_deg[i] -= 1
            trial_poly = np.prod(factors[:, 0] ** trial_deg)
            val = poly_matrix(trial_poly, mat)

        trial_deg[i] += 1

    return np.prod(factors[:, 0] ** trial_deg)
