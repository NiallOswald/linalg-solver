"""Module for finding favorable forms of matrices."""

from .polynomials import field_roots, as_factors
from .basis import kernel_basis, extend_basis
from .spaces import FieldSpace, Kernel, Span, QuotientSpace
import numpy as np


def triangularise(mat, field="R", accuracy=10):
    """Triangularise a matrix over the given field."""
    n = mat.shape[0]
    space = FieldSpace(field, n)

    w = field_eigvecs(mat, field)[0][0]
    subspace = Span([w])
    basis = extend_basis(subspace.basis, field)

    for i in range(1, n - 1):
        q_space = QuotientSpace(space, subspace)
        q_basis = q_space.basis

        # Rounding the matrices to minimise the effects of errors
        map_basis = np.around(
            basis_change(mat, space.basis, basis), decimals=accuracy
        )
        qmap_basis = map_basis[np.ix_(np.arange(i, n), np.arange(i, n))]

        w_field = field_eigvecs(qmap_basis, field)[0][0]
        w_coset = np.dot(q_basis, w_field)
        w = w_coset.vector

        subspace = subspace + Span([w])
        basis = extend_basis(subspace.basis, field)

    return basis


def jordan_canonical_form(mat, field="R"):
    """Find the Jordan canonical form of a matrix."""
    n = mat.shape[0]

    # Find the max block sizes and corresponding eigenvalues
    factors = as_factors(min_poly(mat, field))
    e_vals = [poly.root(0) for poly in factors[:, 0]]
    max_sizes = factors[:, 1]

    # Find the subspace dimensions
    decomposition = primary_decomposition(mat, field)

    # Initialise JCF list
    jcf = []

    # Iterate over subspaces
    for eig_val, r, subspace in zip(e_vals, max_sizes, decomposition):
        l = n - subspace.dim
        nilpotent_map = mat - eig_val * np.eye(n)
        block_numbers = np.zeros(r)

        # Iterate over possible block sizes
        for i in range(1, r):
            k = n - Kernel(np.linalg.matrix_power(nilpotent_map, r - i)).dim
            block_numbers[r - i] = (
                k - l - np.dot(np.arange(2, i + 1), block_numbers[r - i + 1 :])
            )

        block_numbers[0] = (
            n - l - np.dot(np.arange(2, r + 1), block_numbers[1:])
        )

        jcf.append((eig_val, block_numbers))

    return jcf


def basis_change(mat, basis, new_basis):
    """Find the matrix with respect to the new_basis."""
    b1 = basis_change_matrix(basis)
    b2 = basis_change_matrix(new_basis)

    return np.linalg.inv(b2) @ b1 @ mat @ np.linalg.inv(b1) @ b2


def basis_change_matrix(basis):
    """Return the change of basis matrix from given to canonical basis."""
    return np.transpose(basis)


def field_eigvals(mat, field="R", accuracy=10):
    """Return the eigenvalues of a matrix over an arbitrary field."""
    from sympy import Matrix, Symbol

    x = Symbol("x")
    char = Matrix(mat).charpoly(x)
    eigvals = np.asarray(field_roots(char, field), dtype="float64")

    return np.unique(np.around(eigvals, decimals=accuracy))


def field_eigvecs(mat, field="R"):
    """Return the eigenvectors of a matrix over an arbitrary field."""
    eigvals = field_eigvals(mat, field)

    eigvecs = np.array(
        [
            kernel_basis(mat - x * np.identity(mat.shape[0]), field)
            for x in eigvals
        ]
    )

    return eigvecs


def poly_matrix(poly, mat):
    """Substitute a matrix into a polynomial."""
    coeffs = poly.all_coeffs()[::-1]

    res = coeffs[0] * np.identity(mat.shape[0])
    for i in range(1, len(coeffs)):
        res += coeffs[i] * np.linalg.matrix_power(mat, i)

    return res


def min_poly(mat, field="R"):
    """Return the minimal polynomial of a matrix."""
    import sympy as sy

    # Find the characteristic polynomial
    x = sy.Symbol("x")
    char = sy.Matrix(mat).charpoly(x)

    # Split the characteristic polynomial into irreducible factors
    factors = as_factors(char, field)

    # Find the minimal polynomial
    trial_deg = factors[:, 1]

    # Iterate over the factors
    for i in range(factors.shape[0]):
        trial_poly = np.prod(factors[:, 0] ** trial_deg)
        val = poly_matrix(trial_poly, mat).astype("float64")

        # Convert the output matrix to be over the correct field
        if not (field == "R" or field == "C"):
            val = np.mod(val, field)

        # Reduce the degree until the evaluation is non-zero
        while not val.any():
            trial_deg[i] -= 1
            trial_poly = np.prod(factors[:, 0] ** trial_deg)
            val = poly_matrix(trial_poly, mat)

        trial_deg[i] += 1

    return np.prod(factors[:, 0] ** trial_deg)


def primary_decomposition(mat, field="R"):
    """Return the primary decomposition of a matrix."""
    # Find the irreducible factors in the minimal polynomial
    factors = as_factors(min_poly(mat), field)

    # Find the primary decomposition
    subspaces = [
        Kernel(poly_matrix(poly**deg, mat), field) for poly, deg in factors
    ]

    return subspaces
