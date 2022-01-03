"""Module containing vector space related objects."""

import numpy as np
from .basis import linearise, kernel_basis
from numbers import Number


class Span:
    """A Spanning set over a field."""

    def __init__(self, vecs, field='R'):  # noqa:D107
        self.basis = linearise(vecs)
        self.dim = self.basis.shape[0]
        self.space = self.basis.shape[1]
        self.field = field

    def __contains__(self, other):
        """Check whether a space contains a vector."""
        if isinstance(other, np.ndarray):
            return not Span(np.concatenate(
                [self.basis, np.array([other])]), self.field) == self.dim

        else:
            return NotImplemented

    def __add__(self, other):
        """Return the sum of the two subspaces."""
        if isinstance(other, np.ndarray):
            return Coset(self, other, self.field)

        elif isinstance(other, Span):
            if self.field != other.field:
                raise TypeError("Subspaces must be over the same field.")

            return Span(np.concatenate([self.basis, other.basis]), self.field)

        else:
            return NotImplemented


class Kernel(Span):
    """The kernel of a matrix over a field."""

    def __init__(self, mat, field='R'):  # noqa:D107
        return super().__init__(kernel_basis(mat, field), field)


class QuotientSpace(Span):
    """The quotient space of a subspace and a vectorspace."""

    def __init__(self, space, subspace):  # noqa:D107
        if subspace not in space:
            raise ValueError(
                "The subspace must be contained within the given space.")

        if subspace.field != space.field:
            raise ValueError(
                "The subspace and space must be over the same field.")

        self.field = space.field
        self.basis = np.array([
            Coset(subspace, vec)
            for vec in kernel_basis(subspace.basis, self.field)
        ])
        self.dim = space.dim - subspace.dim

    def __contains__(self, other):
        """Check whether a quotient space contains a coset."""
        if isinstance(other, Coset):
            return other.vector in Span([coset.vector for coset in self.basis],
                                        self.field)

        else:
            return NotImplemented


class Coset(Span):
    """A right-coset."""

    def __init__(self, subspace, vec):  # noqa:D107
        self.vector = np.array(vec)
        self.subspace = subspace
        return super().__init__(
            np.array([self.vector + w for w in subspace.basis]),
            subspace.field)

    def __eq__(self, other):
        """Check whether two cosets are equivalent."""
        return self.vector - other.vector in self.subspace

    def __add__(self, other):
        """Return the sum of two cosets."""
        if isinstance(other, Coset):
            return Coset(self.subspace, self.vector + other.vector)

        else:
            return NotImplemented

    def __mul__(self, other):
        """Return the scalar multiple of a coset."""
        if isinstance(other, Number):
            return Coset(self.subspace, other * self.vector)

        else:
            return NotImplemented

    def __rmul__(self, other):
        """Return the scalar multiple of a coset."""
        if isinstance(other, Number):
            return Coset(self.subspace, other * self.vector)

        else:
            return NotImplemented


class CyclicSubspace(Span):
    """The cyclic subspace of a vector with respect to a map."""

    def __init__(self, vec, mat, field='R'):  # noqa:D107
        basis = cyclic_basis(vec, mat, field)
        return super().__init__(basis, field)


class FieldSpace(Span):
    """A vector space of column vectors."""

    def __init__(self, field, n):  # noqa:D107
        vecs = np.identity(n)
        return super().__init__(vecs, field)


def primary_decomposition(mat, field):
    """Return the primary decomposition of a matrix."""


def cyclic_basis(vec, mat, field):
    """Return the basis of a cyclic subspace."""
    subspace = Span([vec], field)

    while not (image := mat @ np.transpose(vec)) in subspace:
        subspace = subspace + Span([image], field)

    return subspace.basis
