"""Module containing vector space related objects."""

import numpy as np
from .basis import linearise, kernel_basis


class Span:
    """A Spanning set over a field."""

    def __init__(self, vecs, field='R'):
        self.basis = linearise(vecs)
        self.dim = self.basis.shape[0]
        self.space = self.basis.shape[1]
        self.field = field

    def __contains__(self, other):
        """Check whether a space contains a vector or subspace."""
        if isinstance(other, np.ndarray):
            return not Span(np.concatenate([self.basis, np.array([other])]), self.field) == self.dim

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

    def __init__(self, mat, field='R'):
        return super().__init__(kernel_basis(mat, field), field)


class QuotientSpace(Span):
    """The quotient space of a subspace and a vectorspace."""

    def __init__(self, space, subspace, field):
        if subspace not in space:
            raise ValueError(
                "The subspace must be contained within the given space."
            )

        self.basis = np.array(Coset(subspace, vec, field) for vec in linearise(space.basis))
        self.dim = space.dim - subspace.dim
        self.field = field

    def __contains__(self, other):
        if isinstance(other, Coset):
            return other.vector in Span([coset.vector for coset in self.basis], self.field)

        else:
            return NotImplemented


class Coset(Span):
    """A right-coset."""

    def __init__(self, subspace, vec):
        self.vector = np.array(vec)
        self.subspace = subspace
        return super().__init__(np.array(self.vector + w for w in subspace.basis), subspace.field)

    def __eq__(self, other):
        return self.vector - other.vector in self.subspace

    def __add__(self, other):
        if isinstance(other, Coset):
            return Coset(self.subspace, self.vec + other, self.field)

        else:
            return NotImplemented


class FieldSpace(Span):
    """A vector space of column vectors."""

    def __init__(self, field, n):
        vecs = np.identity(n)
        return super().__init__(vecs, field)
