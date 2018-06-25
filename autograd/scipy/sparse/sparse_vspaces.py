import scipy.sparse as sp
from autograd.extend import VSpace

class SparseArrayVSpace(VSpace):

    def __init__(self, value):
        self.t = type(value)
        self.value = self.t(value)
        self.shape = value.shape
        self.dtype = value.dtype

    @property
    def size(self):
        return self.value.size

    @property
    def ndim(self):
        return len(self.shape)

    def randn(self):
        a = sp.random(m=self.shape[0], n=self.shape[1])
        return self.t(a)

    def zeros(self):
        return self.t(self.shape)


VSpace.register(sp.csc_matrix, lambda x: SparseArrayVSpace(x))
VSpace.register(sp.csr_matrix, lambda x: SparseArrayVSpace(x))
VSpace.register(sp.coo_matrix, lambda x: SparseArrayVSpace(x))
VSpace.register(sp.dia_matrix, lambda x: SparseArrayVSpace(x))
