from autograd import elementwise_grad as egrad
import pytest
from autograd import grad
import pdb
import numpy as np
from autograd.test_util import check_grads
import scipy.sparse as sp
from autograd.scipy.sparse.sparse_wrapper import dot


# ----- pytest fixture for sparse arrays ----- #
@pytest.fixture
def eye():
    return sp.eye(5).tocsr()


# ----- tests for array creation ----- #
@pytest.mark.works
@pytest.mark.scipy_sparse
def test_sparse_coo_matrix():
    """This just has to not error out."""
    data = np.array([1, 2, 3]).astype('float32')
    rows = np.array([1, 2, 3]).astype('float32')
    cols = np.array([1, 3, 4]).astype('float32')
    sparse = sp.csr_matrix((data, (rows, cols)))


# ----- tests for array multiplication ----- #
@pytest.mark.works
@pytest.mark.scipy_sparse
def test_sparse_dense_multiplication(eye):
    """This just has to not error out."""
    dense = np.random.random(size=(5, 4))
    eye.dot(dense)


@pytest.mark.test
@pytest.mark.scipy_sparse
def test_sparse_dot(eye):
    dense = np.random.random(size=(5, 5))

    def fun(eye):
        return dot(eye, dense)

    check_grads(fun)(dense)

