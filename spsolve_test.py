import autograd.numpy as np
import numpy.random as npr
from autograd.extend import primitive, defvjp
from scipy.sparse.linalg import spsolve
from autograd.test_util import check_grads
from autograd import grad, jacobian
from copy import deepcopy
from autograd.scipy.sparse.sparse_wrapper import csc_matrix, dia_matrix
import scipy.sparse as sp
from functools import partial
from autograd.numpy import numpy_wrapper as anp
from autograd.core import SparseObject, sparse_add
from time import time

# transpose by swapping last two dimensions
def T(x): return anp.swapaxes(x, -1, -2)

# fancy dot product?
_dot = partial(anp.einsum, '...ij,...jk->...ik')

D = 200*200

# matrix to solve (Ax = b)
A = 10.0 * np.eye(D) + npr.random((D, D))
A = csc_matrix(A)

# source
b = npr.random((D, 1))

t = time()
# forward solve
@primitive
def sparsesolve(a, b):
    # a = csr_matrix(a)
    return spsolve(a, b)

# vector-jacobian product
def grad_sparsesolve(argnum, ans, a, b):
    # a = csr_matrix(a)    
    updim = lambda x: x if x.ndim == a.ndim else x[...,None]
    if argnum == 0:
        return lambda g: - np.dot(updim(spsolve(a.T, g)), T(updim(ans)))
    else:
        return lambda g: - spsolve(a.T, g)

# add to autograd
defvjp(sparsesolve, partial(grad_sparsesolve, 0), partial(grad_sparsesolve, 1))
print('autograd def took {} seconds'.format(time() - t))

# function of A, b.  returns jacobian of x = spsolve(A, b) w.r.t A
t = time()
jac = jacobian(sparsesolve)
print('jacobian setup took {} seconds'.format(time() - t))

# solution of x = spsolve(A, b)
t = time()
x_orig = sparsesolve(A, b)
print('solve took {} seconds'.format(time() - t))

# jacobian evaluated at A, b.  Tensor of shape (D x D x D)
t = time()
jac_analytical = jac(A, b)
print('jacobian creation took {} seconds'.format(time() - t))

# make copy of A with a perturbed element at position (1,3)
t = time()
epsilon = 1e-4
A_new = deepcopy(A)
A_new[1, 3] += epsilon

# compute new solution to Ax = b
x_new = sparsesolve(A_new, b)

# compute derivative of solution w.r.t element (1,3) of A
grad_numerical = (x_new - x_orig) / epsilon
print('numerical gradient took {} seconds'.format(time() - t))

# analytical gradient is just the jacobian evaluated at (:, 1, 3)
grad_analytical = jac_analytical[:, 1, 3]


# print the respective gradients.  Should be identical
print('')
for d in range(min(D, 10)):
    print('d = {}:   autograd = {}, \tnumerical = {}'.format(d, grad_analytical[d], grad_numerical[d]))
"""
I get this output
    d = 0:   autograd = -0.0008400996855799314, numerical = -0.0008400997827950896
    d = 1:   autograd = -0.007347497229732127, numerical = -0.00734749807988766
    d = 2:   autograd = -0.0006018979434279238, numerical = -0.0006018980130625962
    d = 3:   autograd = 8.100330722175541e-05, numerical = 8.100331655436221e-05
    d = 4:   autograd = 5.962900322764243e-06, numerical = 5.962901006295596e-06
"""    

## Here I construct an objective function of a sparse matrix 'eps' 
def J(eps, b):       
    A_total = A0 + eps  
    x = sparsesolve(A_total, b)
    return np.sum(np.abs(x))

dJdeps = grad(J, 0)

eps = np.array(npr.random((D,)))
row = np.array([d for d in range(D)])
col = np.array([d for d in range(D)]) 
eps = csc_matrix((eps, (row, col)), shape=(D,D))
A0 = csc_matrix(npr.random((D, D)))

J_orig = J(eps, b)

grad_numerical = np.zeros((D,))
for d in range(D):
    eps_new = deepcopy(eps)
    eps_new[d, d] += epsilon
    J_new = J(eps_new, b)
    grad_numerical[d] = (J_new - J_orig) / epsilon

grad_analytical = dJdeps(eps, b)

print('')
for d in range(min(D, 10)):
    print('d = {}:   autograd = {}, \tnumerical = {}'.format(d, grad_analytical[d, d], grad_numerical[d]))
"""
I get this output
    d = 0:   autograd = 1.0354755007339855, numerical = 1.0357903948854386
    d = 1:   autograd = 0.3591787952642103, numerical = 0.3591600354191371
    d = 2:   autograd = -1.0999449413589661, numerical = -1.099767966472598
    d = 3:   autograd = -0.07852505200249746, numerical = -0.07852005994291034
    d = 4:   autograd = 2.3138168023220933, numerical = 2.3137619155888522
"""



