from autograd.extend import defvjp
from .sparse_wrapper import dot

def dot_vjp(ans, sparse, dense):
    return lambda g: g * dot(sparse, dense)


defvjp(dot, dot_vjp)
