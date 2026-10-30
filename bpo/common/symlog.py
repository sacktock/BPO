import numpy as np
import jax.numpy as jnp
from jax import jit

def symlog(x):
    return np.sign(x) * np.log(1 + np.abs(x))

def symexp(x):
    return np.sign(x) * (np.exp(np.abs(x)) - 1)

@jit
def jsymlog(x):
    return jnp.sign(x) * jnp.log(1 + jnp.abs(x))

@jit
def jsymexp(x):
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)