import jax.numpy as jnp
from jax import vmap


def get_dataset(n_x):
    L = 1 # per case 2
    x_star = jnp.linspace(0, L, n_x)
    # Dummy function to replace analytical solution.
    u_exact_fn = lambda t, x: 0
    u_exact = vmap(u_exact_fn)(x_star)
    
    return u_exact, x_star