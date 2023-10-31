import jax.numpy as jnp
from jax import vmap


def get_dataset(n_t=200, n_x=128):
    # TODO: add real dataset 
    T = 0.007 # Reduced to 0.007 for better fit of domain. 
    L = 1 # per case 2
    t_star = jnp.linspace(0, T, n_t)
    x_star = jnp.linspace(0, L, n_x)

    # Dummy function to replace analytical solution.
    u_exact_fn = lambda t, x: 0
    u_exact = vmap(vmap(u_exact_fn, (None, 0)), (0, None))(t_star, x_star)
    n_exact = vmap(vmap(u_exact_fn, (None, 0)), (0, None))(t_star, x_star)

    return u_exact, n_exact, t_star, x_star
