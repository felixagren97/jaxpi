import jax.numpy as jnp
from jax import vmap
import con


def get_dataset(n_t, n_x, true_mu, n_inj, n_0):
    # TODO: add real dataset 
    T = 0.01 # per case 2
    L = 1 # per case 2
    E_ext = 1e6 # per case 2
    t_star = jnp.linspace(0, T, n_t)
    x_star = jnp.linspace(0, L, n_x)

    # Analytical solution
    u_exact_fn = lambda t, x: n_inj if x >= E_ext * true_mu * t else n_0
    
    u_exact = vmap(vmap(u_exact_fn, (None, 0)), (0, None))(t_star, x_star)

    return u_exact, t_star, x_star