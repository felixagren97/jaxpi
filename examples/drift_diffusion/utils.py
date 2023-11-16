import jax.numpy as jnp
from jax import vmap

def get_dataset(n_t, n_x, config):
    # Gather constants
    T = 0.007 # per case 2
    L = 1 # per case 2
    
    mu = config.setting.mu_n 
    E_ext = config.setting.E_ext
    n_inj = config.setting.n_inj
    n_0 = config.setting.n_0
    
    t_star = jnp.linspace(0, T, n_t)
    x_star = jnp.linspace(0, L, n_x)

    u_exact_fn = lambda t, x: jnp.where(x <= E_ext * mu * t, n_inj, n_0)
    u_exact = vmap(vmap(u_exact_fn, (None, 0)), (0, None))(t_star, x_star)

    return u_exact, t_star, x_star
