import jax.numpy as jnp
from jax import vmap


def get_dataset(r_0=0.001, r_1=1, n_r=128):
    r_star = jnp.linspace(r_0, r_1, n_r)
    C = 1/(jnp.log(r_0)-jnp.log(r_1))
    A = -jnp.log(r_1)*C
    
    u_exact_fn = lambda r: C*jnp.log(r)+A
    u_exact = vmap(u_exact_fn)(jnp.exp(r_star))
    
    return u_exact, r_star