import jax.numpy as jnp
from jax import vmap


def get_dataset(r_0=0.001, r_1=1, n_r=128):
    r_star = jnp.linspace(r_0, r_1, n_r)
    #C = 1/(jnp.log(r_0)-jnp.log(r_1))
    #A = -jnp.log(r_1)*C
    r0 = r_star[0]
    r1 = r_star[-1]

    eps = 8.85e-12
    true_rho = 5e-10

    C_1 = ((4*eps*jnp.log(r1) + true_rho * r0**2 * jnp.log(r1) - true_rho * r1**2 * jnp.log(r0)) /
       (4 * eps * (-jnp.log(r0) + jnp.log(r1))))
    
    C_2 = (-4 * eps - true_rho*r0**2 + true_rho * r1**2) / (4 * eps * (-jnp.log(r0) + jnp.log(r1)))

    
    u_exact_fn = lambda r: C_1 + C_2 * jnp.log(r) - (true_rho * r**2) / (4 * eps)
    u_exact = vmap(u_exact_fn)(r_star)
    
    return u_exact, r_star