import jax.numpy as jnp
from jax import vmap


def get_dataset(r_0=0.001, r_1=0.5, n_r=128, true_rho=5e-10):
    r_star = jnp.linspace(r_0, r_1, n_r)

    r0 = r_star[0]
    r1 = r_star[-1]

    eps = 8.85e-12

    C_1 = ((4*eps*jnp.log(r1) + true_rho * r0**2 * jnp.log(r1) - true_rho * r1**2 * jnp.log(r0)) /
       (4 * eps * (-jnp.log(r0) + jnp.log(r1))))
    
    C_2 = (-4 * eps - true_rho*r0**2 + true_rho * r1**2) / (4 * eps * (-jnp.log(r0) + jnp.log(r1)))

    
    u_exact_fn = lambda r: C_1 + C_2 * jnp.log(r) - (true_rho * r**2) / (4 * eps)
    u_exact = vmap(u_exact_fn)(r_star)
    
    return u_exact, r_star

def get_observations(n_obs):
   # open numpy array from file called obs.dat
   obs = jnp.load('obs.dat')
   obs_x = obs[:,0]
   obs_u = obs[:,1]

   #selct n_obs random indices
   idx = jnp.random.randint(0, len(obs_x), n_obs)
   obs_x = obs_x[idx]
   obs_u = obs_u[idx]

   return obs_x, obs_u

