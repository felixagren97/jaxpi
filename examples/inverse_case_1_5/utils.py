import jax.numpy as jnp
from jax import vmap


def get_dataset(n_x):
    L = 1 # per case 2
    x_star = jnp.linspace(0, L, n_x)
    # Dummy function to replace analytical solution.
    u_exact_fn = lambda x: 0
    u_exact = vmap(u_exact_fn)(x_star)
    
    return u_exact, x_star

def get_observations(n_obs):
   # open numpy array from file called obs.dat
   obs = jnp.load('obs.dat', allow_pickle=True)
   obs_x = obs[:,0]
   obs_u = obs[:,1]

   #selct n_obs random indices
   idx = jnp.random.randint(0, len(obs_x), n_obs)
   obs_x = obs_x[idx]
   obs_u = obs_u[idx]

   return obs_x, obs_u

