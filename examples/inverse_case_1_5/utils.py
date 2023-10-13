import jax.numpy as jnp
from jax import vmap
import numpy as np
import jax

def get_dataset(n_x):
    L = 1 # per case 2
    x_star = jnp.linspace(0, L, n_x)
    # Dummy function to replace analytical solution.
    u_exact_fn = lambda x: 0
    u_exact = vmap(u_exact_fn)(x_star)
    
    return u_exact, x_star

def get_observations(n_obs):
   # open numpy array from file called obs.dat
   obs = np.loadtxt('obs.dat')
   obs = jnp.array(obs)

   obs_x = obs[:,0]
   obs_u = obs[:,1]

   #selct n_obs random indices
   key = jax.random.PRNGKey(42) 
   idx = jax.random.randint(key, minval=0, maxval=len(obs_x), shape=(n_obs,))
   obs_x = obs_x[idx]
   obs_u = obs_u[idx]

   return obs_x, obs_u

