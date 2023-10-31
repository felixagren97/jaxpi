import jax.numpy as jnp
from jax import vmap
import numpy as np
import jax


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

def get_observations(config):
    n_obs = config.setting.n_obs
    obs_file=config.setting.obs_file
    # open numpy array from file called obs.dat
    obs = np.loadtxt(obs_file)
    obs = jnp.array(obs)

    obs_t = obs[:,0]
    obs_x = obs[:,1]
    obs_u = obs[:,2]

    #selct n_obs random indices
    key = jax.random.PRNGKey(config.seed) 
    idx = jax.random.randint(key, minval=0, maxval=len(obs_x), shape=(n_obs,))
    obs_t = obs_x[idx]
    obs_x = obs_x[idx]
    obs_u = obs_u[idx]

    return obs_t, obs_x, obs_u