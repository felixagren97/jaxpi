import jax.numpy as jnp
from jax import vmap
import pandas as pd
import re


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

def get_reference_dataset(config, file_path):
    injection = format(config.setting.n_inj, '.0e').replace('e+0', 'e').replace('e+', 'e').replace('e', 'E') # convert 5e9 to '5E9'

    # read file
    header = pd.read_csv(file_path, nrows=1, skiprows=7, header=None, delimiter=',\s*', engine='python')
    data = pd.read_csv(file_path, skiprows=8, delim_whitespace=True, header=None)
    data.columns = header.iloc[0, :-1] # remove last entry that is just injection

    # bad 
    header_list = data.columns.tolist()

    # Find rows in the header that contain '5E13'
    matching_columns = ['x'] + [col for col in header_list if injection in col]
    filtered_data = data[matching_columns]
    filtered_data.columns =  filtered_data.columns.str.extract(r'(t=\d+\.\d+)', expand=False) # reformat as "t=0.00x"
    # rename two exception cases
    filtered_data.columns.values[0] = 'x'
    filtered_data.columns.values[-1] = 't=2E-6'
    filtered_data.insert(1, 't=2E-6', filtered_data.pop('t=2E-6')) # move t=2E-6 to front

    # Extract x_star and t_star
    x_star = jnp.array(filtered_data['x'].values)
    t_star = jnp.arange(0.001, 0.008, 0.001)
    t_star = jnp.insert(t_star,0, 1e-6)

    # Imitate vmap
    u_ref = jnp.array(filtered_data.values[:,1:].T) # transpose to get time as rows and space as columns, remove 0:th x column
    return t_star, x_star, u_ref