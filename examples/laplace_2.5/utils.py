import jax.numpy as jnp
from jax import vmap
import io
import pandas as pd


def get_dataset(n_x):
    L = 1 # per case 2
    x_star = jnp.linspace(0, L, n_x)
    # Dummy function to replace analytical solution.
    u_exact_fn = lambda x: 0
    u_exact = vmap(u_exact_fn)(x_star)
    
    return u_exact, x_star

def get_reference_dataset(config, e_path, u_path):
    # Load data
    # Read the file, skipping lines starting with "%"
    
    # Reading comsom data for E
    with open(e_path, 'r') as file:
        lines = [line for line in file if not line.startswith('%')]

    # Use StringIO to create a virtual file-like object for pandas to read from
    virtual_file = io.StringIO(''.join(lines))

    # Read data into pandas DataFrame
    df_E = pd.read_csv(virtual_file, delim_whitespace=True, names=['x', 'E'])

    x_ref = df_E['x'].values
    E_ref = df_E['E'].values


    # Reading comsom data for U
    with open(u_path, 'r') as file:
        lines = [line for line in file if not line.startswith('%')]

    # Use StringIO to create a virtual file-like object for pandas to read from
    virtual_file = io.StringIO(''.join(lines))

    # Read data into pandas DataFrame
    df_U = pd.read_csv(virtual_file, delim_whitespace=True, names=['x', 'U'])

    x_ref = df_U['x'].values
    u_ref = df_U['U'].values

    return x_ref, E_ref, u_ref