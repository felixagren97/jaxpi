import jax.numpy as jnp
from jax import vmap
import pandas as pd


def get_dataset(n_t=200, n_x=128):
    T = 0.007 # Reduced to 0.007 frpom 0.01 for better fit of domain. 
    L = 1 # per case 2
    t_star = jnp.linspace(0, T, n_t)
    x_star = jnp.linspace(0, L, n_x)

    # Dummy function to replace analytical solution.
    u_exact_fn = lambda t, x: 0
    u_exact = vmap(vmap(u_exact_fn, (None, 0)), (0, None))(t_star, x_star)
    n_exact = vmap(vmap(u_exact_fn, (None, 0)), (0, None))(t_star, x_star)

    return u_exact, n_exact, t_star, x_star

def get_reference_dataset(config, file_path):
    # Load data
    data = pd.read_csv(file_path, skiprows=8, delim_whitespace=True, header=None)

    # Assign hard coded header
    part_header = ['t=2e-6'] + [f't={i}e-3' for i in range(1,8)]
    header = ['x'] + part_header * 4
    data.columns = header

    x_star = data['x'].values
    t_star = jnp.arange(0.001, 0.008, 0.001)
    t_star = jnp.insert(t_star,0, 1e-6)

    # Filter out columns corresponding to current injection
    col_range = list(range(1,9))
    column_range = {
        5e9 : col_range,
        5e13 : [x +     len(col_range) for x in col_range],
        1e14 : [x + 2 * len(col_range) for x in col_range],
        5e15 : [x + 3 * len(col_range) for x in col_range]
    }
    data = data.iloc[:, column_range[config.setting.n_inj]]
    u_ref = data.values.T # Transpose to get time as rows and space as columns
    
    return t_star, x_star, u_ref


def get_analytical_n_ref(config, t_star, x_star):
    # Define variables
    mu = 2e-4
    E_ext = config.setting.u_0
    n_inj = config.setting.n_inj
    n_0 = config.setting.n_0
    
    # Make predictions
    n_exact_fn = lambda t, x: jnp.where(x <= E_ext * mu * t, n_inj, n_0)
    n_exact = vmap(vmap(n_exact_fn, (None, 0)), (0, None))(t_star, x_star)
    return n_exact