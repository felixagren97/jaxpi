import os

import ml_collections

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

from jaxpi.utils import restore_checkpoint
import models
from utils import get_dataset


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    
    # Problem setup
    r_0 = 0.005  # inner radius
    r_1 = 0.5      # outer radius
    n_r = 10000    # used to be 128, but increased and kept separate for unique points
    C = 1/(jnp.log(r_0)-jnp.log(r_1))

    # Get  dataset
    u_ref, r_star = get_dataset(r_0, r_1, n_r)

    # Initial condition (TODO: Looks as though this is for t = 0 in their solution, should we have for x = 0)?
    u0 = u_ref[0]
    u1 = u_ref[-1] # need to add to loss as well? 

    # Restore model
    model = models.InversePoisson(config, u0, u1, r_star)
    ckpt_path = os.path.join(workdir, "ckpt", config.wandb.name)
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    # Compute L2 error
    l2_error = model.compute_l2_error(params, u_ref)
    print("L2 error: {:.3e}".format(l2_error))

    u_pred = model.u_pred_fn(params, model.r_star)
    e_pred_fn = jax.vmap(lambda params, r: jax.grad(model.u_net, argnums=1)(params, r), (None, 0))

    #du_dr = jax.grad(model.u_pred_fn) # e = d/dr U
    e_pred = e_pred_fn(params, model.r_star)
    e_ref = C/model.r_star
    # Convert them to NumPy arrays for Matplotlib
    r_star_np = jnp.array(r_star)
    u_pred_np = jnp.array(u_pred)
    u_ref_np = jnp.array(u_ref)

    # Create a Matplotlib figure and axis
    fig = plt.figure(figsize=(18, 8))
    plt.subplot(2, 2, 1)
    plt.xlabel('Radius [m]')
    plt.ylabel('Potential V(r)')
    plt.title('Predicted and Analyical Potential')

    # Plot the prediction values as a solid line
    plt.plot(r_star_np, u_pred_np, label='Prediction', color='blue')

    # Plot the analytical solution as a dashed line
    plt.plot(r_star_np, u_ref_np, linestyle='--', label='Analytical Solution', color='red')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    
    # Set x-axis limits to [r_star[0], r_star[-1]]
    plt.xlim(r_star_np[0], r_star_np[-1])

    # plot absolute errors 
    plt.subplot(2, 2, 3)
    plt.xlabel('Radius [m]')
    plt.ylabel('Potenial [V]')
    plt.title('Absolute Potential Error')

    plt.plot(r_star_np, jnp.abs(u_pred_np - u_ref_np) , label='Absolute error', color='red')
    plt.grid()
    plt.xlim(r_star_np[0], r_star_np[-1])
    plt.tight_layout()

    # plot electrical field
    plt.subplot(2, 2, 2)

    plt.xlabel('Radius [m]')
    plt.ylabel('Electric field [V/m]')
    plt.title('Predicted and Analytical Electrical field')

    # Plot the prediction values as a solid line
    plt.plot(r_star_np, e_pred, label='Prediction', color='blue')

    # Plot the analytical solution as a dashed line
    plt.plot(r_star_np, e_ref, linestyle='--', label='Analytical Solution', color='red')
    plt.grid()
    plt.legend()
    plt.xlim(r_star_np[0], r_star_np[-1])
    plt.tight_layout()

    # plot absolute field errors 
    plt.subplot(2, 2, 4)
    plt.xlabel('Radius [m]')
    plt.ylabel('Electrical field [V/m]')
    plt.title('Absolute Electrical field')

    plt.plot(r_star_np, jnp.abs(e_pred - e_ref) , label='Absolute error', color='red')
    plt.grid()
    plt.xlim(r_star_np[0], r_star_np[-1])
    plt.tight_layout()

    

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    fig_path = os.path.join(save_dir, "laplace.pdf")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
 