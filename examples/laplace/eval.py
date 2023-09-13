import os

import ml_collections

import jax.numpy as jnp

import matplotlib.pyplot as plt

from jaxpi.utils import restore_checkpoint

import models
from utils import get_dataset


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    
    # Problem setup
    r_0 = 0.001  # inner radius
    r_1 = 1      # outer radius
    n_r = 10000    # used to be 128, but increased and kept separate for unique points

    # Get  dataset
    u_ref, r_star = get_dataset(r_0, r_1, n_r)

    # Initial condition (TODO: Looks as though this is for t = 0 in their solution, should we have for x = 0)?
    u0 = u_ref[0]
    u1 = u_ref[-1] # need to add to loss as well? 

    # Restore model
    model = models.Laplace(config, u0, u1, r_star)
    ckpt_path = os.path.join(workdir, "ckpt", config.wandb.name)
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    # Compute L2 error
    l2_error = model.compute_l2_error(params, u_ref)
    print("L2 error: {:.3e}".format(l2_error))

    u_pred = model.u_pred_fn(params, model.r_star)

    
    # Convert them to NumPy arrays for Matplotlib
    r_star_np = jnp.array(r_star)
    u_pred_np = jnp.array(u_pred)
    u_ref_np = jnp.array(u_ref)

    # Create a Matplotlib figure and axis
    fig = plt.figure(figsize=(18, 5))
    plt.xlabel('radius [m]')
    plt.ylabel('Potential V(r)')

    # Plot the prediction values as a solid line
    plt.plot(r_star_np, u_pred_np, label='Prediction', color='blue')

    # Plot the analytical solution as a dashed line
    plt.plot(r_star_np, u_ref_np, linestyle='--', label='Analytical Solution', color='red')

    # Add a legend
    plt.legend()
    plt.tight_layout()

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    fig_path = os.path.join(save_dir, "laplace.pdf")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    """
    TT, XX = jnp.meshgrid(t_star, x_star, indexing="ij")

    # Plot results
    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(TT, XX, u_ref, cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Exact")
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.pcolor(TT, XX, u_pred, cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Predicted")
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.pcolor(TT, XX, jnp.abs(u_ref - u_pred), cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Absolute error")
    plt.tight_layout()

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    fig_path = os.path.join(save_dir, "adv.pdf")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)

    """