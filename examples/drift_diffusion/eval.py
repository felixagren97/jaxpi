import os

import ml_collections

import jax.numpy as jnp

import matplotlib.pyplot as plt

from jaxpi.utils import restore_checkpoint

import models
from utils import get_dataset


def evaluate(config: ml_collections.ConfigDict, workdir: str):
   # Problem setup
    E_ext = 1e6
    n_0 = 0.1
    n_inj = 1
    n_t = 200  # number of time steps TODO: Increase?
    n_x = 128  # number of spatial points

    # Get  dataset
    u_ref, t_star, x_star = get_dataset(n_t, n_x)


    # Restore model
    model = models.DriftDiffusion(config, n_inj, n_0, E_ext, t_star, x_star)
    ckpt_path = os.path.join(workdir, "ckpt", config.wandb.name)
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    # Compute L2 error
    l2_error = model.compute_l2_error(params, u_ref)
    print("L2 error: {:.3e}".format(l2_error))

    u_pred = model.u_pred_fn(params, model.t_star, model.x_star)
    TT, XX = jnp.meshgrid(t_star, x_star, indexing="ij")
    
    print('shape u_pred:', u_pred.shape)
    print('shape x_star:', x_star.shape)
    print('shape t_star:', t_star.shape)
    
    # Plot results
    # TODO: Change exact plot to match Arni's
    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(u_pred, x_star, cmap="jet")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("n")
    plt.title("Arni")
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

    fig_path = os.path.join(save_dir, "drift_diffusion.pdf")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
