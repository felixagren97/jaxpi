import os

import ml_collections

import jax.numpy as jnp

import matplotlib.pyplot as plt

from jaxpi.utils import restore_checkpoint

import models
from utils import get_dataset


def evaluate(config: ml_collections.ConfigDict, workdir: str):
   
    # Problem setup
    n_0 = 0.1
    n_inj = 1
    u_0 = 1e6
    u_1 = 0
    n_t = 200  # number of time steps TODO: Increase?
    n_x = 128  # number of spatial points


    # Get  dataset
    u_ref, n_ref, t_star, x_star = get_dataset(n_t, n_x)

    # Restore model
    model = models.CoupledCase(config, n_inj, n_0, u_0, u_1, t_star, x_star)
    ckpt_path = os.path.join(workdir, "ckpt", config.wandb.name)
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    # Compute L2 error
    u_error, n_error = model.compute_l2_error(params, u_ref, n_ref)
    print("L2 error u: {:.3e}".format(u_error))
    print("L2 error n: {:.3e}".format(n_error))

    u_pred = model.u_pred_fn(params, model.t_star, model.x_star)
    n_pred = model.n_pred_fn(params, model.t_star, model.x_star)
    TT, XX = jnp.meshgrid(t_star, x_star, indexing="ij")
    
    print('Max predicted n:' , jnp.max(n_pred))
    
    # Plot results
    #fig = plt.figure(figsize=(18, 5))
    #plt.subplot(1, 2, 1)
    #idx_step = int(n_t/10)
    #plt.plot(x_star, u_pred[idx_step * 0, :], label='t=0.000')
    #plt.plot(x_star, u_pred[idx_step * 1, :], label='t=0.001')
    #plt.plot(x_star, u_pred[idx_step * 2, :], label='t=0.002')
    #plt.plot(x_star, u_pred[idx_step * 3, :], label='t=0.003')
    #plt.plot(x_star, u_pred[idx_step * 4, :], label='t=0.004')
    #plt.plot(x_star, u_pred[idx_step * 5, :], label='t=0.005')
    #plt.plot(x_star, u_pred[idx_step * 6, :], label='t=0.006')
    #plt.grid()
    #plt.xlabel("x")
    #plt.ylabel("n")
    #plt.title("Charge Density over x for different timesteps")
    #plt.legend()
    #plt.tight_layout()
#
    #plt.subplot(1, 2, 2)
    #plt.pcolor(TT, XX, u_pred, cmap="jet")
    #plt.colorbar()
    #plt.xlabel("t")
    #plt.ylabel("x")
    #plt.title("Predicted")
    #plt.tight_layout()

    #plt.subplot(1, 3, 3)
    #plt.pcolor(TT, XX, jnp.abs(u_ref - u_pred), cmap="jet")
    #plt.colorbar()
    #plt.xlabel("t")
    #plt.ylabel("x")
    #plt.title("Absolute error")
    #plt.tight_layout()

    # Save the figure
    #save_dir = os.path.join(workdir, "figures", config.wandb.name)
    #if not os.path.isdir(save_dir):
    #    os.makedirs(save_dir)
#
    #fig_path = os.path.join(save_dir, "drift_diffusion.pdf")
    #fig.savefig(fig_path, bbox_inches="tight", dpi=300)

