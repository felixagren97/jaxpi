import os

import ml_collections

import jax.numpy as jnp

import matplotlib.pyplot as plt

from jaxpi.utils import restore_checkpoint
import numpy as np
import models
from utils import get_dataset


def evaluate(config: ml_collections.ConfigDict, workdir: str, step=''):
    # Get  dataset
    u_ref, t_star, x_star = get_dataset(n_t=7, n_x=10_000, config=config)

    # Restore model
    model = models.DriftDiffusion(config, t_star, x_star)
    ckpt_path = os.path.join(workdir, "ckpt", config.wandb.name)
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    # Compute L2 error
    l2_error = model.compute_l2_error(params, u_ref)
    print("L2 error: {:.3e}".format(l2_error))

    u_pred = model.u_pred_fn(params, model.t_star, model.x_star)
    
    print(f'Max overshoot: {jnp.max(u_pred)}')
    print(f'Min overshoot: {jnp.max(u_pred)}')

    
    # Plot results
    fig = plt.figure()
    plt.plot(x_star, u_pred[0,:], label='t=0.000')
    plt.plot(x_star, u_pred[1,:], label='t=0.001')
    plt.plot(x_star, u_pred[2,:], label='t=0.002')
    plt.plot(x_star, u_pred[3,:], label='t=0.003')
    plt.plot(x_star, u_pred[4,:], label='t=0.004')
    plt.plot(x_star, u_pred[5,:], label='t=0.005')
    plt.plot(x_star, u_pred[6,:], label='t=0.006')
    plt.grid()
    plt.xlabel("Distance [m]")
    plt.ylabel("Charge density [#/m3]")
    plt.legend()
    plt.tight_layout()

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    fig_path = os.path.join(save_dir, f"drift_diffusion_{step}.png")
    fig.savefig(fig_path, bbox_inches="tight", dpi=800)

    if step == "":
        # save plot information as csv for later use
        combined_array = np.column_stack((t_star, x_star, u_pred, u_ref))
        csv_file_path = "Drift Diffusion.csv"
        header_names = ['t_star', 'x_star', 'u_pred', 'u_ref']
        np.savetxt(csv_file_path, combined_array, delimiter=",", header=",".join(header_names), comments='')
