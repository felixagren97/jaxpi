import os

import ml_collections

import jax.numpy as jnp

import matplotlib.pyplot as plt
import jax
from jaxpi.utils import restore_checkpoint
import numpy as np
import models
from utils import get_dataset


def evaluate(config: ml_collections.ConfigDict, workdir: str, step=''):
    # Get  dataset
    u_ref, t_star, x_star = get_dataset(n_t=8, n_x=10_000, config=config)

    # Restore model
    model = models.DriftDiffusion(config, t_star, x_star)
    ckpt_path = os.path.join(workdir, "ckpt", config.wandb.name)
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    # Compute L2 error
    l2_error = model.compute_l2_error(params, u_ref)
    print("L2 error: {:.3e}".format(l2_error))

    u_pred = model.u_pred_fn(params, model.t_star, model.x_star)
    
    print(f'Max overshoot:  {jnp.max(u_pred)}')
    print(f'Min undershoot: {jnp.min(u_pred)}')

    
    # Plot results
    fig = plt.figure()
    plt.plot(x_star, u_pred[0,:], label='t=0.000')
    plt.plot(x_star, u_pred[1,:], label='t=0.001')
    plt.plot(x_star, u_pred[2,:], label='t=0.002')
    plt.plot(x_star, u_pred[3,:], label='t=0.003')
    plt.plot(x_star, u_pred[4,:], label='t=0.004')
    plt.plot(x_star, u_pred[5,:], label='t=0.005')
    plt.plot(x_star, u_pred[6,:], label='t=0.006')
    plt.plot(x_star, u_ref[1,:], label='test analytical')
    plt.grid()
    plt.title('Charge density predictions')
    plt.xlabel("Distance [m]")
    plt.ylabel(r'Charge density [$\# / \mathrm{m}^3}$]')
    plt.legend()
    plt.tight_layout()

     # Plot results
    fig_2 = plt.figure()
    plt.plot(x_star, u_pred[0,:], color='blue', label='Model Prediction')
    plt.plot(x_star, u_pred[1,:], color='blue', label='_no_legend')
    plt.plot(x_star, u_pred[2,:], color='blue', label='_no_legend')
    plt.plot(x_star, u_pred[3,:], color='blue', label='_no_legend')
    plt.plot(x_star, u_pred[4,:], color='blue', label='_no_legend')
    plt.plot(x_star, u_pred[5,:], color='blue', label='_no_legend')
    plt.plot(x_star, u_pred[6,:], color='blue', label='_no_legend')
    plt.plot(x_star, u_ref[0,:], linestyle='dashed', color='red', label='Analytical Solution')
    plt.plot(x_star, u_ref[1,:], linestyle='dashed', color='red', label='_no_legend')
    plt.plot(x_star, u_ref[2,:], linestyle='dashed', color='red', label='_no_legend')
    plt.plot(x_star, u_ref[3,:], linestyle='dashed', color='red', label='_no_legend')
    plt.plot(x_star, u_ref[4,:], linestyle='dashed', color='red', label='_no_legend')
    plt.plot(x_star, u_ref[5,:], linestyle='dashed', color='red', label='_no_legend')
    plt.plot(x_star, u_ref[6,:], linestyle='dashed', color='red', label='_no_legend')
    plt.grid()
    plt.title('Predicted and Analytical Charge Density')
    plt.xlabel("Distance [m]")
    plt.ylabel(r'Charge density [$\# / \mathrm{m}^3}$]')

    plt.legend()
    plt.tight_layout()

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    fig_path = os.path.join(save_dir, f"drift_diffusion_{step}.png")
    fig.savefig(fig_path, bbox_inches="tight", dpi=800)

    fig_2_path = os.path.join(save_dir, "analytical.png")
    fig_2.savefig(fig_2_path, bbox_inches="tight", dpi=800)

    if step == '':
        # save plot information as csv for later use        
        TT, XX = jnp.meshgrid(t_star, x_star, indexing='ij')

        u_pred = jax.device_get(u_pred)

        TT = jax.device_get(TT)
        XX = jax.device_get(XX)

        u_pred = u_pred.reshape(-1)
        u_ref = u_ref.reshape(-1)
        TT = TT.reshape(-1)
        XX = XX.reshape(-1)
        combined_array = np.column_stack((TT, XX, u_pred, u_ref))
        csv_file_path = "Drift Diffusion.csv"
        header_names = ['t_star', 'x_star', 'n_pred', 'n_ref']
        np.savetxt(csv_file_path, combined_array, delimiter=",", header=",".join(header_names), comments='')