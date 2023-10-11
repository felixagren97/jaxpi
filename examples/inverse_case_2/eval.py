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
    n_0 = 0.1/1e9
    n_inj = 1
    n_t = 200  # number of time steps TODO: Increase?
    n_x = 10_000  # number of spatial points

    true_mu = config.setting.true_mu

    # Get  dataset
    u_ref, t_star, x_star, u_exact_fn = get_dataset(n_t, n_x, true_mu, n_inj, n_0)
    t_star = jnp.linspace(0, 0.006, 7)

    # Restore model
    model = models.InverseDriftDiffusion(config, n_inj, n_0, E_ext, t_star, x_star, u_exact_fn)
    ckpt_path = os.path.join(workdir, "ckpt", config.wandb.name)
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    # Compute L2 error
    print('Max predicted n:' , jnp.max(u_pred))
    print('Min predicted n:' , jnp.min(u_pred))

    u_pred = model.u_pred_fn(params, model.t_star, model.x_star)
    TT, XX = jnp.meshgrid(t_star, x_star, indexing="ij")
    
    print('jnp.max(u_pred):' , jnp.max(u_pred))
    print('shape u_pred:', u_pred.shape)
    print('shape x_star:', x_star.shape)
    print('shape t_star:', t_star.shape)
    
    # Plot results
    fig = plt.figure(figsize=(5, 5))
    plt.plot(x_star, u_pred[0,:], label='t=0.000')
    plt.plot(x_star, u_pred[1,:], label='t=0.001')
    plt.plot(x_star, u_pred[2,:], label='t=0.002')
    plt.plot(x_star, u_pred[3,:], label='t=0.003')
    plt.plot(x_star, u_pred[4,:], label='t=0.004')
    plt.plot(x_star, u_pred[5,:], label='t=0.005')
    plt.plot(x_star, u_pred[6,:], label='t=0.006')
    plt.grid()
    plt.xlabel("x [m]")
    plt.ylabel("Charge density n(x) [#/m3]")
    plt.title("Charge Density over x for different timesteps")
    plt.legend()
    plt.tight_layout()


    #plt.subplot(1, 3, 3)
    #plt.pcolor(TT, XX, jnp.abs(u_ref - u_pred), cmap="jet")
    #plt.colorbar()
    #plt.xlabel("t")
    #plt.ylabel("x")
    #plt.title("Absolute error")
    #plt.tight_layout()

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    fig_path = os.path.join(save_dir, "Inverse_drift_diffusion.pdf")
    fig.savefig(fig_path, bbox_inches="tight", dpi=800)
    fig_path = os.path.join(save_dir, "Inverse_drift_diffusion.png")
    fig.savefig(fig_path, bbox_inches="tight", dpi=800)
    
    # --- final result prints ---
    print('\n--------- SUMMARY ---------\n')
    # print L2 error
    l2_error = model.compute_l2_error(params, u_ref)
    print("L2 error:       {:.3e}".format(l2_error))  

    # print the predicted & final rho values
    mu_pred = jnp.exp(model.state.params['params']['mu_param'][0]) 
    mu_ref = config.setting.true_mu
    rel_error = (mu_pred-mu_ref)/mu_ref
    #pred_scale = abs(floor(log(rho_pred, 10)))
    #rho_pred = round(rho_pred, pred_scale + 3)
    print(f'Predicted Rho:  {mu_pred}')
    print(f'True Rho:       {mu_ref}')
    print(f'Relative error: {rel_error:.1%}\n')
    print('---------------------------\n')

