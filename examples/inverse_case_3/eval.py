import os

import ml_collections

import jax.numpy as jnp

import matplotlib.pyplot as plt
import train
from jaxpi.utils import restore_checkpoint
from jax import grad, vmap
import models
from utils import get_dataset
import jax 
import numpy as np


def evaluate(config: ml_collections.ConfigDict, workdir: str):
   
    # Problem Setup

    n_0 = config.setting.n_0
    n_inj = config.setting.n_inj
    u_0 = config.setting.u_0
    u_1 = config.setting.u_1
    n_t = 3   # Dummy, overwrite later
    n_x = 10_000   

    # Get  dataset
    u_ref, n_ref, t_star, x_star = get_dataset(n_t, n_x)
    t_star = jnp.linspace(0, 0.006, 7) # overwrite t b/c only need 7 values

    # Restore model
    model = models.InverseCoupledCase(config, n_inj, n_0, u_0, u_1, t_star, x_star)
    ckpt_path = os.path.join(workdir, "ckpt", config.wandb.name)
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    # Compute L2 error [Cannot do with small]
    #u_error, n_error = model.compute_l2_error(params, u_ref, n_ref)
    u_pred = model.u_pred_fn(params, model.t_star, model.x_star) # TODO: Ensure rescaled
    n_pred = model.n_pred_fn(params, model.t_star, model.x_star)
    
    #print("L2 error u: {:.3e}".format(u_error))
    #print("L2 error n: {:.3e}".format(n_error))

    print('Max predicted n:' , jnp.max(n_pred))
    print('Min predicted n:' , jnp.min(n_pred))

    print('Max predicted u:' , jnp.max(u_pred))
    print('Min predicted u:' , jnp.min(u_pred))
    
    # Plot results
    fig = plt.figure(figsize=(18, 12))
    plt.subplot(3, 1, 1)
    idx_step = int(n_t/10)
    plt.plot(x_star, n_pred[0,:], label='t=0.000')
    plt.plot(x_star, n_pred[1,:], label='t=0.001')
    plt.plot(x_star, n_pred[2,:], label='t=0.002')
    plt.plot(x_star, n_pred[3,:], label='t=0.003')
    plt.plot(x_star, n_pred[4,:], label='t=0.004')
    plt.plot(x_star, n_pred[5,:], label='t=0.005')
    plt.plot(x_star, n_pred[6,:], label='t=0.006')
    plt.grid()
    plt.xlabel("x [m]")
    plt.ylabel("Charge density n(x) [#/m3]")
    plt.title("Charge Density over x for different timesteps")
    plt.legend()
    plt.tight_layout()

    ####### ELECTRIC FIELD ######
    du_x = lambda params, t, x: -grad(model.u_net, argnums=2)(params, t, x)
    e_pred_fn = vmap(vmap(du_x, (None, None, 0)), (None, 0, None))
    e_pred = e_pred_fn(params, model.t_star, model.x_star)
    
    # plot Potential field
    plt.subplot(3, 1, 2)
    idx_step = int(n_t/10)
    plt.plot(x_star, u_pred[0,:], label='t=0.000')
    plt.plot(x_star, u_pred[1,:], label='t=0.001')
    plt.plot(x_star, u_pred[2,:], label='t=0.002')
    plt.plot(x_star, u_pred[3,:], label='t=0.003')
    plt.plot(x_star, u_pred[4,:], label='t=0.004')
    plt.plot(x_star, u_pred[5,:], label='t=0.005')
    plt.plot(x_star, u_pred[6,:], label='t=0.006')
    plt.xlabel("x [m]")
    plt.ylabel("Potential [V]")
    plt.title("Predicted Potentials")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.xlim(x_star[0], x_star[-1])

    # plot electrical field
    plt.subplot(3, 1, 3)
    idx_step = int(n_t/10)
    plt.plot(x_star, e_pred[0,:], label='t=0.000')
    plt.plot(x_star, e_pred[1,:], label='t=0.001')
    plt.plot(x_star, e_pred[2,:], label='t=0.002')
    plt.plot(x_star, e_pred[3,:], label='t=0.003')
    plt.plot(x_star, e_pred[4,:], label='t=0.004')
    plt.plot(x_star, e_pred[5,:], label='t=0.005')
    plt.plot(x_star, e_pred[6,:], label='t=0.006')
    plt.xlabel("x [m]")
    plt.ylabel("Electric field [V/m]")
    plt.title("Predicted Electrical field")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.xlim(x_star[0], x_star[-1])

    

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    fig_path = os.path.join(save_dir, "coupled_case.pdf")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
 

