import os

import ml_collections

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

from jaxpi.utils import restore_checkpoint
import models
from utils import get_dataset


def evaluate(config: ml_collections.ConfigDict, workdir: str, step=""):
    
     # Problem setup
    n_x = config.setting.n_x    # number of spatial points (old: 128 TODO: INCREASE A LOT?)
    n_scale = config.setting.n_scale

    # Get  dataset
    u_ref, x_star = get_dataset(n_x=n_x)

    # Initial condition (TODO: Looks as though this is for t = 0 in their solution, should we have for x = 0)?
    u0 = config.setting.u0
    u1 = config.setting.u1

    # Define domain
    x0 = x_star[0]
    x1 = x_star[-1]

    dom = jnp.array([x0, x1]) 
    # Restore model
    model = models.InversePoisson(config, u0, u1, x_star, n_scale)
    ckpt_path = os.path.join(workdir, "ckpt", config.wandb.name)
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    # Compute L2 error
    l2_error = model.compute_l2_error(params, u_ref)
    print("L2 error: {:.3e}".format(l2_error))

    u_pred = model.u_pred_fn(params, model.x_star)
    u_pred *= u0
    e_pred_fn = jax.vmap(lambda params, x: -jax.grad(model.u_net, argnums=1)(params, x), (None, 0))

    n_pred = model.n_pred_fn(params, model.x_star)
    n_pred *= n_scale   # TODO: check if correct


    #du_dr = jax.grad(model.u_pred_fn) # e = d/dr U
    e_pred = e_pred_fn(params, model.x_star)
    e_pred *= u0
    
    n_values = n_scale * jax.vmap(model.heaviside)(x_star)

    r_pred = model.r_pred_fn(params, model.x_star)**2


    # Create a Matplotlib figure and axis
    fig = plt.figure(figsize=(18, 14))
    plt.subplot(4,1,1)
    plt.xlabel('Distance [m]')
    plt.ylabel('Charge density n(x)')
    plt.title('Charge density')
    plt.plot(x_star, n_pred, label='pred (x)', color='blue')
    plt.plot(x_star, n_values, linestyle='--', label='true n(x)', color='red')
    plt.legend()
    plt.tight_layout()    
    plt.xlim(x_star[0], x_star[-1])
    plt.grid()


    plt.subplot(4, 1, 2)
    plt.xlabel('Distance [m]')
    plt.ylabel('Potential V(x)')
    plt.title('Potential')

    # Plot the prediction
    plt.plot(x_star, u_pred, label='Predicted V(x)', color='blue')

    # Plot original V(x)
    plt.plot(x_star, 1e6*(-x_star + 1), linestyle='--', label='Original V(x)', color='red') 
    plt.grid()
    plt.legend()
    plt.tight_layout()    
    plt.xlim(x_star[0], x_star[-1])

    # plot electrical field
    plt.subplot(4, 1, 3)

    plt.xlabel('Distance [m]')
    plt.ylabel('Electric field [V/m]')
    plt.title('Electrical field')

    # Plot the prediction values as a solid line
    plt.plot(x_star, e_pred, color='blue')
    plt.grid()
    plt.xlim(x_star[0], x_star[-1])
    plt.tight_layout()    

    plt.subplot(4, 1, 4)
    plt.scatter(x_star, r_pred, color='blue', marker='o', s=1, alpha=0.5)  # Use marker='o' for circular markers, adjust 's' for marker size
    plt.yscale('log')
    plt.plot(x_star, jnp.full_like(x_star, jnp.mean(r_pred)), label='Mean', linestyle='--', color='red')

    plt.xlabel('Distance [m]')
    plt.ylabel('Squared Residual Loss')
    plt.title('Squared Residual Loss')
    plt.legend()
    plt.grid()
    plt.xlim(x_star[0], x_star[-1])
    plt.tight_layout()

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    fig_path = os.path.join(save_dir, f"laplace_2.5_{step}.pdf")
    fig.savefig(fig_path, bbox_inches="tight", dpi=800)