import os

import ml_collections

import jax.numpy as jnp
import numpy as np
import jax 
import pandas as pd
import matplotlib.pyplot as plt
import train
from jaxpi.utils import restore_checkpoint
from jax import grad, vmap
import models
from utils import get_dataset, get_reference_dataset


def evaluate(u_config: ml_collections.ConfigDict, n_config: ml_collections.ConfigDict, workdir: str, step=''):

    # Get  dataset
    n_t = 200
    n_x = 10_000
    _, _, _, x_star = get_dataset(n_t, n_x)
    t_star = jnp.linspace(0, 0.006, 7) # overwrite t b/c only need 7 values


    # Restore u_model
    u_model = models.UModel(u_config, t_star, x_star, None)
    ckpt_path = os.path.join(workdir, "ckpt", u_config.wandb.name, u_model.tag)
    u_model.state = restore_checkpoint(u_model.state, ckpt_path)
    u_params = u_model.state.params
    u_pred = u_model.u_pred_fn(u_params, t_star, x_star) 

    # restore n_model 
    n_model = models.NModel(n_config, t_star, x_star, u_model)
    ckpt_path = os.path.join(workdir, "ckpt", n_config.wandb.name, n_model.tag)
    n_model.state = restore_checkpoint(n_model.state, ckpt_path)
    n_params = n_model.state.params
    n_pred = n_model.n_pred_fn(n_params, t_star, x_star) 

    print('Max predicted n:' , jnp.max(n_pred))
    print('Min predicted n:' , jnp.min(n_pred))

    print('Max predicted u:' , jnp.max(u_pred))
    print('Min predicted u:' , jnp.min(u_pred))
    
    # Plot results
    fig = plt.figure(figsize=(7, 12))
    plt.subplot(3, 1, 1)
    plt.plot(x_star, n_pred[0,:], label='t=0.000')
    plt.plot(x_star, n_pred[1,:], label='t=0.001')
    plt.plot(x_star, n_pred[2,:], label='t=0.002')
    plt.plot(x_star, n_pred[3,:], label='t=0.003')
    plt.plot(x_star, n_pred[4,:], label='t=0.004')
    plt.plot(x_star, n_pred[5,:], label='t=0.005')
    plt.plot(x_star, n_pred[6,:], label='t=0.006')
    plt.grid()
    plt.xlabel("Distance [m]", fontsize=14)
    plt.ylabel(r'Charge density [$\# / \mathrm{m}^3}$]', fontsize=14)
    plt.title("Predicted charge density", fontsize=14)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.xlim(x_star[0], x_star[-1])

    ####### ELECTRIC FIELD ######
    du_x = lambda params, t, x: -grad(u_model.u_net, argnums=2)(params, t, x)
    e_pred_fn = vmap(vmap(du_x, (None, None, 0)), (None, 0, None))
    e_pred = e_pred_fn(u_params, t_star, x_star)
    
    # plot Potential field
    plt.subplot(3, 1, 2)
    plt.plot(x_star, u_pred[0,:], label='t=0.000')
    plt.plot(x_star, u_pred[1,:], label='t=0.001')
    plt.plot(x_star, u_pred[2,:], label='t=0.002')
    plt.plot(x_star, u_pred[3,:], label='t=0.003')
    plt.plot(x_star, u_pred[4,:], label='t=0.004')
    plt.plot(x_star, u_pred[5,:], label='t=0.005')
    plt.plot(x_star, u_pred[6,:], label='t=0.006')
    plt.plot([x_star[0], x_star[-1]], [u_config.setting.u_0, u_config.setting.u_1], linestyle='--', color='black')
    plt.xlabel("Distance [m]", fontsize=14)
    plt.ylabel("Potential [V]", fontsize=14)
    plt.title("Predicted potential", fontsize=14)
    plt.grid()
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.xlim(x_star[0], x_star[-1])

    # plot electrical field
    plt.subplot(3, 1, 3)
    plt.plot(x_star, e_pred[0,:], label='t=0.000')
    plt.plot(x_star, e_pred[1,:], label='t=0.001')
    plt.plot(x_star, e_pred[2,:], label='t=0.002')
    plt.plot(x_star, e_pred[3,:], label='t=0.003')
    plt.plot(x_star, e_pred[4,:], label='t=0.004')
    plt.plot(x_star, e_pred[5,:], label='t=0.005')
    plt.plot(x_star, e_pred[6,:], label='t=0.006')
    plt.xlabel("Distance [m]", fontsize=14)
    plt.ylabel("Electric field [V/m]", fontsize=14)
    plt.title("Predicted electric field", fontsize=14)
    plt.grid()
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.xlim(x_star[0], x_star[-1])

    # Save the figure
    save_dir = os.path.join(workdir, "figures", u_config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    fig_path = os.path.join(save_dir, f"seq_coupled_case_{step}.png")
    fig.savefig(fig_path, bbox_inches="tight", dpi=800)
    plt.close(fig)

    # save image data to csv
    if step == u_config.training.max_steps:

        df = pd.DataFrame({'x_values': x_star})
        for i, t in enumerate(t_star):
            df[f'e_pred_{t}'] = e_pred[i, :]
        df.to_csv('field_data.csv', index=False)

        df = pd.DataFrame({'x_values': x_star})
        for i, t in enumerate(t_star):
            df[f'u_pred_{t}'] = u_pred[i, :]
        df.to_csv('potential_data.csv', index=False)
        
        df = pd.DataFrame({'x_values': x_star})
        for i, t in enumerate(t_star):
            df[f'n_pred_{t}'] = n_pred[i, :]
        df.to_csv('charge_density_data.csv', index=False)

    # Save COMSOL comparison
    file_paths = [u_config.eval.potential_file_path, u_config.eval.field_file_path, u_config.eval.ion_density_file_path]
    has_ref_data = all(path is not None for path in file_paths)
    has_ref_inj = u_config.setting.n_inj in [5e9, 5e13, 1e14, 5e15]
    if has_ref_data and has_ref_inj:
        t_ref_star, x_ref_star, u_ref = get_reference_dataset(u_config, u_config.eval.potential_file_path)
        _, _, e_ref = get_reference_dataset(u_config, u_config.eval.field_file_path)
        _, _, n_ref = get_reference_dataset(u_config, u_config.eval.ion_density_file_path)

        # get new pred data
        u_ref_pred = u_model.u_pred_fn(u_params, t_ref_star, x_ref_star)
        n_ref_pred = n_model.n_pred_fn(n_params, t_ref_star, x_ref_star)
        e_ref_pred = e_pred_fn(u_params, t_ref_star, x_ref_star)
        
        # Plot n results
        fig = plt.figure(figsize=(8, 12))
        plt.subplot(3, 1, 1)
        for i, t in enumerate(t_star): 
            plt.plot(x_ref_star, n_ref_pred[i,:], label='PINN' if i == 0 else '', color='blue')
            plt.plot(x_ref_star, n_ref[i,:], label='COMSOL' if i == 0 else '', color='red')
        plt.grid()
        plt.xlabel("Distance [m]", fontsize=14)
        plt.ylabel(r'Charge density [$\# / \mathrm{m}^3}$]', fontsize=14)
        plt.title("Charge density predictions using PINN and COMSOL", fontsize=14)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.xlim(x_star[0], x_star[-1])

        # plot Potential field
        plt.subplot(3, 1, 2)
        for i, t in enumerate(t_star): 
            plt.plot(x_ref_star, u_ref_pred[i,:], label='PINN' if i == 0 else '', color='blue')
            plt.plot(x_ref_star, u_ref[i,:], label='COMSOL' if i == 0 else '', color='red', linestyle='--')
        plt.xlabel("Distance [m]", fontsize=14)
        plt.ylabel("Potential [V]", fontsize=14)
        plt.title("Potential predictions using PINN and COMSOL", fontsize=14)
        plt.grid()
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.xlim(x_star[0], x_star[-1])

        # plot electrical field
        plt.subplot(3, 1, 3)
        for i, t in enumerate(t_star): 
            plt.plot(x_ref_star, e_ref_pred[i,:], label='PINN' if i == 0 else '', color='blue')
            plt.plot(x_ref_star, e_ref[i,:], label='COMSOL' if i == 0 else '', color='red', linestyle='--')
        plt.xlabel("Distance [m]", fontsize=14)
        plt.ylabel("Electric field [V/m]", fontsize=14)
        plt.title("Electric field predictions using PINN and COMSOL", fontsize=14)
        plt.grid()
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.xlim(x_star[0], x_star[-1])

        # save image
        fig_path = os.path.join(save_dir, f"comp_seq_coupled_case_{step}.png")
        fig.savefig(fig_path, bbox_inches="tight", dpi=800)
        plt.close(fig)

        # save image data to csv
        if step == u_config.training.max_steps:

            df = pd.DataFrame({'x_values': x_ref_star})
            for i, t in enumerate(t_star):
                df[f'e_ref_pred_{t}'] = e_ref_pred[i, :]
                df[f'e_ref_{t}'] = e_ref[i, :]
            df.to_csv('field_comparison_data.csv', index=False)

            df = pd.DataFrame({'x_values': x_ref_star})
            for i, t in enumerate(t_star):
                df[f'u_ref_pred_{t}'] = u_ref_pred[i, :]
                df[f'u_ref_{t}'] = u_ref[i, :]
            df.to_csv('potential_comparison_data.csv', index=False)
           
            df = pd.DataFrame({'x_values': x_ref_star})
            for i, t in enumerate(t_star):
                df[f'n_ref_pred_{t}'] = n_ref_pred[i, :]
                df[f'n_ref_{t}'] = n_ref[i, :]
            df.to_csv('charge_density_comparison_data.csv', index=False)

    # Save observations
    if step == "":
        n_t = 250
        n_x = 250
        _, _, t_star, x_star = get_dataset(n_t, n_x)

        u_pred = u_model.u_pred_fn(u_params, t_star, x_star)
        
        TT, XX = jnp.meshgrid(t_star, x_star, indexing='ij')

        u_pred = jax.device_get(u_pred)

        TT = jax.device_get(TT)
        XX = jax.device_get(XX)

        u_pred = u_pred.reshape(-1)
        TT = TT.reshape(-1)
        XX = XX.reshape(-1)
        data = np.column_stack((TT, XX, u_pred))

        output_file_path = 'case3_obs.dat'
        np.savetxt(output_file_path, data, delimiter=' ') 

