from abc import ABC, abstractmethod
from functools import partial

import jax
from jax import lax, jit, grad, vmap
import jax.numpy as jnp
from jax import random, pmap, local_device_count
from jax.tree_util import tree_map
import numpy as np

import matplotlib.pyplot as plt
import os

from torch.utils.data import Dataset


# Function for initializing sampler from config file
# argument: model reference, sampler name, and specific kwargs from config file 
def init_sampler(model, config):
    sampler = config.sampler.sampler_name
    batch_size = config.training.batch_size_per_device

    if sampler == "rad":
        return OneDimensionalRadSampler(model, batch_size, config)
    elif sampler == "rad2":
        return OneDimensionalRadSamplerTwo(model, batch_size, config)
    elif sampler == "rad-cosine":
        return RadCosineAnnealing(model, batch_size, config)
    elif sampler == "adaptive-g":
        return GradientSampler(model, batch_size, config)
    else:     
        raise NotImplementedError(f"Sampler {sampler} not implemented!")


class BaseSampler(Dataset):
    def __init__(self, batch_size, rng_key=random.PRNGKey(1234)):
        self.batch_size = batch_size
        self.key = rng_key
        self.num_devices = local_device_count()

    def __getitem__(self, index):
        "Generate one batch of data"
        self.key, subkey = random.split(self.key)
        keys = random.split(subkey, self.num_devices)
        batch = self.data_generation(keys)
        return batch

    def data_generation(self, key):
        raise NotImplementedError("Subclasses should implement this!")


class UniformSampler(BaseSampler):
    def __init__(self, dom, batch_size, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)
        self.dom = dom
        self.dim = dom.shape[0]

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"
        batch = random.uniform(
            key,
            shape=(self.batch_size, self.dim),
            minval=self.dom[:, 0],
            maxval=self.dom[:, 1],
        )
        return batch
 
class OneDimensionalRadSampler(BaseSampler):
    def __init__(self, model, batch_size, config, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)
        self.dim = 1
        self.r_eval = jnp.linspace(config.setting.r_0, config.setting.r_1, 100_000) # 100k used in paper
        self.state = jax.device_get(tree_map(lambda x: x[0], model.state))
        res_pred = jnp.abs(model.r_pred_fn(self.state.params, self.r_eval)) # Verify shape on r_eval
        self.prob = res_pred / jnp.sum(res_pred)
        
    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"
        batch = random.choice(key, self.r_eval, shape=(self.batch_size,), p=self.prob) 
        batch = batch.reshape(-1, 1)
        return batch
    
    def plot(self, workdir, step, name):
        fig = plt.figure(figsize=(8, 8))
        plt.xlabel('Radius [m]')
        plt.ylabel('norm_r_eval')
        plt.title('Residual distribution')
        plt.plot(self.r_eval, self.prob, label='Norm. Residual', color='blue')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        
        # Save the figure
        save_dir = os.path.join(workdir, "figures", name)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        fig_path = os.path.join(save_dir, f"rad_prob_{step}.png")
        fig.savefig(fig_path, bbox_inches="tight", dpi=800)

        plt.close(fig)

class OneDimensionalRadSamplerTwo(BaseSampler):
    # Imporved RAD
    def __init__(self, model, batch_size, config, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)
        self.dim = 1
        self.r_eval = jnp.linspace(config.setting.r_0, config.setting.r_1, 100_000) # 100k used in paper
        self.c = config.sampler.c 
        self.k = config.sampler.k
        
        self.state = jax.device_get(tree_map(lambda x: x[0], model.state))
        res_pred = jnp.abs(model.r_pred_fn(self.state.params, self.r_eval)) # Verify shape on r_eval
    
        prob = jnp.power(res_pred, self.k) / jnp.power(res_pred, self.k).mean() + self.c
        self.norm_prob = prob / prob.sum()

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"
        
        batch = random.choice(key, self.r_eval, shape=(self.batch_size,), p=self.norm_prob) 
        batch = batch.reshape(-1, 1)
        return batch
    
    def plot(self, workdir, step, name):
        fig = plt.figure(figsize=(8, 8))
        plt.xlabel('Radius [m]')
        plt.ylabel('norm_r_eval')
        plt.title('Residual distribution')
        plt.plot(self.r_eval, self.norm_prob, label='Norm. Residual', color='blue')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        
        # Save the figure
        save_dir = os.path.join(workdir, "figures", name)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        fig_path = os.path.join(save_dir, f"rad2_prob_{step}.png")
        fig.savefig(fig_path, bbox_inches="tight", dpi=800)

        plt.close(fig)


class RadCosineAnnealing(BaseSampler):
    def __init__(self, model, batch_size, config, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)
        jax.debug.print("Init cosine annealing")

        self.dim = 1
        self.r_eval = jnp.linspace(config.setting.r_0, config.setting.r_1, 100_000) # 100k used in paper
        self.c = config.sampler.c 
        self.k = config.sampler.k
        
        self.T = 10  # -> should be a annealing period of 100k iterations
        self.T_c = 0 # Should be increased every 10k iterations
        
        self.lr = 0.9 # TODO: Make this a config parameter

        self.state = jax.device_get(tree_map(lambda x: x[0], model.state))
        res_pred = jnp.abs(model.r_pred_fn(self.state.params, self.r_eval)) # Verify shape on r_eval   
        prob_res = jnp.power(res_pred, self.k) / jnp.power(res_pred, self.k).mean() + self.c
        self.norm_prob_res = prob_res / prob_res.sum()
        self.norm_prob_uni = jnp.ones_like(self.norm_prob_res) / len(self.norm_prob_res)

        self.current_prob = self.norm_prob_uni

        self.n = self.cosine_annealing(self.T, self.T_c) #Portion of uniform distribution to be added to current distribution
        self.num_uniform = (jnp.floor(self.n * self.batch_size) - 1).astype(int).item()
        self.num_res = (self.batch_size - self.num_uniform).astype(int).item()
        jax.debug.print("self.n: {x}", x=self.n)
        jax.debug.print("self.num_res: {x}", x=self.num_res)
        jax.debug.print("self.num_uniform: {x}", x=self.num_uniform)
        jax.debug.print("shape r_eval: {x}", x=self.r_eval.shape)
        jax.debug.print("shape current prob: {x}", x=self.current_prob.shape)



    def cosine_annealing(self, T, T_c):
            return 0.5 * (1 + jnp.cos(jnp.pi * T_c / T))

    def update_prob(self, model):
        jax.debug.print("In update_prob fn")

        # Update the current probability distribution every 10k iteration
        res_pred = jnp.abs(model.r_pred_fn(self.state.params, self.r_eval))
        self.norm_prob_res = jnp.power(res_pred, self.k) / jnp.power(res_pred, self.k).mean() + self.c
        self.current_prob += self.lr * self.norm_prob_res
        self.current_prob /= self.current_prob.sum()

        self.T_c = (self.T_c + 1) % self.T    #TODO: Make sure this function is called every 10k iteration 
        self.n = self.cosine_annealing(self.T, self.T_c)
        self.num_res = jnp.floor(self.n * self.batch_size) - 1
        self.num_uniform = self.batch_size - self.num_res
        jax.debug.print("New self.n: {x}", x=self.n)
        jax.debug.print("New self.num_res: {x}", x=self.num_res)
        jax.debug.print("New self.num_uniform: {x}", x=self.num_uniform)

        
    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"
        
        uni_batch = random.uniform(key, shape=(self.num_uniform, ), minval=self.r_eval[0], maxval=self.r_eval[-1])
        res_batch = random.choice(key, self.r_eval, shape=(self.num_res, ), p=self.current_prob) 
        
        batch = jnp.concatenate([res_batch, uni_batch], axis=0)

        batch = batch.reshape(-1, 1)
        return batch
    

    def plot(self, workdir, step, name):
        fig = plt.figure(figsize=(8, 8))
        plt.xlabel('Radius [m]')
        plt.ylabel('norm_r_eval')
        plt.title('Residual distribution')
        plt.plot(self.r_eval, self.current_prob, label='Current prob.', color='blue')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        
        # Save the figure
        save_dir = os.path.join(workdir, "figures", name)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        fig_path = os.path.join(save_dir, f"cosine_prob_{step}.png")
        fig.savefig(fig_path, bbox_inches="tight", dpi=800)

        plt.close(fig)

class GradientSampler(BaseSampler):
    def __init__(self, model, batch_size, config, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)
        self.dim = 1
        self.r_eval = jnp.linspace(config.setting.r_0, config.setting.r_1, 100_000) # 100k used in paper
        self.gamma = config.sampler.gamma 
        self.batch_size = batch_size
        
        self.state = jax.device_get(tree_map(lambda x: x[0], model.state))
        
        #l_grad_fn = jax.vmap(lambda params, r: jax.grad(model.r_net, argnums=1)(params, r), (None, 0))
        #dl_r = jnp.abs(l_grad_fn(self.state.params, self.r_eval))
        dl_r = jnp.abs(self.batched_gradient_computation(model, self.state.params))
        self.norm_prob =  dl_r / dl_r.sum()

    def batched_gradient_computation(self, model, params, grad_batch_size=8192):
        num_batches = len(self.r_eval) // self.batch_size + (len(self.r_eval) % grad_batch_size != 0)
        all_grads = []
        for i in range(num_batches):
            batch_r_eval = self.r_eval[i * grad_batch_size:(i + 1) * grad_batch_size]
            batch_grads = jax.vmap(lambda r: jax.grad(model.r_net, argnums=1)(params, r))(batch_r_eval)
            all_grads.append(batch_grads)
        return jnp.concatenate(all_grads, axis=0)
    
    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"
        print("data_generation")
        batch = random.choice(key, self.r_eval, shape=(self.batch_size,), p=self.norm_prob) 
        batch = batch.reshape(-1, 1)
        return batch
    
    def plot(self, workdir, step, name):
        fig = plt.figure(figsize=(8, 8))
        plt.xlabel('Radius [m]')
        plt.ylabel('norm_r_eval')
        plt.title('Gradient distribution')
        plt.plot(self.r_eval, self.norm_prob, label='Norm. Gradient', color='blue')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        
        # Save the figure
        save_dir = os.path.join(workdir, "figures", name)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        fig_path = os.path.join(save_dir, f"grad_prob_{step}.png")
        fig.savefig(fig_path, bbox_inches="tight", dpi=800)

        plt.close(fig)


class SpaceSampler(BaseSampler):
    def __init__(self, coords, batch_size, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)
        self.coords = coords

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"
        idx = random.choice(key, self.coords.shape[0], shape=(self.batch_size,))
        batch = self.coords[idx, :]

        return batch


class TimeSpaceSampler(BaseSampler):
    def __init__(
        self, temporal_dom, spatial_coords, batch_size, rng_key=random.PRNGKey(1234)
    ):
        super().__init__(batch_size, rng_key)

        self.temporal_dom = temporal_dom
        self.spatial_coords = spatial_coords

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"
        key1, key2 = random.split(key)

        temporal_batch = random.uniform(
            key1,
            shape=(self.batch_size, 1),
            minval=self.temporal_dom[0],
            maxval=self.temporal_dom[1],
        )

        spatial_idx = random.choice(
            key2, self.spatial_coords.shape[0], shape=(self.batch_size,)
        )
        spatial_batch = self.spatial_coords[spatial_idx, :]
        batch = jnp.concatenate([temporal_batch, spatial_batch], axis=1)

        return batch
