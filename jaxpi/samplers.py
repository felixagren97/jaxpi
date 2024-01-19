from abc import ABC, abstractmethod
from functools import partial

import jax
import jax.numpy as jnp
from jax import random, pmap, local_device_count
from jax.tree_util import tree_map

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
        self.data = []
    
    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"
        batch = random.choice(key, self.r_eval, shape=(self.batch_size,), p=self.prob) 
        batch = batch.reshape(-1, 1)
        self.data.append(batch)
        return batch
    
    def get_data(self):
        return jnp.concatenate(self.data, axis=0)
    
    def plot_data_histogram(self, workdir, step, name):
        #plots the sampled data histogram
        fig = plt.figure(figsize=(8, 8))
        plt.xlabel('Radius [m]')
        plt.ylabel('Count')
        plt.title('Sampled data histogram')
        plt.hist(self.get_data(), bins=100, label='Sampled data', color='blue')
        plt.grid()
        plt.legend()
        plt.tight_layout()

        # Save the figure
        save_dir = os.path.join(workdir, "figures", name)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        fig_path = os.path.join(save_dir, f"rad_data_hist_{step}.png")
        fig.savefig(fig_path, bbox_inches="tight", dpi=800)

        plt.close(fig)
    
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
