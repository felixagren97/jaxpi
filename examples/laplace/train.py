import os
import time

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

import ml_collections

# from absl import logging
import wandb

from jaxpi.samplers import BaseSampler, OneDimensionalRadSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint

import models
from utils import get_dataset

from abc import ABC, abstractmethod
from functools import partial

import jax.numpy as jnp
from jax import random, pmap, local_device_count
from eval import evaluate
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class OneDimensionalUniformSampler(BaseSampler):
    def __init__(self, dom, batch_size, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)
        self.dom = dom
        self.dim = 1

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"
        batch = random.uniform(
            key,
            shape=(self.batch_size, self.dim),
            minval=self.dom[0],
            maxval=self.dom[1],
        )

        return batch


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    logger = Logger()
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    # Problem setup
    r_0 = config.setting.r_0      # inner radius
    r_1 = config.setting.r_1      # outer radius
    n_r = config.setting.n_r    # number of spatial points (old: 128 TODO: INCREASE A LOT?)

    # Get  dataset
    u_ref, r_star = get_dataset(r_0, r_1, n_r)

    # Define domain
    r0 = r_star[0]
    r1 = r_star[-1]

    dom = jnp.array([r0, r1])

    # Initialize model
    model = models.Laplace(config, r_star)

    # Initialize residual sampler. starting with uniform sampling 
    res_sampler = iter(OneDimensionalUniformSampler(dom, config.training.batch_size_per_device))

    evaluator = models.LaplaceEvaluator(config, model)

    # jit warm up
    print("Waiting for JIT...")
    for step in range(config.training.max_steps):
        
        # Update RAD points
        if step % config.setting.resample_every_steps == 0 and step != 0:
            # TODO: Create a RAD sampler by passing x-values and associated normalized model preditions as probabilities.
            r_eval = jnp.linspace(r_0, r_1, 10_000)
            u_eval = model.r_pred_fn(model.state.params, r_eval) # not sure about this
            norm_u_eval = u_eval / jnp.sum(u_eval)
            res_sampler = iter(OneDimensionalRadSampler(r_eval, norm_u_eval, config.training.batch_size_per_device))
            
            if config.plot_rad == True:
                fig = plt.figure(figsize=(8, 8))
                plt.xlabel('Radius [m]')
                plt.ylabel('norm_u_eval')
                plt.title('norm_u_eval')
                plt.plot(r_eval, norm_u_eval, label='norm_u_eval', color='blue')
                plt.grid()
                plt.legend()
                plt.tight_layout()
                
                # Save the figure
                save_dir = os.path.join(workdir, "figures", config.wandb.name)
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)

                fig_path = os.path.join(save_dir, f"rad_prob_{step}.png")
                fig.savefig(fig_path, bbox_inches="tight", dpi=800)

                plt.close(fig)

        start_time = time.time()

        batch = next(res_sampler)

        model.state = model.step(model.state, batch)

        # Update weights
        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                # Get the first replica of the state and batch
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch = jax.device_get(tree_map(lambda x: x[0], batch))
                log_dict = evaluator(state, batch, u_ref)
                wandb.log(log_dict, step)
                end_time = time.time()

                logger.log_iter(step, start_time, end_time, log_dict)

        # Saving
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                path = os.path.join(workdir, "ckpt", config.wandb.name)
                save_checkpoint(model.state, path, keep=config.saving.num_keep_ckpts)
                if config.saving.plot == True:
                    evaluate(config, workdir, step +1)


    return model