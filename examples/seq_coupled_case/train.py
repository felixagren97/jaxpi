import os
import time
import copy

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

import ml_collections

# from absl import logging
import wandb

from jaxpi.samplers import UniformSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_sequential_checkpoints

import models
from utils import get_dataset
from eval import evaluate


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    logger = Logger()
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    # Problem setup
    n_t = 200  # number of time steps TODO: Increase?
    n_x = 128  # number of spatial points

    # Get  dataset
    u_ref, n_ref, t_star, x_star = get_dataset(n_t, n_x)

    # Define domain
    t0 = t_star[0]
    t1 = t_star[-1]

    x0 = x_star[0]
    x1 = x_star[-1]

    dom = jnp.array([[t0, t1], [x0, x1]])

    # Initialize models
    
    # Regarding configs, both models use the same config file.
    # Currently, model-specific configs are overwritten in the following way: 
    
    # Config for u_model
    config.weighting.init_weights = ml_collections.ConfigDict({ 
            "ru": 1.0,
        })
    
    u_model = models.UModel(config, t_star, x_star, None)
    u_evaluator = models.UModelEvalutor(config, u_model)
    
    # Specific Config for n_model
    n_config = copy.deepcopy(config)   # Copy to avoid alising
    n_config.arch.activation = "sigmoid"
    n_config.arch.arch_name = "MlpDriftDiffusion"
    n_config.weighting.init_weights = ml_collections.ConfigDict({
            "ics": 1.0,
            "bcs_n": 1.0,
            "rn": 1.0
    })

    # u_model is passed to n_model as last argument
    n_model = models.NModel(config, t_star, x_star, u_model)
    n_evaluator = models.NModelEvalutor(config, n_model)

    u_model.n_model = n_model
    
    # Initialize residual sampler
    res_sampler = iter(UniformSampler(dom, config.training.batch_size_per_device))

    # Start training u_model 
    current_model = u_model
    current_evaluator = u_evaluator
    other_model = n_model
    other_evaluator = n_evaluator

    # Initalize other model parameters
    current_model.update_params()
    other_model.update_params()

    # jit warm up
    print("Waiting for JIT...")
    for step in range(config.training.max_steps):
        start_time = time.time()
        batch = next(res_sampler)

        # alternate current_model between u_model and n_model
        if step % config.setting.switch_every_step == 0:
            current_model, other_model = other_model, current_model
            current_evaluator, other_evaluator = other_evaluator, current_evaluator
            current_model.update_params() # get new weights from old model before training new

        current_model.state = current_model.step(current_model.state, batch)

        # Update weights
        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                current_model.state = current_model.update_weights(current_model.state, batch)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                # Get log for current model 
                state = jax.device_get(tree_map(lambda x: x[0], current_model.state))
                batch = jax.device_get(tree_map(lambda x: x[0], batch))
                log_current = current_evaluator(state, batch, u_ref, n_ref)

                # Get log for other model
                state = jax.device_get(tree_map(lambda x: x[0], other_model.state))
                log_other = other_evaluator(state, batch, u_ref, n_ref)

                # Create joint log
                log_dict = log_current | log_other
                
                # Log to wandb and log
                wandb.log(log_dict, step)
                
                end_time = time.time()
                logger.log_iter(step, start_time, end_time, log_dict)

        # Saving
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                save_sequential_checkpoints(config, workdir, current_model, other_model)
                if config.saving.plot == True:
                    evaluate(config, workdir, step + 1)

    return current_model, current_evaluator
