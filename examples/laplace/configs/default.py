import ml_collections

import jax.numpy as jnp


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"

    config.setting = setting = ml_collections.ConfigDict()
    setting.r_0 = 0.0001 # Prev best with random sampling: 0.0001 (0.1mm)
    setting.r_1 = 0.5
    setting.u_0 = 1
    setting.u_1 = 0
    setting.n_r = 12_000

    setting.regularization = False
    setting.gpinn = False
    setting.num_grad_points = 100
    
    config.sampler = sampler = ml_collections.ConfigDict()
    sampler.sampler_name = "rad2"
    sampler.resample_every_steps = 20_000 # Resample new RAD points every 10_000 steps
    sampler.num_rad_points = 100_000
    sampler.plot_rad = False
    sampler.c = 1
    sampler.k = 0.5
    sampler.gamma = 0
    sampler.cosine_lr = 0.9
    sampler.cosine_T = 10
    sampler.plot_batch = False 

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PINN-Laplace-RAD2-c1_k05-1e-4"
    wandb.name = "default"
    wandb.tag = None

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "Mlp"
    arch.num_layers = 4
    arch.layer_size = 64
    arch.out_dim = 1
    arch.activation = "gelu"
    arch.periodicity = ml_collections.ConfigDict(
        {"period": (1.0,), "axis": (1,), "trainable": (False,)} 
    )
    arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 10.0, "embed_dim": 256})
    arch.reparam = ml_collections.ConfigDict({"type": "weight_fact", "mean": 1.0, "stddev": 0.1})

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.9
    optim.decay_steps = 2000
    optim.grad_accum_steps = 0

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 150_000
    training.batch_size_per_device = 8192

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = None
    weighting.init_weights = ml_collections.ConfigDict({"res": 1.0})
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000

    weighting.use_causal = False
    weighting.causal_tol = 1.0
    weighting.num_chunks = 32

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 1000
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = True
    logging.log_grads = False
    logging.log_ntk = False
    logging.log_preds = False

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = None
    saving.plot = False
    saving.num_keep_ckpts = None

    # # Input shape for initializing Flax models
    config.input_dim = 1

    # Integer for PRNG random seed.
    config.seed = 42

    return config
