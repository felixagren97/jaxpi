import ml_collections

import jax.numpy as jnp


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"

    config.setting = setting = ml_collections.ConfigDict()
    setting.r_0 = 0.001
    setting.r_1 = 0.5
    setting.u0 = 1
    setting.u1 = 0
    config.setting.n_r = 12_000

    setting.n_obs = 1_000
    setting.guassian_noise_perc = None
    setting.loss_scale = 1e-10

    setting.true_rho = 1e-2
    setting.rho_scale = 1e-4

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PINN-Inverse-Poisson-Ablation"
    wandb.name = "no_rwf"
    wandb.tag = None

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "InverseMlpRho"
    arch.num_layers = 6
    arch.layer_size = 256
    arch.out_dim = 1
    arch.activation = "gelu"
    arch.periodicity = ml_collections.ConfigDict(
        {"period": (1.0,), "axis": (1,), "trainable": (False,)} 
    )
    arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 10.0, "embed_dim": 256})
    arch.reparam = None

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
    training.max_steps = 200_000
    training.batch_size_per_device = 516

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = ml_collections.ConfigDict({"res": 1.0, "observ": 1.0})
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000

    weighting.use_causal = False
    weighting.causal_tol = 1.0
    weighting.num_chunks = 32

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = True
    logging.log_grads = False
    logging.log_ntk = False
    logging.log_preds = False

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 10_000
    saving.num_keep_ckpts = 1
    saving.plot = True

    # # Input shape for initializing Flax models
    config.input_dim = 1

    # Integer for PRNG random seed.
    config.seed = 42

    return config
