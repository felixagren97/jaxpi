import ml_collections

import jax.numpy as jnp


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"

    # Setting
    config.setting = setting = ml_collections.ConfigDict()
    setting.switch_every_step = 5_000
    setting.u_0 = 1e6
    setting.u_1 = 0
    setting.n_0 = 0.1
    setting.n_inj = 5e13
    setting.loss_scale = 1.0 # rescale residual loss for u with this factor before squaring (low, positive value to avoid NaN)
    setting.n_model_activation = 'sigmoid' # Activation funtion on hidden layers for n_model

    # Evaluate 
    config.eval = eval = ml_collections.ConfigDict()
    # COMSOL reference solution files (set None if not available for the current n_inj)
    eval.ion_density_file_path = 'Case3-ninj_all-Conc.txt.txt'
    eval.potential_file_path = 'Case3-ninj_all-Pot.txt.txt'
    eval.field_file_path = 'Case3-ninj_all-Field.txt.txt'

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "Ablation-PINN-Sequential-Coupled-case-5e13"
    wandb.name = "no_fourier_feature"
    wandb.tag = None

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "Mlp"
    arch.num_layers = 6
    arch.layer_size = 256
    arch.out_dim = 1
    arch.activation = "gelu"
    arch.periodicity = False 
    arch.fourier_emb = None
    arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 1.0, "stddev": 0.1}
    )

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
    training.max_steps = 200000
    training.batch_size_per_device = 4096

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = None #"grad_norm"
    weighting.init_weights = ml_collections.ConfigDict({
            "ics": 1.0,
            "bcs_n": 1.0, 
            "ru": 1.0,
            "rn": 1.0
        })
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
    logging.log_weights = False
    logging.log_grads = False
    logging.log_ntk = False
    logging.log_preds = False

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 50000
    saving.num_keep_ckpts = 1
    saving.plot = True

    # # Input shape for initializing Flax models
    config.input_dim = 2

    # Integer for PRNG random seed.
    config.seed = 42

    return config
