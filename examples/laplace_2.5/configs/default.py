import ml_collections

import jax.numpy as jnp


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"

    # Problem setting 
    config.setting = setting = ml_collections.ConfigDict()
    setting.n_scale = 5e13
    setting.n_x = 12800
    setting.u0 = 1e6
    setting.u1 = 0
    setting.k = 25
    setting.loss_scale = 1

    # Evaluate 
    config.eval = eval = ml_collections.ConfigDict()
    # COMSOL reference solution files (set None if not available for the current n_inj
    eval.potential_file_path = 'Case1p5_validation_data_U_vs_x_ninj5e13.txt(1).txt'
    eval.field_file_path = 'Case1p5_validation_data_E_vs_x_ninj5e13.txt(1).txt'

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PINN-Laplace-2.5"
    wandb.name = "default"
    wandb.tag = None

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "Mlp"
    arch.num_layers = 4
    arch.layer_size = 256
    arch.out_dim = 1
    arch.activation = "tanh"
    arch.periodicity = ml_collections.ConfigDict({"period": (1.0, ), "axis": (1,), "trainable": (False,)}) # CANNOT BE NONE/False 
    arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 1.0, "embed_dim": 256})
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
    training.max_steps = 200000
    training.batch_size_per_device = 4096

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = None 
    weighting.init_weights = ml_collections.ConfigDict({"inner_bcs": 1.0, "outer_bcs": 1.0, "res": 1.0})
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000

    weighting.use_causal = False # TODO: verify: was true, but changed to false as no temporal domain
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
    saving.save_every_steps = 10000
    saving.num_keep_ckpts = 10
    saving.plot = True

    # # Input shape for initializing Flax models
    config.input_dim = 1

    # Integer for PRNG random seed.
    config.seed = 42

    return config
