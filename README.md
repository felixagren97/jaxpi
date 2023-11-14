# JAX-PI

This repository is an adaptation of the [JAXPI repository from Predictive Intelligence Lab](https://github.com/PredictiveIntelligenceLab/jaxpi) for a master's thesis at the Department of Electrical Engineering at Chalmers University of Technology. It offers a comprehensive implementation of physics-informed neural networks (PINNs), seamlessly integrating several advanced network architectures, and training algorithms from [An Expert's Guide to Training Physics-informed Neural Networks
](https://arxiv.org/abs/2308.08468) adapted to the domain of discharge physics. The implementation supports both **single** and **multi-GPU** training, while evaluation is currently limited to single-GPU setups.

## Demo 
A demonstration notebook for running the code on Google Colab is available [here](https://colab.research.google.com/drive/1a33Zx5J9NJ3mn8uNzxFKQq_m0DLjST9Q?usp=sharing). The next sections will provide a more thorough description of how to use the repo. 

## Installation

Ensure that you have Python 3.8 or later installed on your system.
Our code is GPU-only.
We highly recommend using the most recent versions of JAX and JAX-lib, along with compatible CUDA and cuDNN versions.
The code has been tested and confirmed to work with the following versions:

- JAX 0.4.5
- CUDA 11.7
- cuDNN 8.2


Install JAX-PI with the following commands:

``` 
git clone https://github.com/felixagren97/jaxpi.git
cd jaxpi
pip install .
```

## Quickstart

We use [Weights & Biases](https://wandb.ai/site) to log and monitor training metrics. 
Please ensure you have Weights & Biases installed and properly set up with your account before proceeding. 
You can follow the installation guide provided [here](https://docs.wandb.ai/quickstart).

To illustrate how to use our code, we will use the advection equation as an example. 
First, navigate to the advection directory within the `examples` folder:

``` 
cd jaxpi/examples/laplace
``` 
To train the model, run the following command:
```
python3 main.py 
```

Our code automatically supports multi-GPU execution. 
You can specify the GPUs you want to use with the `CUDA_VISIBLE_DEVICES` environment variable. For example, to use the first two GPUs (0 and 1), use the following command:

```
CUDA_VISIBLE_DEVICES=0,1 python3 main.py
```

**Note on Memory Usage**: Different models and examples may require varying amounts of GPU memory. 
If you encounter an out-of-memory error, you can decrease the batch size using the `--config.batch_size_per_device` option.

To evaluate the model's performance, you can switch to evaluation mode with the following command:

```
python main.py --config.mode=eval
```


