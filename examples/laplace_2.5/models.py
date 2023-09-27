from functools import partial

import jax.numpy as jnp
from jax import lax, jit, grad, vmap

from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn, flatten_pytree

from matplotlib import pyplot as plt


class Laplace(ForwardIVP):
    def __init__(self, config, u0, u1, x_star, n_inj):
        super().__init__(config)

        self.u0 = u0
        self.u1 = u1
        self.x_star = x_star
        self.n_inj = n_inj

        self.x0 = x_star[0]
        self.x1 = x_star[-1]

        # parameters 
        self.q = 1.602e-19
        self.epsilon = 8.85e-12

        # mappings  
        self.u_pred_fn = vmap(self.u_net, (None, 0))
        self.r_pred_fn = vmap(self.r_net, (None, 0))

    def u_net(self, params, x):
        x_reshape = jnp.reshape(x, (1, -1)) # make it a 2d array with just one column to emulate jnp.stack()
        u = self.state.apply_fn(params, x_reshape) # gives r to the neural network's (self.state) forward pass (apply_fn)
        #return u[0] # soft boundary
        return self.u0*(self.x1-x)/(self.x1-self.x0) + (x-self.x0)*(self.x1 - x)*u[0] # hard boundary

    def heaviside(self, x, k=25, a=0.5):
        # https://en.wikipedia.org/wiki/Heaviside_step_function
        # larger k -> steeper step
        # larger a -> larger positive translation
        return 1 - 1 / (1 + jnp.exp(-2 * k * (x - a)))

    def r_net(self, params, x):        
        du_xx = grad(grad(self.u_net, argnums=1), argnums=1)(params, x)
        n = self.n_inj * self.heaviside(x=x, k=25, a=0.5)
        return du_xx + self.q * n / self.epsilon

    @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch): #TODO: think should never be called
        raise NotImplementedError(f"Casual weights not supported yet for 1D Laplace!")

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # inner boundary condition 
        #u_pred = self.u_net(params, self.x0) #Hard boundary
        #inner_bcs_loss = jnp.mean((self.u0 - u_pred) ** 2) #Hard boundary

        # outer boundary condition 
        #u_pred = self.u_net(params, self.x1) #Hard boundary
        #outer_bcs_loss = jnp.mean((self.u1 - u_pred) ** 2) #Hard boundary

        # Residual loss
        if self.config.weighting.use_causal == True:
            raise NotImplementedError(f"Casual weights not supported yet for 1D Laplace!")
        else:
            r_pred = vmap(self.r_net, (None, 0))(params, batch[:,0]) 
            res_loss = jnp.mean((r_pred) ** 2)

        loss_dict = {"res": res_loss} # #Hard boundary {"inner_bcs": inner_bcs_loss, "outer_bcs": outer_bcs_loss, "res": res_loss}
        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_diag_ntk(self, params, batch):

        #inner_bcs_ntk = self.u_net(params, self.x0)
        #outer_bcs_ntk = self.u_net(params, self.x1)


        # Consider the effect of causal weights
        if self.config.weighting.use_causal: 
            raise NotImplementedError(f"Casual weights not supported yet for 1D Laplace!")

        else:
            res_ntk = vmap(ntk_fn, (None, None, 0))(
                self.r_net, params, batch[:, 0]
            )
        #ntk_dict = {"ics": ics_ntk, "res": res_ntk}
        ntk_dict = {"res": res_ntk} #{"inner_bcs": inner_bcs_ntk, "outer_bcs": outer_bcs_ntk, "res": res_ntk}

        return ntk_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u_test):
        u_pred = self.u_pred_fn(params, self.x_star)
        error = jnp.linalg.norm(u_pred - u_test) / jnp.linalg.norm(u_test)
        return error


class LaplaceEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u_ref):
        l2_error = self.model.compute_l2_error(params, u_ref)
        self.log_dict["l2_error"] = l2_error

    def log_preds(self, params):
        u_pred = self.model.u_pred_fn(params, self.model.x_star)
        fig = plt.figure(figsize=(6, 5))
        plt.imshow(u_pred.T, cmap="jet")
        self.log_dict["u_pred"] = fig
        plt.close()

    def __call__(self, state, batch, u_ref):
        self.log_dict = super().__call__(state, batch)

        if self.config.weighting.use_causal:
            _, causal_weight = self.model.res_and_w(state.params, batch)
            self.log_dict["cas_weight"] = causal_weight.min()

        if self.config.logging.log_errors:
            self.log_errors(state.params, u_ref)

        if self.config.logging.log_preds:
            self.log_preds(state.params)

        return self.log_dict