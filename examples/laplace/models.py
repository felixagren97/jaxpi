from functools import partial

import jax.numpy as jnp
from jax import lax, jit, grad, vmap

from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn, flatten_pytree

from matplotlib import pyplot as plt


class Laplace(ForwardIVP):
    def __init__(self, config, u0, u1, r_star):
        super().__init__(config)

        self.u0 = u0
        self.u1 = u1
        self.r_star = r_star

        self.r0 = r_star[0]
        self.r1 = r_star[-1]

        # Predictions over a grid
        self.u_pred_fn = vmap(self.u_net)
        self.r_pred_fn = vmap(self.r_net)

        # old: self.u_pred_fn = vmap(vmap(self.u_net, (None, None, 0)), (None, 0, None))
        #      self.r_pred_fn = vmap(vmap(self.r_net, (None, None, 0)), (None, 0, None))

    def u_net(self, params, r):
        # params = weights for NN 
        r = jnp.reshape(r, (len(r), 1))
        u = self.state.apply_fn(params, r) # gives r to the neural network's (self.state) forward pass (apply_fn)
        return u[0]

    def r_net(self, params, r):
        du_r = grad(self.u_net)(params, r)
        du_rr = grad(grad(self.u_net))(params, r) # Don't need to use hessian b/c scalar f and r        
        return r * du_rr + du_r  # Scaled by r, try w/o? 

    @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch): #TODO: think should never be called
        # Sort temporal coordinates for computing temporal weights
        t_sorted = batch[:, 0].sort()
        # Compute residuals over the full domain
        r_pred = vmap(self.r_net, (None, 0, 0))(params, t_sorted, batch[:, 1])
        # Split residuals into chunks
        r_pred = r_pred.reshape(self.num_chunks, -1)
        l = jnp.mean(r_pred**2, axis=1)
        # Compute temporal weights
        w = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ l)))
        return l, w

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # Initial condition loss
        u_pred = vmap(self.u_net, (None, 0))(params, self.r_star)

        ics_loss = jnp.mean((self.u0 - u_pred) ** 2)

        # Residual loss
        if self.config.weighting.use_causal == True:
            l, w = self.res_and_w(params, batch)
            res_loss = jnp.mean(l * w)
        else:
            #r_pred = vmap(self.r_net, (None, 0, 0))(params, batch[:, 0], batch[:, 1])
            r_pred = vmap(self.r_net, (None, 0))(params, batch[:, 0])
            res_loss = jnp.mean((r_pred) ** 2)

        loss_dict = {"ics": ics_loss, "res": res_loss}
        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_diag_ntk(self, params, batch):
        # TODO: adopt to 1d
        #ics_ntk = vmap(ntk_fn, (None, None, None, 0))(
        #    self.u_net, params, self.t0, self.r_star
        #)
        ics_ntk = vmap(ntk_fn, (None, None, 0))(
            self.u_net, params, self.r_star
        )

        # Consider the effect of causal weights
        if self.config.weighting.use_causal: # Think should always be false, b/c no temporal domain
            # sort the time step for causal loss
            batch = jnp.array([batch[:, 0].sort(), batch[:, 1]]).T
            res_ntk = vmap(ntk_fn, (None, None, 0, 0))(
                self.r_net, params, batch[:, 0], batch[:, 1]
            )

            res_ntk = res_ntk.reshape(self.num_chunks, -1)  # shape: (num_chunks, -1)
            res_ntk = jnp.mean(
                res_ntk, axis=1
            )  # average convergence rate over each chunk
            _, casual_weights = self.res_and_w(params, batch)
            res_ntk = res_ntk * casual_weights  # multiply by causal weights
        else:
            res_ntk = vmap(ntk_fn, (None, None, 0, 0))(
                self.r_net, params, batch[:, 0], batch[:, 1]
            )

        ntk_dict = {"ics": ics_ntk, "res": res_ntk}

        return ntk_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u_test):
        u_pred = self.u_pred_fn(params, self.r_star)
        error = jnp.linalg.norm(u_pred - u_test) / jnp.linalg.norm(u_test)
        return error


class LaplaceEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u_ref):
        l2_error = self.model.compute_l2_error(params, u_ref)
        self.log_dict["l2_error"] = l2_error

    def log_preds(self, params):
        u_pred = self.model.u_pred_fn(params, self.model.r_star)
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