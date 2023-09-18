from functools import partial

import jax.numpy as jnp
from jax import lax, jit, grad, vmap

from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn, flatten_pytree

from matplotlib import pyplot as plt


class CoupledCase(ForwardIVP):
    def __init__(self, config, n_inj, n_0, u_0, u_1, t_star, x_star):
        super().__init__(config)

        # constants
        self.mu_n = 2e-4
        self.Temp = 293
        self.q = 1.602e-19
        self.kb = 1.38e-23
        self.W = self.mu_n * self.E_ext
        self.Diff = self.mu_n * self.kb * self.Temp/self.q 
        self.epsilon = 8.85e-12

        # initial conditions
        self.n_inj = n_inj
        self.n_0 = n_0
        self.n_injs = jnp.full_like(t_star, n_inj)
        self.n_0s = jnp.full_like(x_star, n_0)
        
        # domain
        self.t_star = t_star
        self.x_star = x_star

        self.t0 = t_star[0]
        self.t1 = t_star[-1]

        # Predictions over a grid
        self.u_pred_fn = vmap(vmap(self.u_net, (None, None, 0)), (None, 0, None))
        self.r_pred_fn = vmap(vmap(self.r_net, (None, None, 0)), (None, 0, None))

    def neural_net(self, params, t, x):
        z = jnp.stack([t, x])
        outputs = self.state.apply_fn(params, z)
        u = outputs[0]
        n = outputs[1]
        return u, n
    
    def u_net(self, params, t, x):
        u, _ = self.neural_net(params, t, x)
        return u

    def n_net(self, params, t, x):
        _, n = self.neural_net(params, t, x)
        return n

    def r_net(self, params, t, x):
        """
        n, U = y[:, 0:1], y[:, 1:2]
        dn_t = dde.grad.jacobian(y, x, i=0, j=1)
        dn_x = dde.grad.jacobian(y, x, i=0, j=0)
        dn_xx = dde.grad.hessian(y, x, component = 0, i=0, j=0)
        E = -dde.grad.jacobian(y,x,i=1,j=0)  #E = -du/dx
        dU_xx = dde.grad.hessian(y, x, component = 1, i=0, j=0)
        W = mu_n*E
        source = q/epsilon*n*1e9    # here I multiply by 1e9 to scale the problem correctly
        # multiply with n_inj again to rescale n before feeding¢
        return [1/W*dn_t + dn_x- Diff/W*dn_xx, dU_xx + source]"""
    
        u, n = self.neural_net(params, t, x)
        du_xx = grad(grad(self.u_net, argnums=2), argnums=2)(params, t, x)
        dn_t = grad(self.n_net, argnums=1)(params, t, x)
        dn_x = grad(self.n_net, argnums=2)(params, t, x)
        dn_xx = grad(grad(self.n_net, argnums=2), argnums=2)(params, t, x)

        E = -grad(self.u_net, argnums=2)(params, t, x)
        W = self.mu_n * E
        source = (self.q / self.epsilon * n) * self.n_inj # scale back with n_inj  # TODO: makes sense?
        
        rn = 1/W*dn_t + dn_x - self.Diff/W*dn_xx
        ru = du_xx + source
        return rn, ru

    def rn_net(self, params, t, x):
        rn, _ = self.r_net(params, t, x)
        return rn
    
    def ru_net(self, params, t, x):
        _, ru = self.r_net(params, t, x)
        return ru

    @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch):
        # Sort temporal coordinates for computing  temporal weights
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
        # Initial loss 
        u_pred = vmap(self.u_net, (None, None, 0))(params, self.t0, self.x_star)
        ics_loss = jnp.mean((self.n_0s[1:] - u_pred[1:]) ** 2) # slicing to exclude x = 0

        # Boundary loss
        x_0 = 0
        u_pred = vmap(self.u_net, (None, 0, None))(params, self.t_star, x_0)
        bcs_loss = jnp.mean((self.n_injs - u_pred) ** 2)

        # Residual loss
        if self.config.weighting.use_causal == True:
            l, w = self.res_and_w(params, batch)
            res_loss = jnp.mean(l * w)
        else:
            r_pred = vmap(self.r_net, (None, 0, 0))(params, batch[:, 0], batch[:, 1]) 
            res_loss = jnp.mean((r_pred) ** 2)

        loss_dict = {"ics": ics_loss, "bcs": bcs_loss, "res": res_loss}
        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_diag_ntk(self, params, batch):
        ics_ntk = vmap(ntk_fn, (None, None, None, 0))(
            self.u_net, params, self.t0, self.x_star
        )

        # Consider the effect of causal weights
        if self.config.weighting.use_causal:
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
        u_pred = self.u_pred_fn(params, self.t_star, self.x_star)
        error = jnp.linalg.norm(u_pred - u_test) / jnp.linalg.norm(u_test)
        return error


class CoupledCaseEvalutor(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u_ref):
        l2_error = self.model.compute_l2_error(params, u_ref)
        self.log_dict["l2_error"] = l2_error

    def log_preds(self, params):
        u_pred = self.model.u_pred_fn(params, self.model.t_star, self.model.x_star)
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
