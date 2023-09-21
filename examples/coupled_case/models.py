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
        #self.W = self.mu_n * self.E_ext
        self.Diff = self.mu_n * self.kb * self.Temp/self.q 
        self.epsilon = 8.85e-12

        # Scale factor for charge density to speed up training. 
        self.n_scale = n_inj

        # initial conditions
        self.n_inj = n_inj / self.n_scale
        self.n_0 = n_0 / self.n_scale
        self.n_injs = jnp.full_like(t_star, self.n_inj)
        self.n_0s = jnp.full_like(x_star, self.n_0)
        self.u_0s = jnp.full_like(t_star, u_0)
        self.u_1s = jnp.full_like(t_star, u_1)
        self.u_0 = u_0
        
        # domain
        self.t_star = t_star
        self.x_star = x_star
        self.x0 = x_star[0]
        self.x1 = x_star[-1]


        self.t0 = t_star[0]
        self.t1 = t_star[-1]

        # Predictions over a grid
        self.u_pred_fn = vmap(vmap(self.u_net, (None, None, 0)), (None, 0, None))
        self.n_pred_fn = vmap(vmap(self.scaled_n_net, (None, None, 0)), (None, 0, None))
        self.r_pred_fn = vmap(self.r_net, (None, 0, 0))


    def neural_net(self, params, t, x):
        z = jnp.stack([t, x])
        outputs = self.state.apply_fn(params, z)
        #print('Shape output in neural_net: ', outputs.shape)
        #print('Shape u (output[0]) in neural_net: ', outputs[0].shape)
        #print('Shape n (output[1]) in neural_net: ', outputs[1].shape)
        u = outputs[0]
        n = outputs[1]
        return u, n
    
    def u_net(self, params, t, x):
        u, _ = self.neural_net(params, t, x)
        u = (self.x1-x)/(self.x1-self.x0) * self.u_0 + (x-self.x0)*(self.x1 - x) * u # hard boundary
        return u

    def n_net(self, params, t, x):
        _, n = self.neural_net(params, t, x)
        return n
    
    def scaled_n_net(self, params, t, x):
        return self.n_scale*self.n_net(params, t, x)

    def r_net(self, params, t, x):
        u, n = self.neural_net(params, t, x)
        du_xx = grad(grad(self.u_net, argnums=2), argnums=2)(params, t, x)
        dn_t = grad(self.n_net, argnums=1)(params, t, x)
        dn_x = grad(self.n_net, argnums=2)(params, t, x)
        dn_xx = grad(grad(self.n_net, argnums=2), argnums=2)(params, t, x)

        E = -grad(self.u_net, argnums=2)(params, t, x)
        W = self.mu_n * E
        source = (self.q / self.epsilon * n) * self.n_scale # scale back with n_inj  # TODO: makes sense?
        
        rn = 1/W*dn_t + dn_x - self.Diff/W*dn_xx
        ru = du_xx + source
        return ru, rn

    def ru_net(self, params, t, x):
        ru, _ = self.r_net(params, t, x)
        return ru

    def rn_net(self, params, t, x):
        _, rn = self.r_net(params, t, x)
        return rn

    @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch):
        # Sort temporal coordinates for computing temporal weights
        t_sorted = batch[:, 0].sort()
        # Compute residuals over the full domain
        ru_pred, rn_pred = self.r_pred_fn(params, t_sorted, batch[:, 1])
        # Split residuals into chunks
        ru_pred = ru_pred.reshape(self.num_chunks, -1)
        rn_pred = rn_pred.reshape(self.num_chunks, -1)

        ru_l = jnp.mean(ru_pred**2, axis=1)
        rn_l = jnp.mean(rn_pred**2, axis=1)
        # Compute temporal weights
        ru_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ ru_l)))
        rn_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ rn_l)))

        # Take minimum of the causal weights
        gamma = jnp.vstack([ru_gamma, rn_gamma])
        gamma = gamma.min(0)

        return ru_l, rn_l, gamma

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # Initial loss 
        n_pred = vmap(self.n_net, (None, None, 0))(params, self.t0, self.x_star)
        ics_loss = jnp.mean((self.n_0s[1:] - n_pred[1:]) ** 2) # slicing to exclude x = 0

        # Boundary loss: n(x=0)=n_inj
        x_0 = 0
        n_pred = vmap(self.n_net, (None, 0, None))(params, self.t_star, x_0)
        bcs_n = jnp.mean((self.n_injs - n_pred) ** 2)

        # Boundary loss: U(x=0)=U_0
        #u_pred = vmap(self.u_net, (None, 0, None))(params, self.t_star, x_0)
        #bcs_inner = jnp.mean((self.u_0s - u_pred) ** 2)

        # Boundary loss: U(x=0)=U_0
        x_1 = 1
        #u_pred = vmap(self.u_net, (None, 0, None))(params, self.t_star, x_1)
        #bcs_outer = jnp.mean((self.u_1s - u_pred) ** 2)

        # Residual loss
        if self.config.weighting.use_causal == True:
            ru_l, rn_l, gamma = self.res_and_w(params, batch)
            ru_loss = jnp.mean(ru_l * gamma)
            rn_loss = jnp.mean(rn_l * gamma)
        else:
            ru_pred, rn_pred = self.r_pred_fn(params, batch[:, 0], batch[:, 1])
            # Compute loss
            ru_loss = jnp.mean(ru_pred**2)
            rn_loss = jnp.mean(rn_pred**2)

        loss_dict = {
            "ics": ics_loss,
            "bcs_n": bcs_n, 
            #"bcs_inner": bcs_inner, Hard boundary
            #"bcs_outer": bcs_outer, Hard boundary
            "ru": ru_loss,
            "rn": rn_loss
        }
        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_diag_ntk(self, params, batch):
        # n(t=0)
        ics_ntk = vmap(ntk_fn, (None, None, None, 0))(
            self.n_net, params, self.t0, self.x_star
        )
        #TODO: Do we need to specify boundary values somewhere?
        # Boundary loss: n(x=0)=n_inj
        x_0 = 0
        bcs_n_ntk = vmap(ntk_fn, (None, None, 0, None))(self.n_net, params, self.t_star, x_0)

        # Boundary loss: U(x=0)=u_0
        #bcs_inner_ntk = vmap(ntk_fn, (None, None, 0, None))(self.u_net, params, self.t_star, x_0)

        # Boundary loss: U(x=0)=u_1
        x_1 = 1
        #bcs_outer_ntk = vmap(self.u_net, (None, 0, None))(params, self.t_star, x_1)

        # Residual loss
        if self.config.weighting.use_causal:
            # sort the time step for causal loss
            batch = jnp.array([batch[:, 0].sort(), batch[:, 1]]).T
            
            u_res_ntk = vmap(ntk_fn, (None, None, 0, 0))(
                self.u_net, params, batch[:, 0], batch[:, 1]
            )
            n_res_ntk = vmap(ntk_fn, (None, None, 0, 0))(
                self.n_net, params, batch[:, 0], batch[:, 1]
            )

            # shape: (num_chunks, -1)
            u_res_ntk = u_res_ntk.reshape(self.num_chunks, -1)  
            n_res_ntk = n_res_ntk.reshape(self.num_chunks, -1)  
            
            # average convergence rate over each chunk
            u_res_ntk = jnp.mean(u_res_ntk, axis=1)
            n_res_ntk = jnp.mean(n_res_ntk, axis=1)

            # multiply by causal weights
            _, _, casual_weights = self.res_and_w(params, batch)
            u_res_ntk *= casual_weights
            n_res_ntk *= casual_weights
        else:
            u_res_ntk = vmap(ntk_fn, (None, None, 0, 0))(
                self.u_net, params, batch[:, 0], batch[:, 1]
            )
            n_res_ntk = vmap(ntk_fn, (None, None, 0, 0))(
                self.n_net, params, batch[:, 0], batch[:, 1]
            )

        ntk_dict = {
            "ics": ics_ntk,
            "bcs_n": bcs_n_ntk, 
            #"bcs_inner": bcs_inner_ntk, Hard boundary
            #"bcs_outer": bcs_outer_ntk, Hard boundary
            "ru": u_res_ntk,
            "rn": n_res_ntk
        }

        return ntk_dict


    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u_ref, n_ref):
        #TODO: Other methods have implemented for general t,x arrays, should we? 
        u_pred = self.u_pred_fn(params, self.t_star, self.x_star)
        n_pred = self.n_pred_fn(params, self.t_star, self.x_star)
        
        u_error = jnp.linalg.norm(u_pred - u_ref) / jnp.linalg.norm(u_ref)
        n_error = jnp.linalg.norm(n_pred - n_ref) / jnp.linalg.norm(n_ref)
        return u_error, n_error


class CoupledCaseEvalutor(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u_ref, n_ref):
        u_error, n_error = self.model.compute_l2_error(params, u_ref, n_ref)
        self.log_dict["u_error"] = u_error
        self.log_dict["n_error"] = n_error

    def log_preds(self, params):
        pass

    def __call__(self, state, batch, u_ref, n_ref):
        self.log_dict = super().__call__(state, batch)

        if self.config.weighting.use_causal:
            _, _, causal_weight = self.model.res_and_w(state.params, batch)
            self.log_dict["cas_weight"] = causal_weight.min()

        if self.config.logging.log_errors:
            self.log_errors(state.params, u_ref, n_ref)

        if self.config.logging.log_preds:
            self.log_preds(state.params)

        return self.log_dict
