from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, jit, grad, vmap
from jax.tree_util import tree_map

from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn, flatten_pytree

from matplotlib import pyplot as plt

class UModel(ForwardIVP):
    def __init__(self, config, t_star, x_star, n_model):
        super().__init__(config)

        # Constants
        self.q = 1.602e-19
        self.epsilon = 8.85e-12

        # initial conditions
        self.u_0 = config.setting.u_0
        self.u_1 = config.setting.u_1

        self.u_0s = jnp.full_like(t_star, self.u_0)
        self.u_1s = jnp.full_like(t_star, self.u_1)
        
        
        # domain
        self.t_star = t_star
        self.x_star = x_star
        self.x0 = x_star[0]
        self.x1 = x_star[-1]


        self.t0 = t_star[0]
        self.t1 = t_star[-1]

        # Reference to n model
        self.n_model = n_model

        # Predictions over a grid
        self.u_pred_fn = vmap(vmap(self.u_net, (None, None, 0)), (None, 0, None))
        self.r_pred_fn = vmap(self.r_net, (None, 0, 0))
        #self.n_pred_fn = vmap(self.n_model.n_net, (None, 0, 0))

        self.tag = "u_model"

    #def set_n_model(self, n_model):
    #    self.n_model = n_model 
    #    self.n_pred_fn = vmap(self.n_model.n_net, (None, 0, 0))


    def u_net(self, params, t, x):
        z = jnp.stack([t, x])
        outputs = self.state.apply_fn(params, z)
        u = outputs[0]
        u = (self.x1-x)/(self.x1-self.x0) * self.u_0 + (x-self.x0)*(self.x1 - x) * u # hard boundary
        return u
    
    def r_net(self, params, t, x):
        # parameters of the n model
        n_state = jax.device_get(tree_map(lambda x: x[0], self.n_model.state))
        n_params = n_state.params

        du_xx = grad(grad(self.u_net, argnums=2), argnums=2)(params, t, x)
        source = (self.q / self.epsilon * self.n_model.n_net(n_params, t, x)) * self.n_model.n_scale # scale back with n_inj 
        
        ru = du_xx + source
        return ru
    
    @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch):
        # Sort temporal coordinates for computing temporal weights
        t_sorted = batch[:, 0].sort()
        # Compute residuals over the full domain
        ru_pred = self.r_pred_fn(params, t_sorted, batch[:, 1])
        # Split residuals into chunks
        ru_pred = ru_pred.reshape(self.num_chunks, -1)
        
        ru_l = jnp.mean(ru_pred**2, axis=1)
        # Compute temporal weights
        w = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ ru_l)))
    
        return ru_l, w
    
    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        
        # Boundary loss: U(x=0)=U_0
        #u_pred = vmap(self.u_net, (None, 0, None))(params, self.t_star, x_0)
        #bcs_inner = jnp.mean((self.u_0s - u_pred) ** 2)

        # Boundary loss: U(x=0)=U_0
        #x_1 = 1
        #u_pred = vmap(self.u_net, (None, 0, None))(params, self.t_star, x_1)
        #bcs_outer = jnp.mean((self.u_1s - u_pred) ** 2)

        # Residual loss
        if self.config.weighting.use_causal == True:
            ru_l, w = self.res_and_w(params, batch)
            ru_loss = jnp.mean(ru_l * w)

        else:
            ru_pred = self.r_pred_fn(params, batch[:, 0], batch[:, 1])
            # Compute loss
            ru_loss = jnp.mean(ru_pred**2)
            
        loss_dict = {
            #"bcs_inner": bcs_inner, Hard boundary
            #"bcs_outer": bcs_outer, Hard boundary
            "ru": ru_loss,
        }
        return loss_dict
    
    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u_ref):
        #TODO: Other methods have implemented for general t,x arrays, should we? 
        u_pred = self.u_pred_fn(params, self.t_star, self.x_star)
        
        u_error = jnp.linalg.norm(u_pred - u_ref) / jnp.linalg.norm(u_ref)
        return u_error

class NModel(ForwardIVP):
    def __init__(self, config, t_star, x_star, u_model):
        super().__init__(config)
        self.mu_n = 2e-4
        self.Temp = 293
        self.q = 1.602e-19
        self.kb = 1.38e-23
        #self.W = self.mu_n * self.E_ext
        self.Diff = self.mu_n * self.kb * self.Temp/self.q 
        self.epsilon = 8.85e-12

        self.n_scale = config.setting.n_inj

        # initial conditions
        self.u_0 = config.setting.u_0
        self.u_1 = config.setting.u_1
        
        self.n_inj = self.n_scale / self.n_scale
        self.n_0 = config.setting.n_0 / self.n_scale
        
        self.n_injs = jnp.full_like(t_star, self.n_inj)
        self.n_0s = jnp.full_like(x_star, self.n_0)
        
        self.u_0s = jnp.full_like(t_star, self.u_0)
        self.u_1s = jnp.full_like(t_star, self.u_1)
        
        
        # domain
        self.t_star = t_star
        self.x_star = x_star
        self.x0 = x_star[0]
        self.x1 = x_star[-1]


        self.t0 = t_star[0]
        self.t1 = t_star[-1]

        self.n_pred_fn = vmap(vmap(self.scaled_n_net, (None, None, 0)), (None, 0, None))
        self.r_pred_fn = vmap(self.r_net, (None, 0, 0))

        self.u_model = u_model

        self.tag = "n_model"
    
    def n_net(self, params, t, x):
        z = jnp.stack([t, x])
        outputs = self.state.apply_fn(params, z)
        n = outputs[0]
        return n
    
    def scaled_n_net(self, params, t, x):
        return self.n_scale*self.n_net(params, t, x)

    def r_net(self, params, t, x):
        
        dn_t = grad(self.n_net, argnums=1)(params, t, x)
        dn_x = grad(self.n_net, argnums=2)(params, t, x)
        dn_xx = grad(grad(self.n_net, argnums=2), argnums=2)(params, t, x)

        E = -grad(self.u_model.u_net, argnums=2)(self.u_model.state.params, t, x)
        W = self.mu_n * E
        
        rn = 1/W*dn_t + dn_x - self.Diff/W*dn_xx
        
        return rn
    
    @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch):
        # Sort temporal coordinates for computing temporal weights
        t_sorted = batch[:, 0].sort()
        # Compute residuals over the full domain
        rn_pred = self.r_pred_fn(params, t_sorted, batch[:, 1])
        # Split residuals into chunks
        rn_pred = rn_pred.reshape(self.num_chunks, -1)

        rn_l = jnp.mean(rn_pred**2, axis=1)
        # Compute temporal weights
        w = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ rn_l)))

        # Take minimum of the causal weights
        return rn_l, w
    
    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # Initial loss 
        n_pred = vmap(self.n_net, (None, None, 0))(params, self.t0, self.x_star)
        ics_loss = jnp.mean((self.n_0s[1:] - n_pred[1:]) ** 2) # slicing to exclude x = 0

        # Boundary loss: n(x=0)=n_inj
        x_0 = 0
        n_pred = vmap(self.n_net, (None, 0, None))(params, self.t_star, x_0)
        bcs_n = jnp.mean((self.n_injs - n_pred) ** 2)
    

        # Residual loss
        if self.config.weighting.use_causal == True:
            rn_l, w = self.res_and_w(params, batch)
            rn_loss = jnp.mean(rn_l * w)
        else:
            rn_pred = self.r_pred_fn(params, batch[:, 0], batch[:, 1])
            # Compute loss
            rn_loss = jnp.mean(rn_pred**2)

        loss_dict = {
            "ics": ics_loss,
            "bcs_n": bcs_n, 
            "rn": rn_loss
        }
        return loss_dict
    
    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, n_ref):
        #TODO: Other methods have implemented for general t,x arrays, should we? 
        n_pred = self.n_pred_fn(params, self.t_star, self.x_star)
        n_error = jnp.linalg.norm(n_pred - n_ref) / jnp.linalg.norm(n_ref)
        return n_error
    


class UModelEvalutor(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u_ref, n_ref):
        u_error = self.model.compute_l2_error(params, u_ref)
        self.log_dict["u_error"] = u_error
        
    def log_preds(self, params):
        pass

    def __call__(self, state, batch, u_ref, n_ref):
        self.log_dict = super().__call__(state, batch)

        if self.config.weighting.use_causal:
            _, _, causal_weight = self.model.res_and_w(state.params, batch)
            self.log_dict["cas_weight"] = causal_weight.min()

        if self.config.logging.log_errors:
            self.log_errors(state.params, u_ref)

        if self.config.logging.log_preds:
            self.log_preds(state.params)

        return self.log_dict

class NModelEvalutor(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u_ref, n_ref):
        n_error = self.model.compute_l2_error(params, n_ref)
        self.log_dict["n_error"] = n_error
        
    def log_preds(self, params):
        pass

    def __call__(self, state, batch, u_ref, n_ref):
        self.log_dict = super().__call__(state, batch)

        if self.config.weighting.use_causal:
            _, _, causal_weight = self.model.res_and_w(state.params, batch)
            self.log_dict["cas_weight"] = causal_weight.min()

        if self.config.logging.log_errors:
            self.log_errors(state.params, n_ref)

        if self.config.logging.log_preds:
            self.log_preds(state.params)

        return self.log_dict