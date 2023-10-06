from functools import partial
import jax
import jax.numpy as jnp
from jax import lax, jit, grad, vmap

from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn, flatten_pytree

from matplotlib import pyplot as plt


class Laplace(ForwardIVP):
    def __init__(self, config, u0, u1, r_star):
        super().__init__(config)

        self.eps = 8.85e-12
        self.true_rho = 5e-10

        self.u0 = u0
        self.u1 = u1
        self.r_star = r_star

        self.r0 = r_star[0]
        self.r1 = r_star[-1]

        # TODO: Do we need to define these constants here? A bit ugly imo.
        self.C_1 = ((4*self.eps*jnp.log(self.r1) + self.true_rho * self.r0**2 * jnp.log(self.r1) - self.true_rho * self.r1**2 * jnp.log(self.r0)) /
       (4 * self.eps * (-jnp.log(self.r0) + jnp.log(self.r1))))
        self.C_2 = (-4 * self.eps - self.true_rho*self.r0**2 + self.true_rho * self.r1**2) / (4 * self.eps * (-jnp.log(self.r0) + jnp.log(self.r1)))

        self.obs_r = jax.random.uniform(jax.random.PRNGKey(0), (100,), minval=self.r0, maxval=self.r1)
        self.obs_u = self.analytical_potential(self.true_rho, self.obs_r) 

        #new  
        self.u_pred_fn = vmap(self.u_net, (None, 0))
        self.rho_pred_fn = vmap(self.rho_net, (None, 0))
        self.r_pred_fn = vmap(self.r_net, (None, 0))

    def analytical_potential(self, true_rho, r): # TODO: does this work for jnp arrays?
        return self.C_1 + self.C_2 * jnp.log(r) - (true_rho * r**2) / (4 * self.eps)
        
        
    def neural_net(self, params, r):
        # params = weights for NN 
        r_reshape = jnp.reshape(r, (1, -1)) # make it a 2d array with just one column to emulate jnp.stack()
        y = self.state.apply_fn(params, r_reshape) # gives r to the neural network's (self.state) forward pass (apply_fn)
        u = y[0] # first output of the neural network
        rho = y[1] # second output of the neural network
        return u, rho
        #return (self.r1-r)/(self.r1-self.r0) + (r-self.r0)*(self.r1 - r)*u[0] # hard boundary
    
    def u_net(self, params, r):
        return self.neural_net(params, r)[0] # Soft boundary
    
    def rho_net(self, params, r):
        return self.neural_net(params, r)[1]

    def r_net(self, params, r):        
        du_r = grad(self.u_net, argnums=1)(params, r)
        du_rr = grad(grad(self.u_net, argnums=1), argnums=1)(params, r)
        rho = self.rho_net(params, r)
        return r * du_rr + du_r + (1e-10 * rho/self.eps) * r # TODO: check if this is correct  

    @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch): #TODO: think should never be called
        raise NotImplementedError(f"Casual weights not supported yet for 1D Laplace!")

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):    #TODO: Implement loss for observed synthetic data.
        # inner boundary condition 
        u_pred = self.u_net(params, self.r0)
        inner_bcs_loss = jnp.mean((self.u0 - u_pred) ** 2)

        # outer boundary condition 
        u_pred = self.u_net(params, self.r1)
        outer_bcs_loss = jnp.mean((self.u1 - u_pred) ** 2)

        # Residual loss
        if self.config.weighting.use_causal == True:
            raise NotImplementedError(f"Casual weights not supported yet for 1D Laplace!")
        else:
            print('CRASH 1')
            r_pred = vmap(self.r_net, (None, 0))(params, batch[:,0]) 
            print('CRASH 2')
            
            res_loss = jnp.mean((r_pred) ** 2)

        # Observation loss
        obs_u_pred = vmap(self.u_net, (None, 0))(params, self.obs_r)
        obs_loss = jnp.mean((self.obs_u - obs_u_pred) ** 2)
        #        n_pred = vmap(self.n_net, (None, None, 0))(params, self.t0, self.x_star)

        loss_dict = {"inner_bcs": inner_bcs_loss, "outer_bcs": outer_bcs_loss, "res": res_loss, "observ": obs_loss}
        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_diag_ntk(self, params, batch):
        #ics_ntk = vmap(ntk_fn, (None, None, 0))(
        #    self.u_net, params, self.r_star
        #)
        inner_bcs_ntk = self.u_net(params, self.r0)
        outer_bcs_ntk = self.u_net(params, self.r1)


        # Consider the effect of causal weights
        if self.config.weighting.use_causal: 
            raise NotImplementedError(f"Casual weights not supported yet for 1D Laplace!")

        else:
            res_ntk = vmap(ntk_fn, (None, None, 0))(
                self.r_net, params, batch[:, 0]
            )
        #ntk_dict = {"ics": ics_ntk, "res": res_ntk}
        ntk_dict = {"inner_bcs": inner_bcs_ntk, "outer_bcs": outer_bcs_ntk, "res": res_ntk}

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