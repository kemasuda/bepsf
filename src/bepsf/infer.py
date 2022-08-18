__all__ = ["optimize", "numpyro_model", "run_hmc"]

import numpy as np
import jax.numpy as jnp
from jax import random
import jaxopt
import numpyro
import numpyro.distributions as dist
from numpyro.infer import init_to_value

def parameter_bounds(p_init, widths):
    """ return lower and upper bounds for parameters """
    p_low, p_high = {}, {}
    for key,val in p_init.items():
        p_low[key] = val + widths[key][0]
        p_high[key] = val + widths[key][1]
    return (p_low, p_high)

def optimize(gridpsf, image_obs,
             method="TNC", n_iter=1,
             lnfluxlim=[-10.,2.], xyclim=[-2.,2.], lnlenxlim=[-0.5,1.5], lnlenylim=[-0.5,1.5],
             lnamplim=[-3.,3.], lnmulim=[-3.,3.], fix_mean=True):

    """ optimize the source positions, fluxes, and the ePSF hyperparameters

        Args:
            gridpsf: ePSF model (GridePSFModel class)
            image_obs: observed image (PixelImage class)
            method: method for optimization in jaxopt.ScipyBoundedMinimize
            n_iter: number of iterations (not sure if >1 helps)
            lnfluxlim, xyclim, lnlenxlim, lnlenylim, lnamplim, lnmulim:
                widths for log flux, x/y source positions, log x/y scale lengths for ePSF,
                ePSF covariance amplitude, ePSF mean
            fix_mean: if true, mean of the ePSF is fixed to zero

    """


    lnfluxes_guess, xcenters_guess, ycenters_guess, idx_anchor \
        = image_obs.lnfinit, image_obs.xinit, image_obs.yinit, image_obs.idx_anchor
    mask1d = image_obs.mask1d

    p_init = {
        "lnfluxes": lnfluxes_guess,
        "xcenters": xcenters_guess,
        "ycenters": ycenters_guess,
        "lnlenx": np.float64(0.5), # np.float64() required so that pytree leaf has "shape"
        "lnleny": np.float64(0.5),
        "lnamp": np.float64(-np.log(gridpsf.pixarea)),
    }

    if fix_mean:
        p_init["lnmu"] = np.float64(-1e10)
        lnmulim = [0,0]
    else:
        p_init["lnmu"] = np.float64(-np.log(gridpsf.pixarea)-np.log(2))

    def objective(p):
        fluxes = jnp.exp(jnp.r_[p['lnfluxes'][:idx_anchor], lnfluxes_guess[idx_anchor], p['lnfluxes'][idx_anchor+1:]])
        xcenters = jnp.r_[p['xcenters'][:idx_anchor], xcenters_guess[idx_anchor], p['xcenters'][idx_anchor+1:]]
        ycenters = jnp.r_[p['ycenters'][:idx_anchor], ycenters_guess[idx_anchor], p['ycenters'][idx_anchor+1:]]
        lenx, leny, amp2, mu = jnp.exp(p['lnlenx']), jnp.exp(p['lnleny']), jnp.exp(2*p['lnamp']), jnp.exp(p['lnmu'])
        return -gridpsf.log_likelihood(fluxes, xcenters, ycenters, lenx, leny, amp2, mu,
                                       image_obs.X1d[~mask1d], image_obs.Y1d[~mask1d], image_obs.Z1d[~mask1d],
                                       image_obs.Zerr1d[~mask1d]
                                      )

    solver = jaxopt.ScipyBoundedMinimize(fun=objective, method=method)

    # optimize flux and position alone first
    widths = {"lnfluxes": lnfluxlim, "xcenters": xyclim, "ycenters": xyclim,
              "lnlenx": [0,0], "lnleny": [0,0], "lnamp": [0,0], "lnmu": [0,0]}
    bounds = parameter_bounds(p_init, widths)
    print ("# optimizing flux and position...")
    for i in range(n_iter):
        res = solver.run(p_init, bounds=bounds)
        p_init, state = res
        print (state)
        print ()

    # optimize ePSF hyperparameters
    widths = {"lnfluxes": [0,0], "xcenters": [0,0], "ycenters": [0,0],
              "lnlenx": lnlenxlim, "lnleny": lnlenylim, "lnamp": lnamplim, "lnmu": lnmulim}
    bounds = parameter_bounds(p_init, widths)
    print ("# optimizing GP parameters...")
    for i in range(n_iter):
        res = solver.run(p_init, bounds=bounds)
        p_init, state = res
        print (state)
        print ()

    # optimize all parameters simultaneously
    widths = {"lnfluxes": lnfluxlim, "xcenters": xyclim, "ycenters": xyclim,
              "lnlenx": lnlenxlim, "lnleny": lnlenylim, "lnamp": lnamplim, "lnmu": lnmulim}
    bounds = parameter_bounds(p_init, widths)
    print ("# optimizing all parameters...")
    for i in range(n_iter):
        res = solver.run(p_init, bounds=bounds)
        p_init, state = res
        print (state)

    return res

def numpyro_model(gridpsf, image_obs, popt, fix_mean=True):
    """ numpyro model for HMC

        Args:
            gridpsf: ePSF model (GridePSFModel class)
            image_obs: observed image (PixelImage class)
            popt: initial guess for the parameters (dictionary)
            fix_mean: if true, ePSF mean is fixed to zero

    """

    # sample hyperparameters
    lnlenx = numpyro.sample("lnlenx", dist.Uniform(low=0*(popt['lnlenx']-2), high=popt['lnlenx']+2))
    lnleny = numpyro.sample("lnleny", dist.Uniform(low=0*(popt['lnleny']-2), high=popt['lnleny']+2))
    lna = numpyro.sample("lna", dist.Uniform(low=popt['lnamp']-5, high=popt['lnamp']+2))
    lenx, leny, amp2 = jnp.exp(lnlenx), jnp.exp(lnleny), jnp.exp(2*lna)
    if fix_mean:
        mu = 0.
    else:
        lnmu = numpyro.sample("lnmu", dist.Uniform(low=popt['lnmu']-5, high=popt['lnmu']+2))
        mu = jnp.exp(lnmu)

    # omit anchored source (otherwise the sampling is slow)
    f = numpyro.sample("f", dist.Uniform(low=0*popt['lnfluxes_drop'], high=2*jnp.exp(popt['lnfluxes_drop'])))
    x = numpyro.sample("x", dist.Uniform(low=popt['xcenters_drop']-1., high=popt['xcenters_drop']+1))
    y = numpyro.sample("y", dist.Uniform(low=popt['ycenters_drop']-1., high=popt['ycenters_drop']+1))
    idx_anchor = image_obs.idx_anchor
    fluxes = jnp.r_[f[:idx_anchor], jnp.exp(popt['lnfluxes'][idx_anchor]), f[idx_anchor:]]
    xcenters = jnp.r_[x[:idx_anchor], popt['xcenters'][idx_anchor], x[idx_anchor:]]
    ycenters = jnp.r_[y[:idx_anchor], popt['ycenters'][idx_anchor], y[idx_anchor:]]

    # masked pixels are ignored in likelihood evaluation
    mask1d = image_obs.mask1d
    gploglike = gridpsf.log_likelihood(fluxes, xcenters, ycenters, lenx, leny, amp2, mu,
                                       image_obs.X1d[~mask1d], image_obs.Y1d[~mask1d], image_obs.Z1d[~mask1d],
                                       image_obs.Zerr1d[~mask1d])

    numpyro.factor("gploglike", gploglike)

def run_hmc(gridpsf, image_obs, popt, nw=500, ns=500, target_accept_prob=0.90, **kwargs):
    """ run HMC """
    init_strategy = init_to_value(values=popt)
    kernel = numpyro.infer.NUTS(numpyro_model, target_accept_prob=target_accept_prob, init_strategy=init_strategy)
    mcmc = numpyro.infer.MCMC(kernel, num_warmup=nw, num_samples=ns)
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, gridpsf, image_obs, popt, **kwargs)
    mcmc.print_summary()
    return mcmc
