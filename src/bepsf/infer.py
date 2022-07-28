__all__ = ["optimize_flux_and_position", "numpyro_model", "run_hmc"]

import numpy as np
import jax.numpy as jnp
from jax import random
import jaxopt
import numpyro
import numpyro.distributions as dist
from numpyro.infer import init_to_value

def optimize_flux_and_position(gridpsf, image_obs, image_err,
                               lnfluxes_guess, xcenters_guess, ycenters_guess, idx_anchor, method="TNC", n_iter=1, radius=3.,
                               lnfluxlim=[-10.,2.], xyclim=[-2.,2.], lnlenxlim=[0,0], lnlenylim=[0,0], lnamplim=[-3.,3.]):

    #Zanchor_mean = np.mean(image_obs.Z[image_obs.aperture_flux(xcenters_guess[idx_anchor], ycenters_guess[idx_anchor], 3.)])
    
    #flux_ap = image_obs.Z[image_obs.aperture_flux(xcenters_guess[idx_anchor], ycenters_guess[idx_anchor], radius)]
    #dfmedian = np.abs(np.median(np.diff(flux_ap)))
    #lna_guess = np.log(dfmedian)
    
    Npix = (gridpsf.xgrid_edge.max()-gridpsf.xgrid_edge.min())*(gridpsf.ygrid_edge.max()-gridpsf.ygrid_edge.min())
    lna_guess = -np.log(Npix)
    print ("lna_guess:", lna_guess)
    
    p_init = {
        "lnfluxes": lnfluxes_guess,
        "xcenters": xcenters_guess,
        "ycenters": ycenters_guess,
        "lnlenx": np.float64(0.5), # np.float64() required so that pytree leaf has "shape"
        "lnleny": np.float64(0.5),
        "lnamp": np.float64(lna_guess)
    }
       
    p_low, p_high = {}, {}
    widths = {"lnfluxes": lnfluxlim, "xcenters": xyclim, "ycenters": xyclim, 
              "lnlenx": lnlenxlim, "lnleny": lnlenylim, "lnamp": lnamplim}
    for key,val in p_init.items():
        p_low[key] = val + widths[key][0]
        p_high[key] = val + widths[key][1]
       
    mask1d = image_obs.mask1d
    def objective(p):
        fluxes = jnp.exp(jnp.r_[p['lnfluxes'][:idx_anchor], lnfluxes_guess[idx_anchor], p['lnfluxes'][idx_anchor+1:]])
        xcenters = jnp.r_[p['xcenters'][:idx_anchor], xcenters_guess[idx_anchor], p['xcenters'][idx_anchor+1:]]
        ycenters = jnp.r_[p['ycenters'][:idx_anchor], ycenters_guess[idx_anchor], p['ycenters'][idx_anchor+1:]]
        lenx, leny, amp2 = jnp.exp(p['lnlenx']), jnp.exp(p['lnleny']), jnp.exp(2*p['lnamp'])
        #return -gridpsf.gp_marginal(fluxes, xcenters, ycenters, lenx, leny, amp2, image_obs, image_err)
        return -gridpsf.log_likelihood(fluxes, xcenters, ycenters, lenx, leny, amp2, 
                                       image_obs.X1d[~mask1d], image_obs.Y1d[~mask1d], image_obs.Z1d[~mask1d], 
                                       image_err.ravel()[~mask1d])
        
    solver = jaxopt.ScipyBoundedMinimize(fun=objective, method=method)

    for i in range(n_iter):
        res = solver.run(p_init, bounds=(p_low, p_high))
        p_init, state = res
        print (state)
        print ()
        
    return res#, S

def numpyro_model(gridpsf, image_obs, image_err, idx_anchor, popt):
    lnlenx = numpyro.sample("lnlenx", dist.Uniform(low=0*(popt['lnlenx']-2), high=popt['lnlenx']+2)) 
    lnleny = numpyro.sample("lnleny", dist.Uniform(low=0*(popt['lnleny']-2), high=popt['lnleny']+2))
    lna = numpyro.sample("lna", dist.Uniform(low=popt['lnamp']-5, high=popt['lnamp']+2))
    lenx, leny, amp2 = jnp.exp(lnlenx), jnp.exp(lnleny), jnp.exp(2*lna)
    
    # omit anchored source (otherwise the sampling is slow)
    f = numpyro.sample("f", dist.Uniform(low=0*popt['lnfluxes_drop'], high=2*jnp.exp(popt['lnfluxes_drop'])))
    x = numpyro.sample("x", dist.Uniform(low=popt['xcenters_drop']-1., high=popt['xcenters_drop']+1))
    y = numpyro.sample("y", dist.Uniform(low=popt['ycenters_drop']-1., high=popt['ycenters_drop']+1)) 
    fluxes = jnp.r_[f[:idx_anchor], jnp.exp(popt['lnfluxes'][idx_anchor]), f[idx_anchor:]]
    xcenters = jnp.r_[x[:idx_anchor], popt['xcenters'][idx_anchor], x[idx_anchor:]]
    ycenters = jnp.r_[y[:idx_anchor], popt['ycenters'][idx_anchor], y[idx_anchor:]]

    #gploglike = gridpsf.gp_marginal(fluxes, xcenters, ycenters, lenx, leny, amp2, image_obs, image_err)
    mask1d = image_obs.mask1d
    gploglike = gridpsf.log_likelihood(fluxes, xcenters, ycenters, lenx, leny, amp2, 
                                       image_obs.X1d[~mask1d], image_obs.Y1d[~mask1d], image_obs.Z1d[~mask1d], 
                                       image_err.ravel()[~mask1d])

    numpyro.factor("gploglike", gploglike)

def run_hmc(gridpsf, image_obs, image_err, idx_anchor, popt, nw=500, ns=500, target_accept_prob=0.90):
    init_strategy = init_to_value(values=popt)
    kernel = numpyro.infer.NUTS(numpyro_model, target_accept_prob=target_accept_prob, init_strategy=init_strategy)
    mcmc = numpyro.infer.MCMC(kernel, num_warmup=nw, num_samples=ns)
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, gridpsf, image_obs, image_err, idx_anchor, popt)
    mcmc.print_summary()
    return mcmc
