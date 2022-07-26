__all__ = ["simulate_gaussian_sources", "plot_image", "drop_anchor", "check_solution", "check_PSF_fit", "check_image_fit"]

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import jax.numpy as jnp
import numpy as np
from jax import vmap

def plot_image(image, xcenters=None, ycenters=None, title=None):
    if image.Z is None:
        print ("no value is set.")
        return None

    plt.imshow(image.Z, origin='lower', extent=(image.xmin, image.xmax, image.ymin, image.ymax))
    plt.xlabel("x pixel")
    plt.ylabel("y pixel")
    if xcenters is not None:
        for x, y in zip(xcenters, ycenters):
            plt.axvline(x=x, color='gray', lw=1, alpha=0.5)
            plt.axhline(y=y, color='gray', lw=1, alpha=0.5)
    if title is not None:
        plt.title(title)
    plt.colorbar();

def gaussian_source(image, norm, xc, yc, sigma):
    x, y = image.X, image.Y
    return norm * jnp.exp(-0.5* ((x-xc)**2 + (y-yc)**2)/sigma**2) / (2*jnp.pi*sigma**2)

def simulate_gaussian_sources(image, norms, xcenters, ycenters, sigma):
    sources = vmap(gaussian_source, (None,0,0,0,None), 0)
    return jnp.sum(sources(image, norms, xcenters, ycenters, sigma)*image.ds, axis=0)

def drop_anchor(p, idx_anchor, keys=['xcenters', 'ycenters', 'fluxes', 'lnfluxes']):
    for key in keys:
        p[key+"_drop"] = jnp.r_[p[key][:idx_anchor], p[key][idx_anchor+1:]]
    return p

def check_solution(x, y, f, xtrue, ytrue, ftrue, xerr=None, yerr=None, ferr=None):
    if xerr is None:
        xerr = 0 * x
    if yerr is None:
        yerr = 0 * y
    if ferr is None:
        ferr = 0 * f
        
    dx, dy = x - xtrue, y - ytrue
    dmax = np.r_[np.abs(dx)+np.abs(xerr), np.abs(dy)+np.abs(yerr)].max()
    plt.figure()
    plt.xlabel("$\Delta x$ (pixel)")
    plt.ylabel("$\Delta y$ (pixel)")
    plt.xlim(-dmax, dmax)
    plt.ylim(-dmax, dmax)
    plt.axvline(x=0, color='gray', zorder=-1000)
    plt.axhline(y=0, color='gray', zorder=-1000)
    plt.errorbar(dx, dy, xerr=xerr, yerr=yerr, fmt='o', mfc='white', lw=0.5, markersize=5,
                label="$\Delta x=%.3f\pm%.3f$\n$\Delta y=%.3f\pm%.3f$"%(np.mean(dx), np.std(dx), np.mean(dy), np.std(dy)))
    plt.legend(loc='best', bbox_to_anchor=(1,1));

    f, ftrue = f, ftrue
    df = f - ftrue
    plt.figure(figsize=(8,4))
    plt.xlabel("true flux (relative to anchor)")
    plt.ylabel("measured flux $-$ true flux")
    plt.xscale("log")
    plt.axhline(y=0., color='gray')
    plt.errorbar(ftrue, df, mfc='white', fmt='o', yerr=ferr, lw=1.);

def check_PSF_fit(gridpsf, meanpsf, truepsf):
    fig, ax = plt.subplots(1,3,figsize=(10,10))
    for i,(image,title) in enumerate(zip([meanpsf, truepsf, meanpsf-truepsf], ['inferred PSF (mean prediction)', 'true PSF', 'difference'])):
        im = ax[i].imshow(image, origin='lower', 
                          extent=[gridpsf.xgrid_edge[0], gridpsf.xgrid_edge[-1], gridpsf.ygrid_edge[0], gridpsf.ygrid_edge[-1]])
        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)
        ax[i].set_title(title)
    fig.tight_layout();
    
def check_image_fit(image_obs, image_err, image_pred):
    fig, ax = plt.subplots(1,2,figsize=(12,4))
    im = ax[0].imshow(image_obs.Z-image_pred.reshape(*image_obs.shape), origin='lower', 
                      extent=[image_obs.xgrid_edge[0], image_obs.xgrid_edge[-1], image_obs.ygrid_edge[0], 
                              image_obs.ygrid_edge[-1]])
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)
    ax[0].imshow(image_obs.mask, alpha=0.1)
    ax[0].set_title('data minus mean prediction')
    ax[1].hist(np.array(image_obs.Z1d-image_pred)/image_err.ravel(), histtype='step', lw=1, bins=30, density=True)
    ylim = ax[1].get_ylim()
    ax[1].set_yscale("log")
    ax[1].set_xlim(-5,5)
    ax[1].set_ylim(ylim)
    x0 = np.linspace(-5, 5, 100)
    ax[1].plot(x0, np.exp(-0.5*x0**2)/np.sqrt(2*np.pi), color='gray', lw=1)
    ax[1].set_xlabel("residual normalized by error");