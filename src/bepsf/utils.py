__all__ = ["gaussian_sources", "gaussian_PSF", "compute_epsf", "plot_image", "drop_anchor", "check_solution", "check_ePSF_fit", "check_image_fit", "check_anchor"]

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import jax.numpy as jnp
import numpy as np
from jax import vmap
from scipy.signal import convolve2d

def plot_image(image, xcenters=None, ycenters=None, title=None):
    if image.Z is None:
        print ("no value is set.")
        return None

    plt.figure()
    plt.imshow(image.Z, origin='lower', extent=(image.xmin, image.xmax, image.ymin, image.ymax))
    plt.xlabel("x pixel")
    plt.ylabel("y pixel")
    if xcenters is not None:
        for x, y in zip(xcenters, ycenters):
            plt.axvline(x=x, color='gray', lw=1, alpha=0.5)
            plt.axhline(y=y, color='gray', lw=1, alpha=0.5)
    if title is not None:
        plt.title(title)
    plt.colorbar()
    plt.imshow(image.mask, alpha=0.1, origin='lower', extent=(image.xmin, image.xmax, image.ymin, image.ymax));

def gaussian_PSF(x, y, norm, xc, yc, sigma):
    #x, y = image.X, image.Y
    return norm * jnp.exp(-0.5* ((x-xc)**2 + (y-yc)**2)/sigma**2) / (2*jnp.pi*sigma**2)

def gaussian_sources(X, Y, norms, xcenters, ycenters, sigma, ds):
    #sources = vmap(gaussian_source, (None,0,0,0,None), 0)
    #return jnp.sum(sources(image, norms, xcenters, ycenters, sigma)*image.ds, axis=0)
    sources = vmap(gaussian_PSF, (None,None,0,0,0,None), 0)
    return jnp.sum(sources(X, Y, norms, xcenters, ycenters, sigma)*ds, axis=0)

def compute_epsf(grid, psffunc, psfkw):
    assert grid.dx == grid.dy
    psf_grid = psffunc(grid.X, grid.Y, **psfkw)
    Nwindow = int(1./grid.dx)+1
    window = np.ones((Nwindow, Nwindow))
    epsf = convolve2d(psf_grid, window/np.sum(window), mode='same')
    return epsf

def drop_anchor(p, idx_anchor, keys=['xcenters', 'ycenters', 'fluxes', 'lnfluxes']):
    if 'fluxes' not in p.keys():
        p['fluxes'] = jnp.exp(p['lnfluxes'])
    for key in keys:
        p[key+"_drop"] = jnp.r_[p[key][:idx_anchor], p[key][idx_anchor+1:]]
    return p

def check_solution(image_obs, xcenters, ycenters, fluxes, p=None, samples=None):
    idx_anchor = image_obs.idx_anchor
    xtrue = np.r_[xcenters[:idx_anchor], xcenters[idx_anchor+1:]]
    ytrue = np.r_[ycenters[:idx_anchor], ycenters[idx_anchor+1:]]
    ftrue = np.r_[fluxes[:idx_anchor], fluxes[idx_anchor+1:]] / fluxes[idx_anchor]
    
    if p is not None:
        x, y, f = p['xcenters_drop'], p['ycenters_drop'], p['fluxes_drop']/image_obs.finit[idx_anchor] 
        xerr, yerr, ferr = 0 * x, 0 * y, 0 * f
    elif samples is not None:
        x, xerr = np.mean(samples['x'], axis=0), np.std(samples['x'], axis=0)
        y, yerr = np.mean(samples['y'], axis=0), np.std(samples['y'], axis=0)
        f, ferr = np.mean(samples['f'], axis=0), np.std(samples['f'], axis=0)
        f_anchor = image_obs.finit[idx_anchor]
        f /= f_anchor
        ferr /= f_anchor
    else:
        return None

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

    df = f - ftrue
    plt.figure(figsize=(8,4))
    plt.xlabel("true flux (relative to anchor)")
    plt.ylabel("measured flux $-$ true flux")
    plt.xscale("log")
    plt.axhline(y=0., color='gray')
    plt.errorbar(ftrue, df, mfc='white', fmt='o', yerr=ferr, lw=1.);

def check_ePSF_fit(grid, inferred, true):
    fig, ax = plt.subplots(1,3,figsize=(10,10))
    for i,(image,title) in enumerate(zip([inferred, true, inferred-true], ['inferred ePSF (mean prediction)', 'true ePSF', 'difference'])):
        im = ax[i].imshow(image, origin='lower', 
                          extent=[grid.xgrid_edge[0], grid.xgrid_edge[-1], grid.ygrid_edge[0], grid.ygrid_edge[-1]])
        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)
        ax[i].set_title(title)
    fig.tight_layout(); 

    
def check_image_fit(image_obs, image_pred):
    fig, ax = plt.subplots(1,2,figsize=(12,4))
    im = ax[0].imshow(image_obs.Z-image_pred.reshape(*image_obs.shape), origin='lower', 
                      extent=[image_obs.xgrid_edge[0], image_obs.xgrid_edge[-1], image_obs.ygrid_edge[0], 
                              image_obs.ygrid_edge[-1]])
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)
    ax[0].imshow(image_obs.mask, alpha=0.1)
    ax[0].set_title('data minus mean prediction')
    ax[0].set_xlim(image_obs.xgrid_edge[0], image_obs.xgrid_edge[-1])
    ax[0].set_ylim(image_obs.ygrid_edge[0], image_obs.ygrid_edge[-1])
    ax[1].hist(np.array(image_obs.Z1d-image_pred)/image_obs.Zerr1d, histtype='step', lw=1, bins=30, density=True)
    ylim = ax[1].get_ylim()
    ax[1].set_yscale("log")
    ax[1].set_xlim(-5,5)
    ax[1].set_ylim(None, ylim[1]*1.2)
    x0 = np.linspace(-5, 5, 100)
    ax[1].plot(x0, np.exp(-0.5*x0**2)/np.sqrt(2*np.pi), color='gray', lw=1)
    ax[1].set_xlabel("residual normalized by error");
    
def check_anchor(image_obs, idx_anchor=None, image_super=None):
    if idx_anchor is None:
        idx_anchor = image_obs.idx_anchor
        
    xc, yc = image_obs.xinit[idx_anchor:idx_anchor+1], image_obs.yinit[idx_anchor:idx_anchor+1]
    plot_image(image_obs, title='observed image (anchor idx: %d)'%idx_anchor, 
               xcenters=xc, ycenters=yc)
    
    if image_super is not None:
         plot_image(image_super,  title='supersampled image (anchor idx: %d)'%idx_anchor, 
                    xcenters=xc, ycenters=yc)
            
"""
def simulate_gaussian_sources(image, norms, xcenters, ycenters, sigma):
    #sources = vmap(gaussian_source, (None,0,0,0,None), 0)
    #return jnp.sum(sources(image, norms, xcenters, ycenters, sigma)*image.ds, axis=0)
    sources = vmap(gaussian_source, (None,None,0,0,0,None), 0)
    return jnp.sum(sources(image.X, image.Y, norms, xcenters, ycenters, sigma)*image.ds, axis=0)
"""