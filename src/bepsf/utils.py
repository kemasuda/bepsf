__all__ = [
    "gaussian_sources", "gaussian_PSF", "compute_epsf", "plot_image",
    "choose_anchor", "drop_anchor", "check_solution",
    "check_mcmc_hyperparameters", "check_ePSF_fit", "check_image_fit",
    "check_anchor"
]

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import jax.numpy as jnp
import numpy as np
from jax import vmap
from scipy.signal import convolve2d
from scipy.stats import median_abs_deviation
import corner
import pandas as pd
from arviz import plot_trace

def plot_image(image, xcenters=None, ycenters=None, title=None):
    """ plot 2D image (PixelImage class) """

    if image.Z is None:
        raise ValueError("no value is set.")

    plt.figure()
    plt.imshow(image.Z,
               origin='lower',
               extent=(image.xmin, image.xmax, image.ymin, image.ymax))
    plt.xlabel("x pixel")
    plt.ylabel("y pixel")
    if xcenters is not None:
        for x, y in zip(xcenters, ycenters):
            plt.axvline(x=x, color='gray', lw=1, alpha=0.5)
            plt.axhline(y=y, color='gray', lw=1, alpha=0.5)
    if title is not None:
        plt.title(title)
    plt.colorbar()
    plt.imshow(image.mask,
               alpha=0.1,
               origin='lower',
               extent=(image.xmin, image.xmax, image.ymin, image.ymax))


def gaussian_PSF(x, y, norm, xc, yc, sigma):
    return norm * jnp.exp(-0.5 *
                          ((x - xc)**2 +
                           (y - yc)**2) / sigma**2) / (2 * jnp.pi * sigma**2)


def gaussian_sources(X, Y, norms, xcenters, ycenters, sigma, ds):
    """ put Gaussian sources on the grid X, Y """
    sources = vmap(gaussian_PSF, (None, None, 0, 0, 0, None), 0)
    return jnp.sum(sources(X, Y, norms, xcenters, ycenters, sigma) * ds,
                   axis=0)


def compute_epsf(grid, psffunc, psfkw):
    """ compute ePSF from the true PSF assuming the uniform intrapixel sensitivity

        Args:
            grid: supersampled grid for evaluating true PSF
            psffunc: function to compute true PSF

        Returns:
            ePSF evaluated on the imput grid (i.e. PSF integrated over one pixel around each point)

    """
    assert grid.dx == grid.dy
    psf_grid = psffunc(grid.X, grid.Y, **psfkw)
    Nwindow = int(1. / grid.dx) + 1
    window = np.ones((Nwindow, Nwindow))
    epsf = convolve2d(psf_grid, window / np.sum(window), mode='same')
    return epsf


def drop_anchor(p,
                idx_anchor,
                keys=['xcenters', 'ycenters', 'fluxes', 'lnfluxes']):
    """ remove the anchor source (idx_anchor) from the parameter dictionary p """
    if 'fluxes' not in p.keys():
        p['fluxes'] = jnp.exp(p['lnfluxes'])
    for key in keys:
        p[key + "_drop"] = jnp.r_[p[key][:idx_anchor], p[key][idx_anchor + 1:]]
    return p


def check_solution(image_obs,
                   xcenters,
                   ycenters,
                   fluxes,
                   p=None,
                   samples=None):
    """ utility function to check accuracy of a solution
        given as a parameter set (p) or posterior samples (samples)

        Args:
            image_obs: observed image (PixelImage object)
            xcenters, ycenters, fluxes: true (known) positions and fluxes of the sources
            p: parameter set
            samples: posterior samples

    """
    if p is None and samples is None:
        raise ValueError("Provide either a parameter dict (p) or posterior samples (samples).")

    idx_anchor = image_obs.idx_anchor

    # define truths relative to the anchor values
    xtrue = np.r_[xcenters[:idx_anchor],
                  xcenters[idx_anchor + 1:]] - xcenters[idx_anchor]
    ytrue = np.r_[ycenters[:idx_anchor],
                  ycenters[idx_anchor + 1:]] - ycenters[idx_anchor]
    ftrue = np.log10(np.r_[fluxes[:idx_anchor],
                           fluxes[idx_anchor + 1:]]) - np.log10(
                               fluxes[idx_anchor])

    x_anchor = image_obs.xinit[idx_anchor]
    y_anchor = image_obs.yinit[idx_anchor]
    f_anchor = np.log10(image_obs.finit[idx_anchor])

    if p is not None:
        x, y, f = p['xcenters_drop'], p['ycenters_drop'], np.log10(
            p['fluxes_drop'])
        xerr, yerr, ferr = 0 * x, 0 * y, 0 * f
    elif samples is not None:
        x, xerr = np.mean(samples['x'], axis=0), np.std(samples['x'], axis=0)
        y, yerr = np.mean(samples['y'], axis=0), np.std(samples['y'], axis=0)
        f, ferr = np.mean(np.log10(samples['f']),
                          axis=0), np.std(np.log10(samples['f']), axis=0)
    else:
        return None

    x -= x_anchor
    y -= y_anchor
    f -= f_anchor

    dx, dy = x - xtrue, y - ytrue
    dmax = np.r_[np.abs(dx) + np.abs(xerr), np.abs(dy) + np.abs(yerr)].max()
    plt.figure()
    plt.xlabel("$\Delta x$ (pixel)")
    plt.ylabel("$\Delta y$ (pixel)")
    plt.xlim(-dmax, dmax)
    plt.ylim(-dmax, dmax)
    plt.axvline(x=0, color='gray', zorder=-1000)
    plt.axhline(y=0, color='gray', zorder=-1000)
    plt.errorbar(dx,
                 dy,
                 xerr=xerr,
                 yerr=yerr,
                 fmt='o',
                 mfc='white',
                 lw=0.5,
                 markersize=5,
                 label="$\Delta x=%.3f\pm%.3f$\n$\Delta y=%.3f\pm%.3f$" %
                 (np.mean(dx), np.std(dx), np.mean(dy), np.std(dy)))
    plt.legend(loc='best', bbox_to_anchor=(1, 1))

    df = f - ftrue
    plt.figure(figsize=(8, 4))
    plt.xlabel("true flux (relative to anchor)")
    plt.ylabel("$\Delta\log_{10}f$")
    plt.xscale("log")
    plt.axhline(y=0., color='gray')
    plt.errorbar(10**ftrue,
                 df,
                 mfc='white',
                 fmt='o',
                 yerr=ferr,
                 lw=1.,
                 label="$\Delta \log_{10} f=%.3f\pm%.3f$" %
                 (np.mean(df), np.std(df)))
    plt.legend(loc='upper right')


def check_mcmc_hyperparameters(mcmc, pnames=['lnlenx', 'lnleny', 'lna']):
    """ trace plot and corner plot """
    samples = mcmc.get_samples()

    if 'lnmu' in samples.keys():
        pnames += ['lnmu']

    fig = plot_trace(mcmc, var_names=pnames)

    hyper = pd.DataFrame(data=dict(zip(pnames, [samples[k] for k in pnames])))
    fig = corner.corner(hyper, labels=pnames, show_titles="%.2f")


def check_ePSF_fit(grid, inferred, true):
    """ compare inferred and true ePSF """
    fig, ax = plt.subplots(1, 3, figsize=(10, 10))
    for i, (image, title) in enumerate(
            zip([inferred, true, inferred - true],
                ['inferred ePSF (mean prediction)', 'true ePSF', 'difference'
                 ])):
        im = ax[i].imshow(image,
                          origin='lower',
                          extent=[
                              grid.xgrid_edge[0], grid.xgrid_edge[-1],
                              grid.ygrid_edge[0], grid.ygrid_edge[-1]
                          ])
        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)
        ax[i].set_title(title)
    fig.tight_layout()


def check_image_fit(image_obs, image_pred):
    """ compare observed image and inferred true image """
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    im = ax[0].imshow(image_obs.Z - image_pred.reshape(*image_obs.shape),
                      origin='lower',
                      extent=[
                          image_obs.xgrid_edge[0], image_obs.xgrid_edge[-1],
                          image_obs.ygrid_edge[0], image_obs.ygrid_edge[-1]
                      ])
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)
    #ax[0].imshow(image_obs.mask, alpha=0.1)
    ax[0].set_title('data minus mean prediction')
    ax[0].set_xlim(image_obs.xgrid_edge[0], image_obs.xgrid_edge[-1])
    ax[0].set_ylim(image_obs.ygrid_edge[0], image_obs.ygrid_edge[-1])
    ax[1].hist(np.array(image_obs.Z1d - image_pred) / image_obs.Zerr1d,
               histtype='step',
               lw=1,
               bins=30,
               density=True)
    ylim = ax[1].get_ylim()
    ax[1].set_yscale("log")
    ax[1].set_xlim(-5, 5)
    ax[1].set_ylim(None, ylim[1] * 1.2)
    x0 = np.linspace(-5, 5, 100)
    ax[1].plot(x0,
               np.exp(-0.5 * x0**2) / np.sqrt(2 * np.pi),
               color='gray',
               lw=1)
    ax[1].set_xlabel("residual normalized by error")


def check_anchor(image_obs, idx_anchor=None, image_super=None):
    """ check if the chosen anchor looks fine """
    if idx_anchor is None:
        idx_anchor = image_obs.idx_anchor

    xc, yc = image_obs.xinit[idx_anchor:idx_anchor +
                             1], image_obs.yinit[idx_anchor:idx_anchor + 1]
    plot_image(image_obs,
               title='observed image (anchor idx: %d)' % idx_anchor,
               xcenters=xc,
               ycenters=yc)

    if image_super is not None:
        plot_image(image_super,
                   title='supersampled image (anchor idx: %d)' % idx_anchor,
                   xcenters=xc,
                   ycenters=yc)


def choose_anchor(image_obs,
                  xcenters,
                  ycenters,
                  lnfluxes=None,
                  mad_threshold=1.,
                  plot=False):
    """ choose the anchor in a simulated image """

    dx, dy = image_obs.xinit - xcenters, image_obs.yinit - ycenters
    mad_dx = median_abs_deviation(dx)
    mad_dy = median_abs_deviation(dy)

    for i in range(10):
        idx_isolated = (np.abs(dx) < mad_dx * mad_threshold) & (
            np.abs(dy) < mad_dy * mad_threshold)
        if lnfluxes is not None:
            dlnf = image_obs.lnfinit - lnfluxes
            mad_dlnf = median_abs_deviation(dlnf)
            idx_isolated &= np.abs(dlnf) < mad_dlnf * mad_threshold
        if np.sum(idx_isolated):
            break
        else:
            mad_threshold *= 1.5

    idx_lnf_sorted = np.argsort(image_obs.lnfinit)[::-1]
    idx_isolated_sorted = idx_isolated[idx_lnf_sorted]
    idx_anchor = idx_lnf_sorted[idx_isolated_sorted][0]

    if plot:
        plt.figure()
        plt.xlabel("$\Delta x$")
        plt.ylabel("$\Delta y$")
        plt.plot(dx, dy, '.', color='gray', alpha=0.2)
        plt.plot(dx[idx_isolated], dy[idx_isolated], '.', color='gray')
        plt.plot(dx[idx_anchor], dy[idx_anchor], '.', color='salmon')

        if lnfluxes is not None:
            plt.figure()
            plt.xlabel("$\ln f$")
            plt.ylabel("$\Delta \ln f$")
            plt.plot(lnfluxes, dlnf, '.', color='gray', alpha=0.2)
            plt.plot(lnfluxes[idx_isolated],
                     dlnf[idx_isolated],
                     '.',
                     color='gray')
            plt.plot(lnfluxes[idx_anchor],
                     dlnf[idx_anchor],
                     '.',
                     color='salmon')

    return idx_anchor
