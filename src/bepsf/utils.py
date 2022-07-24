__all__ = ["simulate_gaussian_sources", "plot_image"]

import matplotlib.pyplot as plt
import jax.numpy as jnp
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

#def supersampling_matrix(image_obs, image_super):
