_all__ = ["PixelImage", "super_to_obs"]

import jax.numpy as jnp
import numpy as np
from jax import vmap


class PixelImage:
    """ class for 2D images """
    def __init__(self, xmax, ymax, xmin=0., ymin=0., dx=1., dy=1.):
        """ initialization

        Args:
           xmin, xmax, ymin, ymax: x/y coordinates at the grid edges
           dx, dy: grid spacing (normally 1, just count the pixel number)

        """
        self.dx, self.Nx, self.xmin, self.xmax, self.xgrid_edge, self.xgrid_center = self.set_grid(
            dx, xmin, xmax)
        self.dy, self.Ny, self.ymin, self.ymax, self.ygrid_edge, self.ygrid_center = self.set_grid(
            dy, ymin, ymax)
        self.set_meshgrid()

    def set_grid(self, dq, qmin, qmax):
        """set 1-D grid

        Args:
            dq (float): grid width for q-direction (q = x or y)
            qmin (float): minimum of q
            qmax (float): maximum of q

        Returns:
            grid width, number of the grid, minimum, maximum, grid edge, grid center
        """
        Nq = int((qmax - qmin) / dq)
        qgrid_edge = jnp.linspace(qmin, qmax, Nq + 1)
        qgrid_center = 0.5 * (qgrid_edge[1:] + qgrid_edge[:-1])
        return dq, Nq, qmin, qmax, qgrid_edge, qgrid_center

    def set_meshgrid(self):
        """set 2D mesh grid
        """
        self.shape = (self.Nx, self.Ny)
        self.size = self.Nx * self.Ny
        self.ds = self.dx * self.dy
        self.X, self.Y = jnp.meshgrid(self.xgrid_center, self.ygrid_center)
        self.X1d = jnp.tile(self.xgrid_center, self.Ny)
        self.Y1d = jnp.repeat(self.ygrid_center, self.Nx)
        self.Z = None  #-1 * jnp.ones_like(X) # in case Z.shape is needed
        self.Zerr = None
        self.mask = np.array(self.X)**2 < 0.
        self.mask1d = self.mask.ravel()
        self.xinit = None
        self.yinit = None
        self.lnfinit = None

    @property
    def Z1d(self):
        return self.Z.ravel()

    @property
    def Zerr1d(self):
        return self.Zerr.ravel()

    @property
    def finit(self):
        return np.exp(self.lnfinit)

    def circular_aperture_index(self, xc, yc, radius):
        """ 2D index for a circular aperture around (xc, yc) with a given radius"""
        return jnp.sqrt((self.X - xc)**2 + (self.Y - yc)**2) < radius

    def aperture_photometry(self, xcenters, ycenters, radius):
        """ performs aperture photometry for multiple sources """
        def single(x, y, radius):
            idx_ap = self.circular_aperture_index(x, y, radius)
            flux_ap = jnp.where(idx_ap, self.Z, self.Z * 0.)
            return jnp.sum(flux_ap), jnp.average(
                self.X, weights=flux_ap), jnp.average(self.Y, weights=flux_ap)

        func = vmap(single, (0, 0, None), 0)
        return func(xcenters, ycenters, radius)

    def define_mask(self, xcenters, ycenters, limit_dist):
        """ define source mask (True means masked)

        Args:
            xcenters, ycenters: x/y coordinates of the sources
            limit_dist: if the distance to the nearest sources is >limit_dist, the pixel is masked

        """
        # distance matrix (# of sources) x (# of pixels)
        distances = np.sqrt((self.X1d[None, :] - xcenters[:, None])**2 +
                            (self.Y1d[None, :] - ycenters[:, None])**2)
        minimum_dist = np.min(distances, axis=0)
        mask1d = minimum_dist > limit_dist
        mask = mask1d.reshape(*self.shape)
        self.mask = mask
        self.mask1d = mask1d


def super_to_obs(Z_super, Z_obs):
    """ convert 2d supersampled image to undersampled image by summing flux

    Args:
        Z_super: supersampled image of size (Ms, Ns)
        Z_obs: undersampled image of size (Mobs, Nobs)

    Returns:
        Z_super understampled to (Mobs, Nobs) grid

    """

    Ms, Ns = Z_super.shape
    Mobs, Nobs = Z_obs.shape
    K, L = Ms // Mobs, Ns // Nobs
    return Z_super[:Mobs * K, :Nobs * L].reshape(Mobs, K, Nobs,
                                                 L).sum(axis=(1, 3))
