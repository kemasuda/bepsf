_all__ = ["PixelImage", "super_to_obs"]

import jax.numpy as jnp
import numpy as np
from jax import vmap

class PixelImage:
    def __init__(self, xmax, ymax, xmin=0., ymin=0., dx=1., dy=1.):
        """initialization
        Args:
           xmin, xmax, ymin, ymax: x/y coordinates at the grid edge
           dx, dy: grid spacing
        """
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.Nx = int((xmax - xmin) / dx)
        self.Ny = int((ymax - ymin) / dy)
        self.shape = (self.Nx, self.Ny)
        self.size = self.Nx * self.Ny
        self.ds = dx * dy

        self.xgrid_edge = jnp.linspace(xmin, xmax, self.Nx+1)
        self.ygrid_edge = jnp.linspace(ymin, ymax, self.Ny+1)
        self.xgrid_center = 0.5 * (self.xgrid_edge[1:] + self.xgrid_edge[:-1])
        self.ygrid_center = 0.5 * (self.ygrid_edge[1:] + self.ygrid_edge[:-1])

        X, Y = jnp.meshgrid(self.xgrid_center, self.ygrid_center)
        self.X = X
        self.Y = Y
        self.X1d = jnp.tile(self.xgrid_center, self.Ny)
        self.Y1d = jnp.repeat(self.ygrid_center, self.Nx)
        self.Z = -1 * jnp.ones_like(X)
        self.mask = np.array(X)**2 < 0.
        self.mask1d = self.mask.ravel()

    @property
    def Z1d(self):
        return self.Z.ravel()

    def aperture_flux(self, xc, yc, radius):
        return np.sqrt((self.X-xc)**2+(self.Y-yc)**2) < radius
    
    def define_mask(self, xcenters, ycenters, limit_dist):
        distances = np.sqrt((self.X1d[None,:]-xcenters[:,None])**2 + (self.Y1d[None,:]-ycenters[:,None])**2) # Nsource, Npix
        minimum_dist = np.min(distances, axis=0)
        mask1d = minimum_dist > limit_dist
        mask = mask1d.reshape(*self.shape)
        self.mask = mask
        self.mask1d = mask1d
        
def super_to_obs(Z_super, Z_obs):
    # convert 2d supersampled image Z_super (Ms,Ns) to 2d undersampled image Z_obs (Mobs,Nobs)
    Ms, Ns = Z_super.shape
    Mobs, Nobs = Z_obs.shape
    K, L = Ms // Mobs, Ns // Nobs
    return Z_super[:Mobs*K, :Nobs*L].reshape(Mobs, K, Nobs, L).sum(axis=(1, 3))

"""
def supersampling_matrix(image_super, image_obs):
    # matrix to convert 1d supersampled image (Ms*Ns,) to 1d understampled image (Mobs*Nobs,)
    Ms, Ns = image_super.shape
    super_to_obs1d = lambda Zsuper1d: super_to_obs(Zsuper1d.reshape(Ms,Ns), image_obs).ravel()
    super_to_obs1d_vmap = vmap(super_to_obs1d, (0), 1)
    S = super_to_obs1d_vmap(np.eye(Ms*Ns))
    return S
"""
