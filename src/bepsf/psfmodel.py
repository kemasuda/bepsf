__all__ = ["GridePSFModel"]

import jax.numpy as jnp
import numpy as np
from jax.scipy.ndimage import map_coordinates
from jax import jit
from functools import partial
from jax import vmap
import numpyro.distributions as dist


class GridePSFModel:
    def __init__(self, x_extent, y_extent, dx, dy):
        """ define ePSF via grid interpolation

        Args:
           x_extent, y_extent: size of the ePSF model grid (in units of observed pixels)
           dx, dy: grid spacing (in units of observed pixels)

        """
        Nx = int(x_extent / dx // 2) * 2 + 1
        Ny = int(y_extent / dy // 2) * 2 + 1
        xgrid_edge = jnp.linspace(-0.5 * dx, (Nx - 0.5) * dx, Nx + 1)
        ygrid_edge = jnp.linspace(-0.5 * dy, (Ny - 0.5) * dy, Ny + 1)
        xgrid_center = 0.5 * (xgrid_edge[1:] + xgrid_edge[:-1])
        ygrid_center = 0.5 * (ygrid_edge[1:] + ygrid_edge[:-1])
        xm, ym = jnp.median(xgrid_center), jnp.median(ygrid_center)

        self.xmin, self.xmax = xgrid_edge.min(), xgrid_edge.max()
        self.ymin, self.ymax = ygrid_edge.min(), ygrid_edge.max()
        self.Nx = Nx
        self.Ny = Ny
        self.xgrid_edge = xgrid_edge - xm
        self.ygrid_edge = ygrid_edge - ym
        self.xgrid_center = xgrid_center - xm
        self.ygrid_center = ygrid_center - ym
        X, Y = jnp.meshgrid(self.xgrid_center, self.ygrid_center)
        self.X = X
        self.Y = Y
        self.X1d = jnp.tile(self.xgrid_center, self.Ny)
        self.Y1d = jnp.repeat(self.ygrid_center, self.Nx)

        self.dx = dx
        self.dy = dy
        self.ds = dx * dy
        self.Nx = Nx
        self.Ny = Ny
        self.shape = (self.Nx, self.Ny)
        self.size = self.Nx * self.Ny
        self.eye = jnp.eye(self.size)
        self.pixarea = (self.xgrid_edge.max() - self.xgrid_edge.min()) * (
            self.ygrid_edge.max() - self.ygrid_edge.min())

        print("PSF grid shape:", self.shape)
        print("grid edge: x=[%f, %f], y=[%f, %f]" %
              (self.xgrid_edge[0], self.xgrid_edge[-1], self.ygrid_edge[0],
               self.ygrid_edge[-1]))
        print("grid center: x=%f, y=%f" %
              (np.median(self.xgrid_center), np.median(self.ygrid_center)))

    @partial(jit, static_argnums=(0, ))
    def evaluate_ePSF(self, X, Y, xcenter, ycenter, params):
        """ compute model ePSF values on the 2D grid X, Y for a single source

        Args:
            X, Y: meshgrids at which model values are evaluated
            xcenter, ycenter: star coordinates
            params: values of the ePSF model (flattened 1D)

        Returns:
            interpolated ePSF values on the input 2D grid

        """
        xidx = (X - xcenter - self.xgrid_center[0]) / self.dx
        yidx = (Y - ycenter - self.ygrid_center[0]) / self.dy
        Z = params.reshape(self.Nx, self.Ny)
        return map_coordinates(Z, [xidx, yidx], order=1)

    @partial(jit, static_argnums=(0, ))
    def get_obs1d(self, norms, xcenters, ycenters, X1d, Y1d, params):
        """ get shifted and scaled ePSFs for multiple sources

        Args:
            norms: total fluxes of the sources
            xcenters: x coordinates of the sources
            ycenters: y coordinates of the sources
            X1d: x coordinates of the output grid (flattened 1D)
            Y1d: y coordinates of the output grid (flattened 1D)
            params: values of the ePSF model (flattened 1D)

        Returns:
            flux on the output grid (flattened 1D)

        """
        epsfvalues_vmap_sources = vmap(self.evaluate_ePSF,
                                       (None, None, 0, 0, None), 0)
        ims_obs1d = epsfvalues_vmap_sources(X1d, Y1d, xcenters, ycenters,
                                            params)
        return jnp.sum(norms[:, None] * ims_obs1d, axis=0)

    @partial(jit, static_argnums=(0, ))
    def U_matrix(self, norms, xcenters, ycenters, X1d, Y1d):
        """ matrix U to convert the 1D ePSF parameters into 1D model flux

        Args:
            norms: total fluxes of the sources
            xcenters: x coordinates of the sources
            ycenters: y coordinates of the sources
            X1d: x coordinates of the output grid (flattened 1D)
            Y1d: y coordinates of the output grid (flattened 1D)

        Returns:
            (# of output grid) x (# of ePSF parameters) matrix

        """
        get_obs1d_vmap = vmap(self.get_obs1d,
                              (None, None, None, None, None, 0), 1)
        return get_obs1d_vmap(norms, xcenters, ycenters, X1d, Y1d, self.eye)

    def log_likelihood(self, fluxes, xcenters, ycenters, lenx, leny, amp2,
                       mupsf, obsX1d, obsY1d, obsZ1d, obserr1d):
        """ marginal log likelihood to obtain data given the source parameters and ePSF hyperparameters

        Args:
            norms, xcenters, ycenters: total fluxes/x-coordinates/y-coordinates of the sources
            lenx, leny: scale lengths in the x/y directions for the ePSF covariance
            amp2: squared amplitude for the ePSF covariance
            mupsf: mean value of the ePSF
            obsX1d, obsY1d: x/y coordinates of the observed image grid (1D flattened)
            obsZ1d, obserr1d: observed flux values and errors (1D flattened)

        Returns:
            log-likelihood to obtain the observed image marginalized over the ePSF parameters

        """
        cov_f = gpkernel(self.X1d, self.Y1d, lenx, leny, amp2)
        cov_d = jnp.diag(obserr1d**2)
        U = self.U_matrix(fluxes, xcenters, ycenters, obsX1d, obsY1d)

        mean = jnp.dot(U, mupsf * jnp.ones_like(self.X1d))
        cov = jnp.dot(U, jnp.dot(cov_f, U.T)) + cov_d
        mv = dist.MultivariateNormal(loc=mean, covariance_matrix=cov)

        return mv.log_prob(obsZ1d)

    def predict_mean(self, fluxes, xcenters, ycenters, lenx, leny, amp2, mupsf,
                     obsX1d, obsY1d, obsZ1d, obserr1d):
        """ predict mean ePSF and image conditioned on the data

        Args:
            same as above

        Returns:
            mean prediction for the ePSF (1D) and the true image (1D)

        """
        cov_f = gpkernel(self.X1d, self.Y1d, lenx, leny, amp2)
        cov_d = jnp.diag(obserr1d**2)
        U = self.U_matrix(fluxes, xcenters, ycenters, obsX1d, obsY1d)

        Sigma_Sfinv = jnp.eye(
            self.size) - cov_f @ U.T @ jnp.linalg.inv(cov_d +
                                                      U @ cov_f @ U.T) @ U
        Sigma_pred = Sigma_Sfinv @ cov_f
        prec_d = jnp.diag(1. / obserr1d**2)
        epsf_pred = Sigma_pred @ U.T @ prec_d @ obsZ1d + Sigma_Sfinv @ (
            mupsf * jnp.ones(self.size))
        image_pred = U @ epsf_pred

        return epsf_pred, image_pred


def gpkernel(X1d, Y1d, lenx, leny, amp2):
    """ 2D squared-exponential kernel """
    dx = X1d[:, None] - X1d[None, :]
    dy = Y1d[:, None] - Y1d[None, :]
    dx2 = jnp.power(dx / lenx, 2.0)
    dy2 = jnp.power(dy / leny, 2.0)
    cov = amp2 * jnp.exp(-0.5 * dx2 - 0.5 * dy2)
    return cov
