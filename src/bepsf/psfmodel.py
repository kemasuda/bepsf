__all__ = ["GridPSFModel", "super_to_obs"]

import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
from jax import jit
from functools import partial
from jax import vmap
import numpyro.distributions as dist


class GridPSFModel:
    def __init__(self, x_extent, y_extent, dx, dy):
        """ define PSF model on a supersampled grid
        Args:
           x_extent, y_extent: size of the model PSF grid (in units of observed pixels)
           dx, dy: supersampled grid spacing (in units of observed pixels)
        """
        Nx = int(x_extent / dx // 2)*2 + 1
        Ny = int(y_extent / dy // 2)*2 + 1
        xgrid_edge = jnp.linspace(-0.5*dx, (Nx-0.5)*dx, Nx+1)
        ygrid_edge = jnp.linspace(-0.5*dy, (Ny-0.5)*dy, Ny+1)
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

        print ("PSF grid shape:", self.shape)
        print ("x centers:", self.xgrid_center)
        print ("y centers:", self.ygrid_center)

    @partial(jit, static_argnums=(0,))
    def psfvalues(self, X, Y, xcenter, ycenter, params):
        """ model PSF values on grid X,Y
        Args:
            X, Y: meshgrids on which model values are computed
            xcenter, ycenter: star coodinrates
            params: pixel values of the PSF model on supersampled grid
        """
        xidx = (X - xcenter - self.xgrid_center[0]) / self.dx
        yidx = (Y - ycenter - self.ygrid_center[0]) / self.dy
        Z = params.reshape(self.Nx, self.Ny)
        return map_coordinates(Z, [xidx, yidx], order=1)
    
    @partial(jit, static_argnums=(0,4,5))
    def get_obs1d(self, norms, xcenters, ycenters, image_obs, image_super, params):
        psfvalues_vmap_sources = vmap(self.psfvalues, (None,None,0,0,None), 0)
        #ims_super = psfvalues_vmap_sources(image_super.X, image_super.Y, xcenters, ycenters, params)
        #im_super = jnp.sum(norms[:,None,None] * ims_super, axis=0)
        mask1d = image_super.mask1d
        ims_super1d_unmasked = jnp.sum(norms[:,None]*psfvalues_vmap_sources(image_super.X1d[~mask1d], image_super.Y1d[~mask1d], xcenters, ycenters, params), axis=0)
        im_super = jnp.zeros(len(mask1d)).at[~mask1d].set(ims_super1d_unmasked)
        im_super = im_super.reshape(image_super.shape)
        im_obs1d = super_to_obs(im_super, image_obs).ravel()
        return im_obs1d

    @partial(jit, static_argnums=(0,4,5))
    def U_matrix(self, norms, xcenters, ycenters, image_obs, image_super):
        get_obs1d_vmap = vmap(self.get_obs1d, (None,None,None,None,None,0), 1)
        return get_obs1d_vmap(norms, xcenters, ycenters, image_obs, image_super, self.eye)
    
    def gp_marginal(self, fluxes, xcenters, ycenters, lenx, leny, amp2, 
                image_obs, image_err, image_super, return_pred=False):
        mask1d = image_obs.mask1d
        
        cov_f = gpkernel(self.X1d, self.Y1d, lenx, leny, amp2)
        Unomask = self.U_matrix(fluxes, xcenters, ycenters, image_obs, image_super)
        U = Unomask[~mask1d]
        #cov_d = sigma_err**2 * jnp.eye(image_obs.size)
        image_err1d = image_err.ravel()[~mask1d]
        cov_d = jnp.diag(image_err1d**2)
        obs1d = image_obs.Z1d[~mask1d]

        if return_pred:
            """ mean prediction for the PSF & image vectors """
            Sigma_pred = cov_f - cov_f@U.T@jnp.linalg.inv(cov_d+U@cov_f@U.T)@U@cov_f
            #prec_d = 1. / sigma_err**2 * jnp.eye(image_obs.size)
            prec_d = jnp.diag(1./image_err1d**2)
            mu_pred = Sigma_pred@U.T@prec_d@obs1d
            return mu_pred, Unomask@mu_pred #Sigma_pred

        cov = jnp.dot(U, jnp.dot(cov_f, U.T)) + cov_d
        mv = dist.MultivariateNormal(loc=0., covariance_matrix=cov)
        return mv.log_prob(obs1d)

        # same but slower
        #SinvZ = jnp.linalg.solve(cov, image_obs.Z1d)
        #return -0.5 * jnp.linalg.slogdet(cov)[1] \
        #    - 0.5 * jnp.dot(image_obs.Z1d.T, SinvZ) - 0.5 * image_obs.size * jnp.log(2*jnp.pi)

def super_to_obs(Z_super, Z_obs):
    # convert 2d supersampled image Z_super (Ms,Ns) to 2d undersampled image Z_obs (Mobs,Nobs)
    Ms, Ns = Z_super.shape
    Mobs, Nobs = Z_obs.shape
    K, L = Ms // Mobs, Ns // Nobs
    return Z_super[:Mobs*K, :Nobs*L].reshape(Mobs, K, Nobs, L).sum(axis=(1, 3))
        
def gpkernel(X1d, Y1d, lenx, leny, amp2):
    dx = X1d[:,None] - X1d[None,:]
    dy = Y1d[:,None] - Y1d[None,:]
    dx2 = jnp.power(dx / lenx, 2.0)
    dy2 = jnp.power(dy / leny, 2.0)
    cov = amp2 * jnp.exp(-0.5*dx2-0.5*dy2)
    return cov

"""
@partial(jit, static_argnums=(0,))
def psfvalues1d(self, X, Y, xcenter, ycenter, params):
    # same as psfvalues when reshaped to len(X),len(Y)
    values1d = self.psfvalues(X, Y, xcenter, ycenter, params).ravel()
    return values1d

# U@T is 1.5x slower than U_matrix for Nx,Ny=(30,30) and super_factor=3
@partial(jit, static_argnums=(0,))
def translation_matrix(self, X, Y, norms, xcenters, ycenters):
    psfvalues1d_vmap = vmap(self.psfvalues1d, (None,None,None,None,0), 1) # map along params axis
    func_tmatrix = vmap(psfvalues1d_vmap, (None,None,0,0,None), 0) # map along xcenter, ycenter
    Ts = norms[:,None,None] * func_tmatrix(X, Y, xcenters, ycenters, self.eye) # Nsource, Nsuperpix, Npsfpix
    return jnp.sum(Ts, axis=0)
    
def gp_marginal(gridpsf, p, p_anchor, idx_anchor, image_obs, sigma_err, image_super, S, return_pred=False):
    lnfluxes = jnp.r_[p['lnfluxes'][:idx_anchor], p_anchor['lnfluxes'], p['lnfluxes'][idx_anchor+1:]]
    xcenters = jnp.r_[p['xcenters'][:idx_anchor], p_anchor['xcenters'], p['xcenters'][idx_anchor+1:]]
    ycenters = jnp.r_[p['ycenters'][:idx_anchor], p_anchor['ycenters'], p['ycenters'][idx_anchor+1:]]
    lenx, leny, amp2 = jnp.exp(p['lnlenx']), jnp.exp(p['lnleny']), jnp.exp(2*p['lnamp'])
    cov_f = gpkernel(gridpsf.X1d, gridpsf.Y1d, lenx, leny, amp2)
    T = gridpsf.translation_matrix(image_super.X, image_super.Y, jnp.exp(lnfluxes), xcenters, ycenters)
    U = jnp.dot(S, T)
    cov_d = sigma_err**2 * jnp.eye(image_obs.size)
    
    if return_pred:
        # mean prediction for the PSF & image vectors
        Sigma_pred = cov_f - cov_f@U.T@jnp.linalg.inv(cov_d+U@cov_f@U.T)@U@cov_f
        prec_d = 1. / sigma_err**2 * jnp.eye(image_obs.size)
        mu_pred = Sigma_pred@U.T@prec_d@image_obs.Z1d
        return mu_pred, U@mu_pred #Sigma_pred
    
    cov = jnp.dot(U, jnp.dot(cov_f, U.T)) + cov_d
    mv = dist.MultivariateNormal(loc=0., covariance_matrix=cov)
    return mv.log_prob(image_obs.Z1d)
"""