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
        """ define ePSF model on a supersampled grid
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
        self.pixarea = (self.xgrid_edge.max()-self.xgrid_edge.min())*(self.ygrid_edge.max()-self.ygrid_edge.min())

        print ("PSF grid shape:", self.shape)
        #print ("x centers:", self.xgrid_center)
        #print ("y centers:", self.ygrid_center)
        print ("grid edge: x=[%f, %f], y=[%f, %f]"%(self.xgrid_edge[0], self.xgrid_edge[-1], self.ygrid_edge[0], self.ygrid_edge[-1]))
        print ("grid center: x=%f, y=%f"%(np.median(self.xgrid_center), np.median(self.ygrid_center)))

    @partial(jit, static_argnums=(0,))
    def evaluate_ePSF(self, X, Y, xcenter, ycenter, params):
        """ model ePSF values on grid X,Y
        Args:
            X, Y: meshgrids on which model values are computed
            xcenter, ycenter: star coodinrates
            params: pixel values of the PSF model on supersampled grid
        """
        xidx = (X - xcenter - self.xgrid_center[0]) / self.dx
        yidx = (Y - ycenter - self.ygrid_center[0]) / self.dy
        Z = params.reshape(self.Nx, self.Ny)
        return map_coordinates(Z, [xidx, yidx], order=1) #/ self.ds
    
    """
    @partial(jit, static_argnums=(0,4,))
    def get_obs1d(self, norms, xcenters, ycenters, image_obs, params):
        epsfvalues_vmap_sources = vmap(self.evaluate_ePSF, (None,None,0,0,None), 0)
        ims_obs = epsfvalues_vmap_sources(image_obs.X, image_obs.Y, xcenters, ycenters, params)
        return jnp.sum(norms[:,None,None]*ims_obs, axis=0).ravel()

    @partial(jit, static_argnums=(0,4,))
    def U_matrix(self, norms, xcenters, ycenters, image_obs):
        get_obs1d_vmap = vmap(self.get_obs1d, (None,None,None,None,0), 1)
        return get_obs1d_vmap(norms, xcenters, ycenters, image_obs, self.eye)
        
    def gp_marginal(self, fluxes, xcenters, ycenters, lenx, leny, amp2, image_obs, image_err, return_pred=False):
        mask1d = image_obs.mask1d
        
        cov_f = gpkernel(self.X1d, self.Y1d, lenx, leny, amp2)
        Unomask = self.U_matrix(fluxes, xcenters, ycenters, image_obs)
        U = Unomask[~mask1d]
        
        image_err1d = image_err.ravel()[~mask1d]
        cov_d = jnp.diag(image_err1d**2)
        obs1d = image_obs.Z1d[~mask1d]

        if return_pred:
            # mean prediction for the PSF & image vectors
            Sigma_pred = cov_f - cov_f@U.T@jnp.linalg.inv(cov_d+U@cov_f@U.T)@U@cov_f
            prec_d = jnp.diag(1./image_err1d**2)
            epsf_pred = Sigma_pred@U.T@prec_d@obs1d
            return epsf_pred, Unomask@epsf_pred #Sigma_pred

        cov = jnp.dot(U, jnp.dot(cov_f, U.T)) + cov_d
        mv = dist.MultivariateNormal(loc=0., covariance_matrix=cov)
        return mv.log_prob(obs1d)
    """
    
    @partial(jit, static_argnums=(0,))
    def get_obs1d(self, norms, xcenters, ycenters, X1d, Y1d, params):
        epsfvalues_vmap_sources = vmap(self.evaluate_ePSF, (None,None,0,0,None), 0)
        ims_obs1d = epsfvalues_vmap_sources(X1d, Y1d, xcenters, ycenters, params)
        return jnp.sum(norms[:,None]*ims_obs1d, axis=0)

    @partial(jit, static_argnums=(0,))
    def U_matrix(self, norms, xcenters, ycenters, X1d, Y1d):
        get_obs1d_vmap = vmap(self.get_obs1d, (None,None,None,None,None,0), 1)
        return get_obs1d_vmap(norms, xcenters, ycenters, X1d, Y1d, self.eye)

    def log_likelihood(self, fluxes, xcenters, ycenters, lenx, leny, amp2, mupsf, obsX1d, obsY1d, obsZ1d, obserr1d):
        cov_f = gpkernel(self.X1d, self.Y1d, lenx, leny, amp2)
        cov_d = jnp.diag(obserr1d**2)
        U = self.U_matrix(fluxes, xcenters, ycenters, obsX1d, obsY1d)
        
        mean = jnp.dot(U, mupsf*jnp.ones_like(self.X1d))
        cov = jnp.dot(U, jnp.dot(cov_f, U.T)) + cov_d
        mv = dist.MultivariateNormal(loc=mean, covariance_matrix=cov)
        
        return mv.log_prob(obsZ1d)
    
    def predict_mean(self, fluxes, xcenters, ycenters, lenx, leny, amp2, mupsf, obsX1d, obsY1d, obsZ1d, obserr1d):
        cov_f = gpkernel(self.X1d, self.Y1d, lenx, leny, amp2)
        cov_d = jnp.diag(obserr1d**2)
        U = self.U_matrix(fluxes, xcenters, ycenters, obsX1d, obsY1d)
        
        #Sigma_pred = cov_f - cov_f@U.T@jnp.linalg.inv(cov_d+U@cov_f@U.T)@U@cov_f
        Sigma_Sfinv = jnp.eye(self.size) - cov_f@U.T@jnp.linalg.inv(cov_d+U@cov_f@U.T)@U
        Sigma_pred = Sigma_Sfinv@cov_f
        prec_d = jnp.diag(1./obserr1d**2)
        epsf_pred = Sigma_pred@U.T@prec_d@obsZ1d + Sigma_Sfinv@(mupsf*jnp.ones(self.size))
        image_pred = U@epsf_pred
        
        return epsf_pred, image_pred
        
def gpkernel(X1d, Y1d, lenx, leny, amp2):
    dx = X1d[:,None] - X1d[None,:]
    dy = Y1d[:,None] - Y1d[None,:]
    dx2 = jnp.power(dx / lenx, 2.0)
    dy2 = jnp.power(dy / leny, 2.0)
    cov = amp2 * jnp.exp(-0.5*dx2-0.5*dy2)
    return cov


"""
def log_likelihood(self, fluxes, xcenters, ycenters, lenx, leny, amp2, obsX1d, obsY1d, obsZ1d, obserr1d):
    cov_f = gpkernel(self.X1d, self.Y1d, lenx, leny, amp2)
    cov_d = jnp.diag(obserr1d**2)
    U = self.U_matrix(fluxes, xcenters, ycenters, obsX1d, obsY1d)

    cov = jnp.dot(U, jnp.dot(cov_f, U.T)) + cov_d
    mv = dist.MultivariateNormal(loc=0., covariance_matrix=cov)

    return mv.log_prob(obsZ1d)
        
def gp_marginal(self, fluxes, xcenters, ycenters, lenx, leny, amp2, image_obs, image_err, return_pred=False):
        mask1d = image_obs.mask1d
        
        cov_f = gpkernel(self.X1d, self.Y1d, lenx, leny, amp2)
        U = self.U_matrix(fluxes, xcenters, ycenters, image_obs.X1d[~mask1d], image_obs.Y1d[~mask1d])
        
        image_err1d = image_err.ravel()[~mask1d]
        cov_d = jnp.diag(image_err1d**2)
        obs1d = image_obs.Z1d[~mask1d]

        if return_pred:
            # mean prediction for the PSF & image vectors
            Sigma_pred = cov_f - cov_f@U.T@jnp.linalg.inv(cov_d+U@cov_f@U.T)@U@cov_f
            prec_d = jnp.diag(1./image_err1d**2)
            epsf_pred = Sigma_pred@U.T@prec_d@obs1d
            image_pred = np.zeros(image_obs.size)
            image_pred[~mask1d] = U@epsf_pred
            return epsf_pred, image_pred#Unomask@mu_pred #Sigma_pred

        cov = jnp.dot(U, jnp.dot(cov_f, U.T)) + cov_d
        mv = dist.MultivariateNormal(loc=0., covariance_matrix=cov)
        return mv.log_prob(obs1d)

        # same but slower
        #SinvZ = jnp.linalg.solve(cov, image_obs.Z1d)
        #return -0.5 * jnp.linalg.slogdet(cov)[1] \
        #    - 0.5 * jnp.dot(image_obs.Z1d.T, SinvZ) - 0.5 * image_obs.size * jnp.log(2*jnp.pi)
"""
