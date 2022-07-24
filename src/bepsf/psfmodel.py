__all__ = ["GridPSFModel"]

import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
from jax import jit
from functools import partial
from jax import vmap


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

    @partial(jit, static_argnums=(0,))
    def psfvalues1d(self, X, Y, xcenter, ycenter, params):
        """ same as psfvalues when reshaped to len(X),len(Y) """
        values1d = self.psfvalues(X, Y, xcenter, ycenter, params).ravel()
        return values1d

    @partial(jit, static_argnums=(0,))
    def translation_matrix(self, X, Y, norms, xcenters, ycenters):
        psfvalues1d_vmap = vmap(self.psfvalues1d, (None,None,None,None,0), 1) # map along params axis
        func_tmatrix = vmap(psfvalues1d_vmap, (None,None,0,0,None), 0) # map along xcenter, ycenter
        Ts = norms[:,None,None] * func_tmatrix(X, Y, xcenters, ycenters, self.eye) # Nsource, Nsuperpix, Npsfpix
        return jnp.sum(Ts, axis=0)
