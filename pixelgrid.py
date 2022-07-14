__all__ = ["PixelGrid"]

import jax.numpy as jnp

class PixelGrid:
    def __init__(self, xmin, xmax, ymin, ymax, dx=1., dy=1.):
        """initialization
        Args:
           xmin, xmax, ymin, ymax: x/y coordinates at the grid edge
           dx, dy: grid spacing
        """
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.Nx = int((xmax - xmin) / dx)
        self.Ny = int((ymax - ymin) / dy)
        self.xgrid_edge = jnp.linspace(xmin, xmax, self.Nx+1)
        self.ygrid_edge = jnp.linspace(ymin, ymax, self.Ny+1)
        self.xgrid_center = 0.5 * (self.xgrid_edge[1:] + self.xgrid_edge[:-1])
        self.ygrid_center = 0.5 * (self.ygrid_edge[1:] + self.ygrid_edge[:-1])

        X, Y = jnp.meshgrid(self.xgrid_center, self.ygrid_center)
        self.X = X
        self.Y = Y
        self.X1d = jnp.tile(self.xgrid_center, self.Ny)
        self.Y1d = jnp.repeat(self.ygrid_center, self.Nx)
