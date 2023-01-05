#%%
import pytest
import numpy as np
from bepsf.image import PixelImage

#%%
def test_image_shape():
    Npix = 50
    im = PixelImage(Npix, Npix)
    assert im.shape == (Npix, Npix)
    assert im.X.shape == (Npix, Npix)
    assert im.Y.shape == (Npix, Npix)
    assert im.mask.sum() == 0
    assert im.X1d.shape == (Npix*Npix,)
    assert im.Y1d.shape == (Npix*Npix,)
    assert im.Z is None

def test_mask():
    Npix, Nsource = 50, 10
    limit_dist = 3.5
    im = PixelImage(Npix, Npix)
    np.random.seed(123)
    xcenters = np.random.rand(Nsource) * im.xmax
    ycenters = np.random.rand(Nsource) * im.ymax
    im.define_mask(xcenters, ycenters, 3.5)
    assert np.sum(im.mask) == 2160
    assert im.mask.shape == (Npix,Npix)
    assert im.mask1d.shape == (Npix*Npix,)

#%%
if __name__ == '__main__':
    test_image_shape()
    test_mask()
