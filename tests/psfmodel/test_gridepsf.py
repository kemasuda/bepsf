#%%
import pytest
import numpy as np
import pkg_resources
from bepsf.image import PixelImage
from bepsf.psfmodel import GridePSFModel

#%%
def model():
    psf_full_extent = 7.
    dx, dy = 1./5., 1./5.
    return GridePSFModel(psf_full_extent, psf_full_extent, dx, dy)

def test_gridepsf_shape():
    g = model()
    assert g.shape == (35, 35)
    assert g.xgrid_edge.min() == -3.5
    assert g.xgrid_edge.max() == 3.5
    assert g.ygrid_edge.min() == -3.5
    assert g.ygrid_edge.max() == 3.5
    assert g.pixarea == 49.
    assert g.size == 1225

def test_evaluate_psf():
    g = model()
    im = PixelImage(90,90)
    shifted_epsf = g.evaluate_ePSF(im.X, im.Y, 10.5, 50.3, np.ones(g.size))
    assert np.argmax(shifted_epsf) == 4237
    assert np.sum(shifted_epsf) == 49.

def test_loglikelihood():
    from jax.config import config
    config.update('jax_enable_x64', True)
    path = pkg_resources.resource_filename('bepsf', 'data/')
    data = np.load(path+"test_image.npz")
    Z, Zerr, fluxes, xcenters, ycenters = data['Z'], data['Zerr'], data['fluxes'], data['xcenters'], data['ycenters']
    im = PixelImage(Z.shape[0], Z.shape[1])
    im.Z = Z
    im.Zerr = Zerr
    g = model()
    lenx, leny = 1., 1.
    amp2 = np.exp(-4.*2)
    mupsf = 0.
    loglike = g.log_likelihood(fluxes, xcenters, ycenters, lenx, leny, amp2, mupsf, im.X1d, im.Y1d, im.Z1d, im.Zerr1d)
    assert loglike == pytest.approx(38638.943277)

#%%
if __name__ == '__main__':
    test_gridepsf_shape()
    test_evaluate_psf()
    test_loglikelihood()
