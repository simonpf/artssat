import numpy as np
import pytest

from parts.scattering.psd.data.psd_data import Mass
from parts.scattering.psd.modified_gamma import ModifiedGamma

################################################################################
# Fixtures
################################################################################

@pytest.fixture(scope = "module")
def random_mgd_psd():
    """
    Generates a random instance of the modified gamma distribution.
    """
    n     = 10.0 ** np.random.uniform(5, 8, size = (10, 1))
    alpha = np.random.uniform(1, 3, size = (10, 1))
    lmbd  = np.random.uniform(1, 3, size = (10, 1))
    nu    = np.random.uniform(1, 3, size = (10, 1))
    return ModifiedGamma(Mass(), n, alpha, lmbd, nu)

################################################################################
# Tests
################################################################################

def test_moments(random_mgd_psd):
    """
    Tests the implementation of the modified gamma distribution by computing
    the first two moments once using the analytic formulas and once using
    numeric PSDData.
    """

    psd = random_mgd_psd

    x = np.logspace(-4, 2, 100000)
    psd_data = psd.evaluate(x)

    m0     = psd.get_moment(0)
    m0_ref = psd_data.get_moment(0)

    assert np.all(np.isclose(m0, m0_ref))

    m1     = psd.get_moment(1)
    m1_ref = psd_data.get_moment(1)

    assert np.all(np.isclose(m1, m1_ref))

    m2     = psd.get_moment(2)
    m2_ref = psd_data.get_moment(2)

    m     = psd.get_mass_density()
    m_ref = psd_data.get_mass_density()

    assert np.all(np.isclose(m, m_ref))
