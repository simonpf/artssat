import numpy as np
import pytest

from artssat.scattering.psd.ab12 import AB12

################################################################################
# Fixtures
################################################################################

@pytest.fixture(scope = "module")
def random_ab12_psd():
    """
    Generates a random instance of the MY05 distribution.
    """
    mu = np.random.uniform(1, 2)
    m     = 10.0 ** np.random.uniform(-8, -4, size = (10, 1))
    return AB12(mu, m)

################################################################################
# Tests
################################################################################

def test_moments(random_ab12_psd):
    """
    Tests the implementation of the MY05 distribution by computing
    the first two moments once and the mass density using the analytic
    formulas and using numeric PSDData.
    """
    psd = random_ab12_psd

    x = np.logspace(-8, -1, 100000)
    psd_data = psd.evaluate(x)


    m0     = psd.get_moment(0)
    m0_ref = psd_data.get_moment(0)

    assert np.all(np.isclose(m0, m0_ref, rtol = 1e-2))

    m1     = psd.get_moment(1)
    m1_ref = psd_data.get_moment(1)

    assert np.all(np.isclose(m1, m1_ref, rtol = 1e-3))

    m2     = psd.get_moment(2)
    m2_ref = psd_data.get_moment(2)

    assert np.all(np.isclose(m2, m2_ref, rtol = 1e-3))

    m3     = psd.get_moment(3)
    m3_ref = psd_data.get_moment(3)

    assert np.all(np.isclose(m3, m3_ref, rtol = 1e-3))

    m     = psd.get_mass_density()
    m_ref = psd_data.get_mass_density()
    assert np.all(np.isclose(m, m_ref, rtol = 1e-2))
