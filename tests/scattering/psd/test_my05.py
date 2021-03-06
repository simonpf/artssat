import numpy as np
import pytest

from artssat.scattering.psd.my05 import MY05

################################################################################
# Fixtures
################################################################################

@pytest.fixture(scope = "module")
def random_my05_psd():
    """
    Generates a random instance of the MY05 distribution.
    """
    mu = np.random.uniform(1, 3)
    nu = np.random.uniform(1, 3)
    a     = np.random.uniform(1.0, 3.0)
    b     = np.random.uniform(2, 3)
    m     = 10.0 ** np.random.uniform(-8, -4, size = (10, 1))
    n     = 10.0 ** np.random.uniform(5, 8, size = (10, 1))
    return MY05(mu, nu, a, b, None, n, m)

################################################################################
# Tests
################################################################################

def test_moments(random_my05_psd):
    """
    Tests the implementation of the MY05 distribution by computing
    the first two moments once and the mass density using the analytic
    formulas and using numeric PSDData.
    """
    psd = random_my05_psd

    x = np.logspace(-8, -1, 100000)
    psd_data = psd.evaluate(x)

    m0     = psd.get_moment(0)
    m0_ref = psd_data.get_moment(0)

    assert np.all(np.isclose(m0, m0_ref, rtol = 1e-1))

    m1     = psd.get_moment(1)
    m1_ref = psd_data.get_moment(1)

    assert np.all(np.isclose(m1, m1_ref, rtol = 1e-1))

    m2     = psd.get_moment(2)
    m2_ref = psd_data.get_moment(2)

    assert np.all(np.isclose(m2, m2_ref, rtol = 1e-1))

    m3     = psd.get_moment(3)
    m3_ref = psd_data.get_moment(3)

    assert np.all(np.isclose(m3, m3_ref, rtol = 1e-1))

    m4     = psd.get_moment(4)
    m4_ref = psd_data.get_moment(4)

    assert np.all(np.isclose(m4, m4_ref, rtol = 1e-1))

    m     = psd.mass_density
    m_ref = psd_data.get_mass_density()

    assert np.all(np.isclose(m, m_ref, rtol = 1e-1))
