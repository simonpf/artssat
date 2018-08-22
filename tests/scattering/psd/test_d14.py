import numpy as np
import pytest

from parts.scattering.psd import D14, D14N

################################################################################
# Fixtures
################################################################################

@pytest.fixture(scope = "module", params = [(-8, -4, D14), (8, 10, D14N)])
def random_d14_psd(request):
    """
    Create a random instance of the D14 PSD.
    """
    low, high, cls = request.param
    m1 = 10 ** np.random.uniform(low, high, size = (10, 1))
    m2 = 1e-6 * np.random.uniform(100, 200, size = (10, 1))
    alpha = np.random.uniform(-1, 1)
    beta  = np.random.uniform(1, 3)
    return cls(alpha, beta, 917.0, m1, m2)

################################################################################
# Tests
################################################################################

def test_d14(random_d14_psd):
    """
    Computes mass and the first two moments of the PSD once using the analytic
    formulas and once using numeric PSD data.
    """
    psd = random_d14_psd
    x = np.logspace(-8, -3, 100000)
    psd_data = psd.evaluate(x)

    m     = psd.get_mass_density()
    m_ref = psd_data.get_mass_density()

    assert np.all(np.isclose(m, m_ref, rtol = 1e-3))

    m1     = psd.get_moment(1)
    m1_ref = psd_data.get_moment(1)
    print(psd_data.data)
    print(m1)
    print(m1_ref)

    assert np.all(np.isclose(m1, m1_ref, rtol = 1e-3))

    m2     = psd.get_moment(2)
    m2_ref = psd_data.get_moment(2)
    print(psd_data.data)
    print(m1)
    print(m1_ref)

    assert np.all(np.isclose(m2, m2_ref, rtol = 1e-3))

def test_d14_from_psd_data(random_d14_psd):
    """
    Test that a D14(N) PSD created from an existing one remains
    the same.
    """
    psd_ref = random_d14_psd
    psd = type(psd_ref).from_psd_data(psd_ref,
                                      psd_ref.alpha,
                                      psd_ref.beta,
                                      psd_ref.rho)

    m     = psd.get_mass_density()
    m_ref = psd_ref.get_mass_density()
    assert np.all(np.isclose(psd.get_mass_density(),
                             psd_ref.get_mass_density()))

    assert np.all(np.isclose(psd.mass_weighted_diameter,
                             psd_ref.mass_weighted_diameter))