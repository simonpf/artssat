import numpy as np
import pytest

from parts.scattering.psd import D14, D14N, D14MN
from parts.scattering.psd import MY05

################################################################################
# Fixtures
################################################################################

@pytest.fixture(scope = "module", params = [D14, D14N, D14MN])
def random_d14_psd(request):
    """
    Create a random instance of the D14 PSD.
    """

    boundaries = {"intercept_parameter" : (6, 12),
                  "mass_weighted_diameter" : (-2, -4),
                  "mass_density" : (-12, -2)}

    cls = request.param
    psd = cls(1.0, 1.0, 917.0)

    low, high = boundaries[psd.moment_names[0]]
    m1 = 10 ** np.random.uniform(low, high, size = (10, 1))

    low, high = boundaries[psd.moment_names[1]]
    m2 = 10 ** np.random.uniform(low, high, size = (10, 1))

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
    x = np.logspace(-10, -1, 100000)
    psd_data = psd.evaluate(x)

    m     = psd.get_mass_density()
    m_ref = psd_data.get_mass_density()

    print(m)
    print(m_ref)

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

def test_conversion_d14_my05(random_d14_psd):
    """
    Test conversion between D14 and MY05. Conversion back and forth
    must conserve predictive moments.
    """
    d14_ref = random_d14_psd
    d14_cls = type(d14_ref)
    a = 2 * d14_ref.size_parameter.a
    b = d14_ref.size_parameter.b

    my05 = MY05.from_psd_data(d14_ref, d14_ref.alpha, d14_ref.beta, a, b)
    print(d14_ref.get_moment(0))

    d14 =  d14_cls.from_psd_data(my05, d14_ref.alpha, d14_ref.beta, d14_ref.rho)
    print(d14.get_moment(0))

    for m1, m2 in zip(d14_ref.moments, d14.moments):
        assert np.all(np.isclose(m1, m2))


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
