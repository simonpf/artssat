import pytest
import numpy as np

from artssat.scattering.psd.data.psd_data import SizeParameter, PSDData

################################################################################
# Fixtures
################################################################################

@pytest.fixture
def exponential_distribution():
    """
    Generate data PSD data from a randomly sampled exponential
    distribution.

    Returns:
        (x, y): Tuple containing the size parameter :code:`x` and
        the PSD data :code:`y`.
    """
    n = 10000
    x = np.linspace(10.0 / n, 10, n).reshape(1, -1)
    lmd = np.random.uniform(0.5, 1.5, size = 10).reshape(-1, 1)
    y = lmd * np.exp(- lmd * x)
    return x, y

@pytest.fixture
def two_size_parameters():
    """
    Randomly generates two sets of parameters of a mass-size relationship and
    creates corresponding :code:`SizeParameter` objects.

    Returns:
        (s1, s2): The two randomly generated size parameters.
    """

    a = np.random.uniform(0.5, 10.0)
    b = np.random.uniform(1.0, 3.0)
    s1 = SizeParameter(a, b)

    a = np.random.uniform(0.5, 10.0)
    b = np.random.uniform(1.0, 3.0)
    s2 = SizeParameter(a, b)

    return s1, s2

################################################################################
# Tests
################################################################################

def test_size_parameter(two_size_parameters, exponential_distribution):
    """
    Test conversion between size parameters by ensuring that the mass
    of the PSD is conserved.
    """
    s1, s2 = two_size_parameters
    x, y = exponential_distribution

    xc, yc = s1.convert(s2, x, y)

    m1 = s1.get_mass_density(xc, yc)
    m2 = s2.get_mass_density(x, y)

    assert np.all(np.isclose(s1.get_mass_density(xc, yc),
                             s2.get_mass_density(x, y), rtol = 1e-2))

    psd = PSDData(x, y, s1)

def test_moments(two_size_parameters, exponential_distribution):
    """
    Test the computation of moments w.r.t to different size parameters.
    Conversion of the PSD must not change the moments of the distribution
    if the reference size parameter stays the same.
    """
    s1, s2 = two_size_parameters
    x, y = exponential_distribution

    psd1 = PSDData(x, y, s1)
    psd2 = PSDData(*s2.convert(s1, x, y), s2)

    for p in range(5):
        m1 = psd1.get_moment(p)
        m2 = psd2.get_moment(p, reference_size_parameter = psd1.size_parameter)

        print(m1)
        print(m2)
        assert np.all(np.isclose(m1, m2, rtol = 1e-3))


def test_psd_data(two_size_parameters, exponential_distribution):
    """
    Tests computation of mass and number density with randomly generated
    PSD data that is converted from one size parameter to another. This
    must not change the mass and number density.
    """
    x, y = exponential_distribution
    s1, s2 = two_size_parameters

    data = PSDData(x, y, s1)
    n1 = data.get_number_density()
    m1 = data.get_mass_density()

    data.change_size_parameter(s2)
    n2 = data.get_number_density()
    m2 = data.get_mass_density()

    print(n1, n2)
    print(m1, m2)

    assert(np.all(np.isclose(n1, n2, rtol = 1e-3)))
    assert(np.all(np.isclose(m1, m2, rtol = 1e-3)))

def test_psd_add(exponential_distribution, two_size_parameters):
    """
    Test addition of PSD data: Adding a PSDData object to itself
    should double all elements in :code:`data`.
    """
    x, y = exponential_distribution
    s1, s2 = two_size_parameters

    data = PSDData(x, y, s1)
    data2 = data + data

    assert np.all(np.isclose(2.0 * data.data, data2.data))

