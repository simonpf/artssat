#from IPython import get_ipython
#ip = get_ipython()
#if not ip is None:
#    ip.magic("%load_ext autoreload")
#    ip.magic("%autoreload 2")

import numpy as np
import pytest
from examples.data_provider import DataProvider
from parts.sensor import ICI
from parts.retrieval.a_priori import Diagonal, SpatialCorrelation, Thikhonov, \
    TemperatureMask, TropopauseMask, And, DataProviderAPriori, FixedAPriori, \
    SensorNoiseAPriori, ReducedVerticalGrid, FunctionalAPriori, MaskedRegularGrid

def test_masks():
    data_provider    = DataProvider()
    temperature = data_provider.get_temperature()

    temperature_mask = TemperatureMask(lower_limit = 230.0,
                                    upper_limit = 280.0)
    mask = temperature_mask(data_provider)
    assert(np.all(temperature[mask] >= 230))
    assert(np.all(temperature[mask] < 280))

    tropopause_mask = TropopauseMask()
    mask = tropopause_mask(data_provider)
    assert(np.all(temperature[mask]) <= 225)

    mask = And(temperature_mask, tropopause_mask)(data_provider)
    assert(np.all(temperature[mask] >= 230))
    assert(np.all(temperature[mask] < 280))
    assert(np.all(temperature[mask]) <= 225)

def test_covariances():
    data_provider    = DataProvider()
    z = data_provider.get_altitude()
    diagonal = Diagonal(2)
    diag = diagonal.get_covariance(data_provider)
    assert(np.all(np.isclose(diag.diagonal(), 2.0 * np.ones(z.size))))

    gauss = SpatialCorrelation(diagonal, 1000.0)
    covmat = gauss.get_covariance(data_provider)
    assert(np.all(np.isclose(covmat.diagonal(), 2.0 * np.ones(z.size))))

    temperature_mask = TemperatureMask(lower_limit = 230.0,
                                       upper_limit = 280.0)
    thik = Thikhonov(scaling = 1.0, z_scaling = False, mask = temperature_mask)
    precmat = thik.get_precision(data_provider)

    mask = np.logical_not(temperature_mask(data_provider))
    assert(np.all(precmat.diagonal()[mask] >= 1e12))
    mask2 = np.logical_not(mask)[2 : -2]
    assert(np.all(precmat.diagonal()[2:-2][mask2] == 6))

def test_data_provider_a_priori():
    temperature_mask = TemperatureMask(lower_limit = 230.0,
                                       upper_limit = 280.0)
    tropopause_mask = TropopauseMask()
    covariance = Diagonal(2.0, And(temperature_mask, tropopause_mask))
    data_provider    = DataProvider()
    data_provider.add(DataProviderAPriori("temperature", covariance))

    t = data_provider.get_temperature()
    assert(np.all(data_provider.get_temperature_xa() == t))

def test_fixed_a_priori():
    temperature_mask = TemperatureMask(lower_limit = 230.0,
                                       upper_limit = 280.0)
    tropopause_mask = TropopauseMask()
    covariance = Diagonal(2.0, And(temperature_mask, tropopause_mask))
    data_provider    = DataProvider()
    t = data_provider.get_temperature()
    data_provider.add(FixedAPriori("temperature", t, covariance))

    assert(np.all(data_provider.get_temperature_xa() == t))

    return data_provider

def test_sensor_noise_a_priori():
    sna = SensorNoiseAPriori([ICI()])
    sna.noise_scaling["ici"] = 2.0
    covmat = sna.get_observation_error_covariance()
    assert(np.allclose((ICI.nedt * 2.0) ** 2.0, covmat.diagonal()))

def test_reduced_grid_a_priori():
    tropopause_mask  = TropopauseMask()
    temperature_mask = TemperatureMask(lower_limit = 230.0,
                                       upper_limit = 280.0)
    covariance       = Diagonal(2.0, And(temperature_mask, tropopause_mask))
    covariance_new   = Diagonal(2.0 * np.ones(10))

    data_provider    = DataProvider()
    t = data_provider.get_temperature()
    a_priori = FixedAPriori("temperature", t, covariance)
    a_priori = ReducedVerticalGrid(a_priori, np.logspace(3, 5, 10)[::-1])
    data_provider.add(a_priori)

    t_xa = data_provider.get_temperature_xa()
    assert(t_xa.size == 10)

    covmat = data_provider.get_temperature_covariance()
    assert(covmat.shape == (10, 10))

    data_provider    = DataProvider()
    a_priori = ReducedVerticalGrid(a_priori, np.logspace(3, 5, 10)[::-1],
                                   covariance = covariance_new)
    data_provider.add(a_priori)
    covmat = data_provider.get_temperature_covariance()
    assert(np.all(covmat.diagonal() == 2.0))

def test_reduced_grid_a_priori_altitude():
    tropopause_mask  = TropopauseMask()
    temperature_mask = TemperatureMask(lower_limit = 230.0,
                                       upper_limit = 280.0)
    covariance       = Diagonal(2.0, And(temperature_mask, tropopause_mask))
    covariance_new   = Diagonal(2.0 * np.ones(10))

    data_provider    = DataProvider()
    t = data_provider.get_temperature()
    a_priori = FixedAPriori("temperature", t, covariance)
    a_priori = ReducedVerticalGrid(a_priori, np.linspace(1, 10, 10), quantity = "altitude")
    data_provider.add(a_priori)

    t_xa = data_provider.get_temperature_xa()
    assert(t_xa.size == 10)

    covmat = data_provider.get_temperature_covariance()
    assert(covmat.shape == (10, 10))

    data_provider    = DataProvider()
    a_priori = ReducedVerticalGrid(a_priori, np.logspace(3, 5, 10)[::-1],
                                   covariance = covariance_new)
    data_provider.add(a_priori)
    covmat = data_provider.get_temperature_covariance()
    assert(np.all(covmat.diagonal() == 2.0))

    assert(data_provider.get_temperature_p_grid().size == 10)

def test_functional_a_priori():
    temperature_mask = TemperatureMask(lower_limit = 230.0,
                                        upper_limit = 280.0)
    tropopause_mask  = TropopauseMask()
    covariance       = Diagonal(2.0, And(temperature_mask, tropopause_mask))
    covariance_new   = Diagonal(2.0 * np.ones(10))

    data_provider    = DataProvider()
    t = data_provider.get_temperature()
    f = lambda x: x ** 2
    a_priori = FunctionalAPriori("temperature", "temperature", f, covariance)
    data_provider.add(a_priori)
    assert(np.all(np.isclose(t ** 2, data_provider.get_temperature_xa())))


