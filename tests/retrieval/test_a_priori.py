from IPython import get_ipython
ip = get_ipython()
if not ip is None:
    ip.magic("%load_ext autoreload")
    ip.magic("%autoreload 2")

import numpy as np
import pytest
from examples.data_provider import DataProvider
from parts.retrieval.a_priori import Diagonal, SpatialCorrelation, Thikhonov, \
    TemperatureMask, TropopauseMask, And, DataProviderAPriori, FixedAPriori

def test_masks():
    data_provider    = DataProvider()
    temperature = data_provider.get_temperature()

    temperature_mask = TemperatureMask(lower_limit = 230.0,
                                    upper_limit = 280.0)
    mask = temperature_mask(data_provider)
    assert(np.all(temperature[mask] >= 230))
    assert(np.all(temperature[mask] < 280))

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

    mask = temperature_mask(data_provider)
    assert(np.all(precmat.diagonal()[mask] >= 1e12))
    mask2 = np.logical_not(mask)[2 : -2]
    assert(np.all(precmat.diagonal()[2:-2][mask2] == 6))

def test_data_provider_a_priori():
    temperature_mask = TemperatureMask(lower_limit = 230.0,
                                       upper_limit = 280.0)
    tropopause_mask = TropopauseMask()
    covariance = Diagonal(2.0, And(temperature_mask, tropopause_mask))
    data_provider    = DataProvider()
    data_provider.add(DataProviderApriori("temperature", covariance))

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
